import datetime
import glob
import os
import time

import matplotlib.pyplot as plt
import monai
import numpy as np
import scipy.stats as stats
import torch
import torchio as tio
from PIL import Image
from tqdm import tqdm

import config
from data.train_transforms import ImplicitModelTransformFastd
from models.model_implicit import coords_to_image
from utils import utils_3D_image
from utils.load_options import load_json, parse_arguments, load_options_from_experiment_id
from utils.utils_image import calculate_ssim_2D, calculate_nrmse_2D, calculate_psnr_2D


def get_mean_and_ci(data_sequence, confidence=0.95):

    data = np.array(data_sequence)
    n = len(data)
    mean, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h

def get_idx(patch_size, img_shape, h, w, d, border, context_width):
    h_start = h * patch_size - 2 * h * (border + context_width)
    w_start = w * patch_size - 2 * w * (border + context_width)
    d_start = d * patch_size - 2 * d * (border + context_width)
    start = [h_start, w_start, d_start]

    h_stop = h * patch_size - 2 * h * (border + context_width) + patch_size
    w_stop = w * patch_size - 2 * w * (border + context_width) + patch_size
    d_stop = d * patch_size - 2 * d * (border + context_width) + patch_size
    stop = [h_stop, w_stop, d_stop]
    for i in range(3):
        if stop[i] > img_shape[i+2]:
            start[i] = (img_shape[i+2] - patch_size)
            stop[i] = img_shape[i+2]

    return start, stop

def get_idx_hr(patch_size, img_shape, h, w, d, border):
    h_start = (h * patch_size - 2 * h * border) + border
    w_start = (w * patch_size - 2 * w * border) + border
    d_start = (d * patch_size - 2 * d * border) + border
    start = [h_start, w_start, d_start]

    h_stop = (h * patch_size - 2 * h * border + patch_size) - border
    w_stop = (w * patch_size - 2 * w * border + patch_size) - border
    d_stop = (d * patch_size - 2 * d * border + patch_size) - border
    stop = [h_stop, w_stop, d_stop]
    for i in range(3):
        if stop[i] > img_shape[i+2]:
            start[i] = (img_shape[i+2] - patch_size) + border
            stop[i] = img_shape[i+2] - border

    return start, stop


def trim_background_slices(img_L, img_E, img_H, dataset_name, threshhold_percentage=0.20):
    # Experimental method for removing slices that only contain background
    # Set a simple threshold for the background. Any voxel larger will be considered brain/bone
    # A slice is discarded if it contains less can 20% foreground voxels.
    if dataset_name == "KIRBY21":
        lr_foreground_threshold = 0.02  # Kirby21 is T2w and therefore we need lower threshold
    elif dataset_name == "2022_QIM_52_Bone" or dataset_name == "Synthetic_2022_QIM_52_Bone":
        lr_foreground_threshold = 0.02  # Bone datasets are very sparse, they should have lower threshold
    elif dataset_name == "IXI":
        # Rotate such that scans are z-sliced, then flip to match orientation of other datasets
        img_L = torch.flip(img_L.permute(0, 3, 1, 2), dims=[1, 2, 3])
        img_E = torch.flip(img_E.permute(0, 3, 1, 2), dims=[1, 2, 3])
        img_H = torch.flip(img_H.permute(0, 3, 1, 2), dims=[1, 2, 3])
        lr_foreground_threshold = 0.05
    elif dataset_name == "BRATS2023":
        lr_foreground_threshold = 0.05  # The rest of the datasets use this:
    else:
        lr_foreground_threshold = 0.05  # The rest of the datasets use this:

    C_lr, H_lr, W_lr, D_lr = img_L.shape
    num_voxels_foreground = torch.sum(img_L.reshape(-1, D_lr) > lr_foreground_threshold, dim=0)
    lr_foreground_slices = num_voxels_foreground > (H_lr * W_lr * threshhold_percentage)
    img_L = img_L[:, :, :, lr_foreground_slices]
    print("Number of LR foreground slices %d / %d" % (torch.sum(lr_foreground_slices).numpy(), D_lr))
    img_E = img_E[:, :, :, lr_foreground_slices.repeat_interleave(opt['up_factor'])]
    img_H = img_H[:, :, :, lr_foreground_slices.repeat_interleave(opt['up_factor'])]
    return img_L, img_E, img_H

def apply_implicit_transform(batch, transform, out_dtype=torch.float32):
    """
    Applies implicit transform to each patch in the batch and returns the transformed batch.
    """
    L = []
    H = []
    H_xyz = []
    for patch_lr, patch_hr in zip(batch[0]['L']['data'], batch[1]['H']['data']):
        patch_dict = transform({'H': patch_hr, 'L': patch_lr})

        L.append(patch_dict['L'])
        H.append(patch_dict['H'])
        H_xyz.append(patch_dict['H_xyz'])

    L = torch.stack(L)
    H = torch.stack(H)
    H_xyz = torch.stack(H_xyz)

    if out_dtype == torch.float32:
        return {'L': L.float(), 'H': H.float(), 'H_xyz': H_xyz}
    else:
        return {'L': L, 'H': H, 'H_xyz': H_xyz}

if __name__ == "__main__":

    # Returns None if no arguments parsed, as when run in PyCharm
    args = parse_arguments()

    # Define experiment parameters
    options_file = args.options_file
    experiment_id = args.experiment_id
    print("options_file", options_file)

    #experiment_id = "debug_home_ID000001"

    if experiment_id is not None:
        # Load saved options file saved based on specified experiment id
        print("Experiment_id", experiment_id)
        opt_path = load_options_from_experiment_id(experiment_id, root_dir=config.ROOT_DIR)

        # Load options
        opt = load_json(opt_path)
        wandb_path = opt_path.rsplit("files", 1)[0]

    elif options_file is not None:
        # Load specified options file
        opt_path = os.path.join(config.ROOT_DIR, 'options', options_file)

        # Load options
        opt = load_json(opt_path)
        experiment_id = opt['experiment_id']

        ID = "*" + experiment_id
        # Select latest modified log directory
        wandb_path = os.path.join("logs", opt['datasets']['name'], "wandb", ID)
        wandb_path = glob.glob(os.path.join(wandb_path))[-1]  # Newest path
        wandb_path = wandb_path.rsplit("files", 1)[0]


    else:
        # Load options from config as default
        if config.MODEL_ARCHITECTURE == "mDCSRN_GAN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN_GAN.json')
        elif config.MODEL_ARCHITECTURE == "mDCSRN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN.json')
        elif config.MODEL_ARCHITECTURE == "SuperFormer":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_SuperFormer.json')
            # opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_superformer_old.json')
        elif config.MODEL_ARCHITECTURE == "ESRGAN3D":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ESRGAN3D.json')
        elif config.MODEL_ARCHITECTURE == "RRDBNet3D":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RRDBNet3D.json')
        elif config.MODEL_ARCHITECTURE == "RCAN3D":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RCAN3D.json')
        elif config.MODEL_ARCHITECTURE == "EDDSR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_EDDSR.json')
        elif config.MODEL_ARCHITECTURE == "MFER":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MFER.json')
        elif config.MODEL_ARCHITECTURE == "ArSSR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ArSSR.json')
        elif config.MODEL_ARCHITECTURE == "MTVNet":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MTVNet_tmp.json')
        else:
            raise NotImplementedError('Model architecture %s not implemented.' % config.MODEL_ARCHITECTURE)

        # Load options
        opt = load_json(opt_path)

        experiment_id = opt['experiment_id']
        ID = "*" + experiment_id
        # Select latest modified log directory
        wandb_path = os.path.join("logs", opt['datasets']['name'], "wandb", ID)
        wandb_path = glob.glob(os.path.join(wandb_path))[-1]  # Newest path
        wandb_path = wandb_path.rsplit("files", 1)[0]

    # Overwrite dataset in options file if specified
    if args.dataset is not None:
        opt['datasets']['name'] = args.dataset

    if args.cluster is not None:
        opt['cluster'] = args.cluster
    else:  # Default is opt['cluster'] = "DTU_HPC"
        opt['cluster'] = "DTU_HPC"

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))

    # Define universal SR model using the KAIR define_Model framework
    from models.select_model import define_Model
    model = define_Model(opt)
    model.init_test(experiment_id)

    # Overwriting dataset type to MonaiDataset for testing
    opt['datasets']["dataset_type"] = "MonaiDataset"

    from data.select_dataset import define_Dataset
    train_dataset, test_dataset, baseline_dataset = define_Dataset(opt)  # optional to have baseline dataloader as final output

    dataloader_params_train = opt['datasets']['train']['dataloader_params']
    dataloader_params_test = opt['datasets']['test']['dataloader_params']

    baseline_loader = monai.data.DataLoader(baseline_dataset,
                                            batch_size=1, #dataloader_params_test['dataloader_batch_size'],
                                            shuffle=False, #dataloader_params_test['dataloader_shuffle'],
                                            num_workers=dataloader_params_test['num_load_workers'],
                                            persistent_workers=dataloader_params_test['persist_workers'],
                                            pin_memory=dataloader_params_test['pin_memory'])

    psnr_list, ssim_list, nrmse_list = [], [], []

    # Testing loop inspired by SuperFormer: https://github.com/BCV-Uniandes/SuperFormer/blob/main/main_test_3D.py

    if opt['model_architecture'] == "MTVNet":
        patch_size = opt['datasets']['patch_size']
        #center_size = opt['datasets']['patch_size'] // 2**(opt['netG']['num_levels'] - 1)
        center_size = opt['netG']['context_sizes'][-1]  # New 
        context_width = (patch_size - center_size) // 2
        patch_size_hr = center_size * opt['up_factor']
    else:
        patch_size = opt['datasets']['patch_size']
        context_width = 0
        patch_size_hr = opt['datasets']['patch_size_hr']

    border = 4  # border voxels
    border_hr = border * opt['up_factor']

    # Create directory for test patch comparisons
    image_dir = os.path.join(wandb_path, "files/", "media/", "images/")
    print("Saving image comparisons to:", image_dir)
    if not os.path.exists(image_dir + "patch_comparisons/"):
        os.makedirs(image_dir + "patch_comparisons/")
    if not os.path.exists(image_dir + "full_slice_comparisons/"):
        os.makedirs(image_dir + "full_slice_comparisons/")

    #implicit_model_transform = ImplicitModelTransformd(opt['up_factor'], mode="test")
    implicit_model_transform = ImplicitModelTransformFastd(opt['up_factor'], mode="test")

    print("RUNNING TEST SIMPLE")

    # These are for computing mean and std across each sample
    psnr_slice_list = []
    ssim_slice_list = []
    nrmse_slice_list = []

    for sample_idx, baseline_batch in enumerate(baseline_loader):

        # Assume batch_size of baseline_loader is always one (only reconstruct one sample in the dataset at a time)
        img_H = baseline_batch['H'][0]
        img_L = baseline_batch['L'][0]
        img_E = torch.zeros_like(img_H)
        del baseline_batch

        overlap_lr = border
        overlap_hr = border * opt['up_factor']
        subject_hr = tio.Subject(H=tio.ScalarImage(tensor=img_H))
        subject_lr = tio.Subject(L=tio.ScalarImage(tensor=img_L))
        grid_sampler_lr = tio.GridSampler(subject=subject_lr, patch_size=patch_size, patch_overlap=2*overlap_lr+2*context_width, padding_mode=None)
        grid_sampler_hr = tio.GridSampler(subject=subject_hr, patch_size=patch_size_hr, patch_overlap=2*overlap_hr, padding_mode=None)
        patch_loader_lr = torch.utils.data.DataLoader(grid_sampler_lr, batch_size=opt['datasets']['test']['dataloader_params']['dataloader_batch_size'])
        patch_loader_hr = torch.utils.data.DataLoader(grid_sampler_hr, batch_size=opt['datasets']['test']['dataloader_params']['dataloader_batch_size'])
        aggregator_hr = tio.inference.GridAggregator(grid_sampler_hr, overlap_mode='hann')

        model.netG.eval()
        i = 0
        time_in = time.time()

        with torch.inference_mode():
            c = 1
            for patches_batch_lr, patches_batch_hr in tqdm(zip(patch_loader_lr, patch_loader_hr)):
                if opt['model'] == 'implicit':
                    data = apply_implicit_transform((patches_batch_lr, patches_batch_hr), implicit_model_transform)
                    model.feed_data(data)
                    model.netG_forward()
                    sr_patch = coords_to_image(model.E, patch_size=opt['datasets']['patch_size_hr'])
                else:
                    model.feed_data({'H': patches_batch_hr['H'], 'L': patches_batch_lr['L']}, add_key='data')
                    #sr_patch = torch.zeros_like(patches_batch_hr['H']['data']) # For debugging
                    model.netG_forward()
                    sr_patch = model.E
                locations_hr = patches_batch_hr['location']
                aggregator_hr.add_batch(sr_patch, locations_hr)

        img_E = aggregator_hr.get_output_tensor().float()  # convert from FP16 to FP32
        print("Full reconstruction size:", img_E.size())
        img_H = img_H.unsqueeze(0)
        img_L = img_L.unsqueeze(0)
        img_E = img_E.unsqueeze(0)

        print(i)
        time_end = time.time()

        print(f'Time taken for sample {sample_idx}: {time_end - time_in} seconds')

        if opt['model_architecture'] == "MTVNet":
            # Crop context if needed
            if context_width > 0:
                img_L = img_L[:, :, context_width:-context_width, context_width:-context_width, context_width:-context_width]

        # Crop the borders
        if border > 0:
            img_L = img_L[:, :, border:-border, border:-border, border:-border]
            img_E = img_E[:, :, border_hr:-border_hr, border_hr:-border_hr, border_hr:-border_hr]
            img_H = img_H[:, :, border_hr:-border_hr, border_hr:-border_hr, border_hr:-border_hr]
        #if opt['datasets']['name'] == "2022_QIM_52_Bone" or opt['datasets']['name'] == "Synthetic_2022_QIM_52_Bone":
        #    mask = baseline_batch['mask'][:, :, border_hr:-border_hr, border_hr:-border_hr, border_hr:-border_hr]

        # Plot slice for visual inspection
        if opt['run_type'] == "HOME PC":
            idx = img_L.shape[4]//2
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(img_L[0, 0, :, :, idx], cmap='gray')
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks
            plt.subplot(1, 3, 2)
            plt.imshow(img_E[0, 0, :, :, idx * opt['up_factor']].float().numpy(), cmap='gray')
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks
            plt.subplot(1, 3, 3)
            plt.imshow(img_H[0, 0, :, :, idx * opt['up_factor']], cmap='gray')
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks
            # Adjust subplot parameters to remove borders
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()

        # Compute performance metrics
        img_L, img_E, img_H = img_L[0], img_E[0], img_H[0]  # Assumes batch size is 1
        img_L, img_E, img_H = trim_background_slices(img_L, img_E, img_H, opt['datasets']['name'], 0.20)

        C, H, W, D = img_H.shape
        # Compute PSNR, SSIM and NRMSE slice-wise. Slice-wise approach is chosen as some dataset samples are very large.
        psnr = 0
        ssim = 0
        nrmse = 0

        for i in range(D):
            H_slice = img_H[:, :, :, i].float().squeeze().clamp(min=0.0, max=1.0).numpy()
            E_slice = img_E[:, :, :, i].float().squeeze().clamp(min=0.0, max=1.0).numpy()

            slice_psnr = calculate_psnr_2D(E_slice, H_slice, border=0)
            psnr_slice_list.append(slice_psnr)
            psnr += slice_psnr

            slice_ssim = calculate_ssim_2D(E_slice, H_slice, border=0)
            ssim_slice_list.append(slice_ssim)
            ssim += slice_ssim

            slice_nrmse = calculate_nrmse_2D(E_slice, H_slice, border=0)
            nrmse_slice_list.append(slice_nrmse)
            nrmse += slice_nrmse

        # Compute slice-wise averages
        psnr /= D
        ssim /= D
        nrmse /= D

        print("Sample PSNR: %0.2f, SSIM: %0.4f, NRMSE: %0.4f" % (psnr, ssim, nrmse))

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        nrmse_list.append(nrmse)

        # Save full slice comparisons over whole sample
        baseline_comparison_tool = utils_3D_image.ImageComparisonTool(
            patch_size_hr=(img_H.shape[1], img_H.shape[2], img_H.shape[3]),
            upscaling_methods=["tio_nearest"],  ## or tio_linear
            unnorm=False,
            div_max=False,
            out_dtype=np.uint8)

        img_dict = {'H': img_H, 'E': img_E, 'L': img_L}
        d = img_H.shape[-1]
        slice_idx_list = np.linspace(d // 4, d - d // 4, 5)  # Save 5 slices even spaced inside the middle of the last scan dimension
        for slice_idx in slice_idx_list:
            grid_image = baseline_comparison_tool.get_comparison_image(img_dict, slice_idx=int(slice_idx))
            file_name = os.path.join(image_dir, "full_slice_comparisons/full_sample_comparison_%d_%s_%dx.png" % (
            slice_idx, opt['model_architecture'], opt['up_factor']))
            grid_image = Image.fromarray(grid_image)
            grid_image.save(file_name)

    print("Final sample-wise performance statistics: %s" % opt['model_architecture'])
    print("Average sample-wise PSNR", np.mean(psnr_list))
    print("Average sample-wise SSIM", np.mean(ssim_list))
    print("Average sample-wise NRMSE", np.mean(nrmse_list))

    print("Final slice-wise performance statistics:  %s" % opt['model_architecture'])
    psnr_slice_mean, ci_psnr = get_mean_and_ci(psnr_slice_list)
    ssim_slice_mean, ci_ssim = get_mean_and_ci(ssim_slice_list)
    nrmse_slice_mean, ci_nrmse = get_mean_and_ci(nrmse_slice_list)
    print("Average slice-wise PSNR", psnr_slice_mean, "+-", ci_psnr)
    print("Average slice-wise SSIM", ssim_slice_mean, "+-", ci_ssim)
    print("Average slice-wise NRMSE", nrmse_slice_mean, "+-", ci_nrmse)

    # Create a file with performance metric statistics
    file_dir = os.path.join(wandb_path, "files/")
    if not os.path.exists(file_dir + "performance_statistics/"):
        os.makedirs(file_dir + "performance_statistics/")

    performance_statistics_dir = os.path.join(file_dir, "performance_statistics/")

    # Specify the file name
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "performance_metrics_" + opt['experiment_id'] + "_" + current_time + ".txt"

    # Open the file in write mode and write the contents
    with open(performance_statistics_dir + file_name, 'w') as file:
        file.write("MODEL ARCHITECTURE: " + opt['model_architecture'] + "\n")
        file.write("DATASET: " + opt['datasets']['name'] + "\n")
        file.write("EXPERIMENT ID: " + opt['experiment_id'] + "\n")
        file.write("RUN NAME: " + opt['run_name'] + "\n")
        file.write("MAX ITERATIONS: " + str(opt['train']['iterations']) + "\n")

        file.write("PATCH SIZE: " + str(opt['datasets']['patch_size']) + "\n")
        file.write("UP FACTOR: " + str(opt['up_factor']) + "\n")
        file.write("LEARNING RATE: " + str(opt['train']['G_optimizer_lr']) + "\n")

        # Write the lists to the file
        file.write("PERFORMANCE METRIC LISTS \n")
        file.write("PSNR SAMPLE LIST: " + str(torch.tensor(psnr_list).numpy()) + "\n")
        file.write("SSIM SAMPLE LIST: " + str(torch.tensor(ssim_list).numpy()) + "\n")
        file.write("NRMSE SAMPLE LIST: " + str(torch.tensor(nrmse_list).numpy()) + "\n")

        # Write the individual values to the file
        file.write("FINAL AVERAGE SAMPLE-WISE PERFORMANCE METRICS \n")
        file.write("AVERAGE SAMPLE-WISE PSNR: " + str(np.mean(psnr_list)) + "\n")
        file.write("AVERAGE SAMPLE-WISE SSIM: " + str(np.mean(ssim_list)) + "\n")
        file.write("AVERAGE SAMPLE-WISE NRSME: " + str(np.mean(nrmse_list)) + "\n")

        # Write the individual values to the file
        file.write("FINAL AVERAGE SLICE-WISE PERFORMANCE METRICS \n")
        file.write("AVERAGE SLICE-WISE PSNR: " + str(psnr_slice_mean) + "+-" + str(ci_psnr) + "\n")
        file.write("AVERAGE SLICE-WISE SSIM: " + str(ssim_slice_mean) + "+-" + str(ci_ssim) + "\n")
        file.write("AVERAGE SLICE-WISE NRSME: " + str(nrmse_slice_mean) + "+-" + str(ci_nrmse) + "\n")

    print(f"File '{file_name}' has been created and saved.")

    print("Done")

