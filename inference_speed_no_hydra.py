import datetime
import os
import glob
import time

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from tqdm import tqdm

import config
from utils.load_options import load_json, parse_arguments, load_options_from_experiment_id
from utils import utils_3D_image

def create_inference_speed_file(inference_time_list, wandb_path, opt, max_memory_reserved=0):

    # Create a file with performance metric statistics
    file_dir = os.path.join(wandb_path, "files/")
    if not os.path.exists(file_dir + "inference_statistics/"):
        os.makedirs(file_dir + "inference_statistics/")

    inference_statistics_dir = os.path.join(file_dir, "inference_statistics/")

    inference_speed_mean, ci = get_mean_and_ci(inference_time_list)

    # Specify the file name
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"inference_statistics_{opt['experiment_id']}_{current_time}.txt"

    # Open the file in write mode and write the contents
    with open(inference_statistics_dir + file_name, 'w') as file:
        file.write("MODEL ARCHITECTURE: " + opt['model_architecture'] + "\n")
        file.write("TRAINABLE PARAMETERS: " + str(opt['params_netG']) + "\n")
        file.write("DATASET: " + opt['datasets']['name'] + "\n")
        file.write("EXPERIMENT ID: " + opt['experiment_id'] + "\n")
        file.write("RUN NAME: " + opt['run_name'] + "\n")

        file.write("PATCH SIZE: " + str(opt['datasets']['patch_size']) + "\n")
        file.write("UP FACTOR: " + str(opt['up_factor']) + "\n")

        # Write the lists to the file
        file.write("INFERENCE SPEED LIST: " + str(torch.tensor(inference_time_list).numpy()) + "\n")

        # Write the individual values to the file
        file.write("AVERAGE INFERENCE SPEED: " + str(inference_speed_mean) + "+-" + str(ci) + "\n")

        # Write max memory usage if available
        file.write("MAX MEMORY RESERVED (GB): " + str(max_memory_reserved / 10 ** 9) + "\n")

    print(f"File '{file_name}' has been created and saved.")


def get_mean_and_ci(data_sequence, confidence=0.95):

    data = np.array(data_sequence)
    n = len(data)
    mean, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h

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


def generate_input_data(input_type, patch_size, device):

    if input_type == "2D":
        input_data = torch.randn(1, 1, patch_size, patch_size).to(device)
    else:
        input_data = torch.randn(1, 1, patch_size, patch_size, patch_size).to(device)

    return input_data


def main():
    # Returns None if no arguments parsed, as when run in IDE
    args = parse_arguments()

    # Define experiment parameters
    options_file = args.options_file
    experiment_id = args.experiment_id
    print("options_file", options_file)

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
        # wandb_path = os.path.join("logs", opt['datasets']['name'], "wandb", ID)
        # wandb_path = glob.glob(os.path.join(wandb_path))[-1]  # Newest path
        # wandb_path = wandb_path.rsplit("files", 1)[0]


    else:
        # Load options from config as default
        if config.MODEL_ARCHITECTURE == "mDCSRN_GAN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN_GAN.json')
        elif config.MODEL_ARCHITECTURE == "mDCSRN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN.json')
        elif config.MODEL_ARCHITECTURE == "SuperFormer":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_SuperFormer.json')
        elif config.MODEL_ARCHITECTURE == "ESRGAN3D":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ESRGAN3D.json')
        elif config.MODEL_ARCHITECTURE == "RRDBNet3D":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RRDBNet3D.json')
        elif config.MODEL_ARCHITECTURE == "EDDSR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_EDDSR.json')
        elif config.MODEL_ARCHITECTURE == "MFER":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MFER.json')
        elif config.MODEL_ARCHITECTURE == "MTVNet":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MTVNet.json')
        elif config.MODEL_ARCHITECTURE == "ArSSR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ArSSR.json')
        elif config.MODEL_ARCHITECTURE == "RCAN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RCAN.json')
        elif config.MODEL_ARCHITECTURE == "SwinIR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_SwinIR.json')
        elif config.MODEL_ARCHITECTURE == "HAT":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_HAT.json')
        elif config.MODEL_ARCHITECTURE == "DRCT":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_DRCT.json')
        else:
            raise NotImplementedError('Model architecture %s not implemented.' % config.MODEL_ARCHITECTURE)

        # Load options
        opt = load_json(opt_path)

        experiment_id = opt['experiment_id']
        ID = "*" + experiment_id
        # Select latest modified log directory
        # wandb_path = os.path.join("logs", opt['datasets']['name'], "wandb", ID)
        # wandb_path = glob.glob(os.path.join(wandb_path))[-1]  # Newest path
        # wandb_path = wandb_path.rsplit("files", 1)[0]

    # Overwrite dataset in options file if specified
    if args.dataset is not None:
        opt['datasets']['name'] = args.dataset

    if args.cluster is not None:
        opt['cluster'] = args.cluster
    else:  # Default is opt['cluster'] = "DTU_HPC"
        opt['cluster'] = "DTU_HPC"

    # Set input type to 3D if not specified
    if 'input_type' not in opt:
        opt['input_type'] = '3D'

    # Overwrite gradient checkpointing
    opt['netG']['use_checkpoint'] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))
    
    print("Experiment ID:", experiment_id)

    from models.select_model import define_Model
    model = define_Model(opt)
    model.init_train()

    # Generate random input data for inference
    input_data = generate_input_data(opt['input_type'], patch_size=opt['datasets']['patch_size'], device=device)

    iterations = 100  # Number of iterations to run inference speed test

    # Warm-up runs
    for _ in range(5):
        model.feed_data({'L': input_data}, need_H=False)
        _ = model.netG_forward()

    inference_time_list = []

    # Measure inference time with GPU synchronization
    for _ in tqdm(range(iterations), desc=f"Running inference speed test", mininterval=1):
        model.feed_data({'L': input_data}, need_H=False)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model.netG_forward()
        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        #print(f"Inference time with GPU synchronization: {inference_time:.4f} seconds")

        inference_time_list.append(inference_time)

    # Print the number of parameters in the model
    params_netG = utils_3D_image.numel(model.netG, only_trainable=True)
    opt['params_netG'] = params_netG

    print(f"Number of trainable parameters: {params_netG}")

    loss_func = nn.MSELoss()

    input_data = generate_input_data(opt['input_type'], opt['datasets']['patch_size'], device=device)
    input_data_hr = generate_input_data(opt['input_type'], int(opt['netG']['context_sizes'][-1]*opt['up_factor']), device=device)
    print("Input data shape:", input_data.shape)
    print("Input HR data shape:", input_data_hr.shape)

    model.netG.train()

    # Record maximum gpu memory usage
    torch.cuda.reset_peak_memory_stats(device)

    # Run forward + backward pass to measure memory usage
    model.feed_data({'L': input_data}, need_H=False)
    _ = model.netG_forward()
    output = model.E 
    print("Output shape:", output.shape)

    loss = loss_func(output, input_data_hr)
    loss.backward()

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    #path = wandb_path
    if not os.path.exists(os.path.join(config.ROOT_DIR, "inference_speed_results/")):
        os.mkdir(os.path.join(config.ROOT_DIR, "inference_speed_results/"))

    path = os.path.join(config.ROOT_DIR, "inference_speed_results/", experiment_id)

    # Create files with performance metric statistics
    create_inference_speed_file(inference_time_list, path, opt, max_memory_reserved)

    print("Done")

if __name__ == "__main__":
    main()
