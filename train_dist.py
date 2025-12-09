import os
import time

import monai.transforms
import torch
from monai.data import SmartCacheDataset

import config
from utils.load_options import load_json, parse_arguments, save_json
from utils.utils_3D_image import crop_context, crop_center
from utils.utils_dist import init_dist, get_dist_info
from torch.distributed import init_process_group, destroy_process_group

from torch.profiler import profile, ProfilerActivity

import lightning as L

from train import train_model


def main():

    print("RUNNING TRAIN DDP TEST")

    #print("Cuda is available", torch.cuda.is_available())
    #print("Cuda device count", torch.cuda.device_count())
    #print("Cuda current device", torch.cuda.current_device())
    #print("Cuda device name", torch.cuda.get_device_name(0))

    # Returns None if no arguments parsed, as when run in IDE
    args = parse_arguments()

    # Define experiment parameters
    options_file = args.options_file
    print("options_file", options_file)

    # Load default experiment option
    if options_file is None:
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
        elif config.MODEL_ARCHITECTURE == "DUMMY":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_DUMMY.json')
        else:
            raise NotImplementedError('Model architecture %s not implemented.' % config.MODEL_ARCHITECTURE)
    else:
        opt_path = os.path.join(config.ROOT_DIR, 'options', options_file)

    # Load options
    opt = load_json(opt_path)

    # Overwrite dataset in options file if specified
    if args.dataset is not None:
        opt['datasets']['name'] = args.dataset

    if args.cluster is not None:
        opt['cluster'] = args.cluster

    # Define universal SR model using the KAIR define_Model framework
    from models.select_model import define_Model
    model = define_Model(opt)

    # Define wandb run
    if opt['rank'] == 0:
        model.define_wandb_run()

    # Run initialization of model for training
    model.init_train()

    # Save copy of options file in wandb directory
    if opt['rank'] == 0:
        save_json(opt, wandb_path=model.run.dir)

    from data.select_dataset import define_Dataset

    print("OVERRIDE DDP CACHE DATASET")
    opt['datasets']['dataset_type'] = "DDP_CacheDataset"
    # in case of DDP_CacheDataset is used, the dataset is split into parts for each rank
    # meaning we do not need to use DistributedSampler
    train_dataset, test_dataset, baseline_dataset = define_Dataset(opt, return_filepaths=False)

    dataloader_params_train = opt['datasets']['train']['dataloader_params']
    dataloader_params_test = opt['datasets']['test']['dataloader_params']

    if opt['dist'] and (opt['datasets']['dataset_type'] != "DDP_CacheDataset"):
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=dataloader_params_train['dataloader_shuffle'],
                                           drop_last=True,
                                           seed=opt['train']['manual_seed'])

        test_sampler = DistributedSampler(test_dataset,
                                          shuffle=dataloader_params_test['dataloader_shuffle'],
                                          drop_last=True,
                                          seed=opt['train']['manual_seed'])

        train_loader = monai.data.DataLoader(train_dataset,
                                             batch_size=dataloader_params_train['dataloader_batch_size'],
                                             shuffle=False,
                                             num_workers=dataloader_params_train['num_load_workers'],
                                             persistent_workers=dataloader_params_train['persist_workers'],
                                             pin_memory=dataloader_params_train['persist_workers'],
                                             sampler=train_sampler)

        test_loader = monai.data.DataLoader(test_dataset,
                                            batch_size=dataloader_params_test['dataloader_batch_size'],
                                            shuffle=False,
                                            num_workers=dataloader_params_test['num_load_workers'],
                                            persistent_workers=dataloader_params_test['persist_workers'],
                                            pin_memory=dataloader_params_test['pin_memory'],
                                            sampler=test_sampler)

    else:
        train_loader = monai.data.DataLoader(train_dataset,
                                             batch_size=dataloader_params_train['dataloader_batch_size'],
                                             shuffle=dataloader_params_train['dataloader_shuffle'],
                                             num_workers=dataloader_params_train['num_load_workers'],
                                             persistent_workers=dataloader_params_train['persist_workers'],
                                             pin_memory=dataloader_params_train['pin_memory'])

        test_loader = monai.data.DataLoader(test_dataset,
                                            batch_size=dataloader_params_test['dataloader_batch_size'],
                                            shuffle=dataloader_params_test['dataloader_shuffle'],
                                            num_workers=dataloader_params_test['num_load_workers'],
                                            persistent_workers=dataloader_params_test['persist_workers'],
                                            pin_memory=dataloader_params_test['pin_memory'])

    # Train model
    if opt['datasets']['dataset_type'] == "MonaiSmartCacheDataset":
        train_dataset.start()
        test_dataset.start()

    iterations = opt['train']['iterations']
    validation_iterations = opt['train']['validation_iterations']

    time_start = time.time()

    if args.run_profile:
        from torch.profiler import profile, tensorboard_trace_handler
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True,
                     record_shapes=False,
                     on_trace_ready=tensorboard_trace_handler("./profiles")) as prof:
            out_dict = train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True)
    else:
        out_dict = train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True)

    time_end = time.time()
    print("Time taken to train: ", time_end - time_start)

    destroy_process_group()

    print("Done")


if __name__ == "__main__":
    main()

    # Run training:
    # python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train_dist.py --opt options/train_mDCSRN.json  --dist True
    # or
    # torchrun --nproc_per_node=2 --master_port=29500 train_dist.py --opt options/train_mDCSRN.json --dist True
