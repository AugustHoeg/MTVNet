
import os
import glob
import torch
import numpy as np
import json
from datetime import datetime
import random
from utils.utils_dist import get_dist_info, init_dist

import argparse
import monai

def get_timestamp():
    return datetime.now().strftime('_%y%m%d_%H%M%S')

def save_json(opt, wandb_path):
    # convert the opt into json file and save in wandb directory
    opt_path = opt['opt_path']
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)

    # Create directory for options file if it doesn't exist
    options_dir = wandb_path + "/options"
    if not os.path.exists(options_dir):
        os.makedirs(options_dir)

    dump_path = os.path.join(options_dir, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments with different model architectures on a dataset.")
    parser.add_argument("--options_file",
                        type=str,
                        help="Specify the .json options file to use for the experiment.")
    parser.add_argument("--experiment_id",
                        type=str,
                        help="Specify the experiment id to load .json options file from.",
                        required=False)
    parser.add_argument("--dataset",
                        type=str,
                        help="Specify the dataset to use for training (overwrites dataset in .json options file).",
                        required=False)
    parser.add_argument("--cluster",
                        type=str,
                        help="Specify name of HPC cluster where the experiment is run.",
                        required=False)

    return parser.parse_args()


def load_options_from_experiment_id(experiment_id, root_dir):
    opt_files = glob.glob(os.path.join(root_dir + "/logs/" + "/*/" "/wandb/" + "*" + experiment_id + "/files/saved_models/*G.h5"))
    opt_files.sort(key=os.path.getmtime, reverse=True)
    opt_file = opt_files[0].rsplit('files', 1)[0]
    opt_path = glob.glob(os.path.join(opt_file, "files/options", "*.json"))[0]  # Get latest modified directory with the specified experiment_id
    return opt_path

def load_json(opt_path):
    import json
    from collections import OrderedDict
    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # Set options path
    opt['opt_path'] = opt_path

    # Calculate high-resolution patch size
    opt['datasets']['patch_size_hr'] = opt['datasets']['patch_size']*opt['up_factor']

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    if opt['gpu_ids'] is not None:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0
        # Flag to run the script as on Home PC or HPC.
        run_type = "HOME PC" if total_gpu_mem < 10 else "HPC"
        print(f"run type: {run_type}.")

        opt['total_gpu_mem'] = total_gpu_mem
        opt['run_type'] = run_type

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Run on CPU
        # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # ----------------------------------------
    # default setting for distributeddataparallel
    # ----------------------------------------
    if 'find_unused_parameters' not in opt:
        opt['find_unused_parameters'] = True
    if 'use_static_graph' not in opt:
        opt['use_static_graph'] = False
    if 'dist' not in opt:
        opt['dist'] = False
    if opt['gpu_ids'] is not None:
        opt['num_gpu'] = len(opt['gpu_ids'])
        print('number of GPUs is: ' + str(opt['num_gpu']))
    else:
        opt['num_gpu'] = 0

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Monai set_deterministic enabled")
    monai.utils.misc.set_determinism(seed)
    np.random.RandomState(seed)

    return opt

