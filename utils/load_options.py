
import os
import glob
import torch
import numpy as np
import json
from datetime import datetime
import random
import logging
import utils.utils_logger as util_logger
import utils.utils_image as util
import utils.utils_option as option
from utils.utils_dist import get_dist_info, init_dist

import argparse
import utils.utils_option as option
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

    # parser.add_argument("--model_architecture",
    #                     type=str,
    #                     choices=["mDCSRN_GAN", "SuperFormer", "ESRGAN3D", "RRDBNet3D"],
    #                     help="Specify the model architecture to use for the experiment.")
    #
    # parser.add_argument("--dataset", type=str,
    #                     choices=["TEST_DATASET", "2022_QIM_52_Bone", "IXI", "HCP_1200", "KIRBY21", "BRATS2023", "2022_QIM_52_Bone_Block"],
    #                     help="Specify the dataset to use for the experiment.")

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

    #if opt['rank'] == 0:
    #    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

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

        opt['total_gpu_mem'] = total_gpu_mem
        opt['run_type'] = run_type

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Run on CPU
        # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    #device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')

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


### Not used anymore
def load_and_prepare_opt(json_path='options/train_msrresnet_gan.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_D, init_path_D = option.find_last_checkpoint(opt['path']['models'], net_type='D')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netD'] = init_path_D
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerD')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    opt['path']['pretrained_optimizerD'] = init_path_optimizerD
    current_step = max(init_iter_G, init_iter_D, init_iter_E, init_iter_optimizerG, init_iter_optimizerD)

    # opt['path']['pretrained_netG'] = ''
    # current_step = 0
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        util_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

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

    return opt
