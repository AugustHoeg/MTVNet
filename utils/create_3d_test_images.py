import argparse
import math
import os
import sys
from glob import glob
import config
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, SmartCacheDataset, create_test_image_3d, partition_dataset

# create random image, mask pairs for training

#n1_img = nib.load("C:/Users/augus/OneDrive/Dokumenter/GitHub/Vedrana_master_project/3D_datasets/datasets/2022_QIM_52_Bone/femur_001/clinical/f_001.nii")
#print(np.mean(n1_img.get_fdata()))
#print(np.std(n1_img.get_fdata()))

dir = "../testimages"
num_img = 2
print(f"generating synthetic data to {dir} (this may take a while)")
# set random seed to generate same random data for every node
#np.random.seed(seed=0)
for i in range(num_img):
    im, seg = create_test_image_3d(1000, 1000, 1000, num_seg_classes=1, channel_dim=-1)
    print(np.max(im), np.min(im))
    print(np.max(seg), np.min(seg))
    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(dir, f"img_HR{i+3:d}.nii"))
    #n = nib.Nifti1Image(seg, np.eye(4))
    #nib.save(n, os.path.join(dir, f"mask{i:d}.nii"))
    print("Image %d" % i)

