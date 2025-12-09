import numpy as np
import torch
import monai.transforms as mt
from monai.transforms import Randomizable
from data.train_transforms import GaussianblurImaged, ImplicitModelTransformd

from data.train_transforms import RandomCropPaird

class binning_transforms_cachable():

    def __init__(self, opt, mode="train", synthetic=False):

        self.synthetic = synthetic  # Whether to create LR data from synthetic degradation of HR data

        self.mode = mode  # train or test
        self.implicit = True if opt['model'] == "implicit" else False

        self.size_hr = opt['datasets']['patch_size_hr']
        self.size_lr = opt['datasets']['patch_size']
        self.blur_sigma = opt['datasets']['blur_sigma']
        self.downsampling_method = opt['datasets']['downsampling_method']
        self.patches_per_batch = opt['datasets']['train']['dataset_params']['patches_per_batch']
        self.channel_dim = opt['datasets']['channel_dim']

        self.crop_type = opt['datasets']['crop_type']
        self.up_factor = opt['up_factor']

        self.pad_transform = mt.Identityd(keys=['H'])
        self.pad_size = 0

        self.enable_femur_padding = opt['datasets']['enable_femur_padding']
        print("Enable femur padding:", self.enable_femur_padding)
        self.pad_transform = mt.Identityd(keys=['L'])
        self.divisible_pad = 4 if opt['up_factor'] % 2 == 0 else 3  # Added adaptive divisible padding

        if opt['model_architecture'] == "MTVNet" and self.enable_femur_padding:
            center_size = opt['netG']['context_sizes'][-1]  # fixed assumption of level_ratio = 2

            self.pad_size = (self.size_lr - center_size) // 2  # pad half of the context on all sides
            #pad_size = pad_size * opt['up_factor']  # Since we pad the HR image, multiply padding by the upscaling factor
            if self.pad_size > 0:
                self.pad_transform = mt.BorderPadd(keys=['L'], spatial_border=[self.pad_size, self.pad_size, self.pad_size], mode='constant')  # Pad here if net is MTVNet

        if opt['datasets']['blur_method'] == '3d_gaussian_blur':
            self.blur_transform = GaussianblurImaged(blur_sigma=self.blur_sigma, in_key="L", out_key="L")
        elif opt['datasets']['blur_method'] == 'monai_gaussian_blur':
            self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['datasets']['blur_sigma'])

        if opt['datasets']['norm_type'] == "scale_intensity":
            self.norm_transform = mt.ScaleIntensityd(keys=["H", "L"], minv=0.0, maxv=1.0)
        elif opt['datasets']['norm_type'] == "znormalization":
            self.norm_transform = mt.NormalizeIntensityd(keys=["H", "L"])

        self.border_hr = opt['up_factor'] * 2  # 2 for 1x, 4 for 2x and 8 for 4x
        self.border_lr = self.border_hr // opt['up_factor']

        crop_pair_mode = "crop_uniform"  # only "crop_uniform" supported
        self.random_crop_pair = RandomCropPaird(self.size_lr,
                                                self.up_factor,
                                                lr_foreground_threshold=0,
                                                crop_pair_mode=crop_pair_mode,
                                                pad_size=self.pad_size)

    def get_transforms(self):

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H", "L"], dtype=torch.float16),
                mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                #mt.DivisiblePadd(keys=["H", "L"], k=self.divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
                #mt.Zoomd(keys=["L"],
                #         zoom=1 / self.up_factor,
                #         mode=self.downsampling_method,
                #         align_corners=True,
                #         keep_size=False),
                self.pad_transform,

                # Random transforms
                self.random_crop_pair,

            ]
        )

        return transforms
