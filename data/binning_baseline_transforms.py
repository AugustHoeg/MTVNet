import torch
import monai.transforms as mt
import torch

from data.baseline_transforms import Resize_functional
from data.train_transforms import GaussianblurImaged, ImplicitModelTransformd

class binning_baseline_transforms():

    def __init__(self, opt, mode="train", synthetic=False):

        self.synthetic = synthetic  # Whether to create LR data from synthetic degradation of HR data

        self.implicit = True if opt['model'] == "implicit" else False
        self.mode = mode

        self.size_hr = opt['datasets']['patch_size_hr']
        self.size_lr = opt['datasets']['patch_size']
        self.up_factor = opt['up_factor']

        self.blur_sigma = opt['datasets']['blur_sigma']
        self.downsampling_method = opt['datasets']['downsampling_method']
        self.patches_per_batch = opt['datasets']['train']['dataset_params']['patches_per_batch']
        self.channel_dim = opt['datasets']['channel_dim']

        if opt['datasets']['blur_method'] == '3d_gaussian_blur':
            self.blur_transform = GaussianblurImaged(blur_sigma=self.blur_sigma, in_key="L", out_key="L")
        elif opt['datasets']['blur_method'] == 'monai_gaussian_blur':
            self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['datasets']['blur_sigma'])

        self.resize_functional = Resize_functional(self.up_factor, self.downsampling_method, in_key='L', out_key='L')
        #self.crop_to_mulitple = CropToMultiple(opt)
        #self.pad_to_multiple = PadToMultiple(opt)

        self.implicit_model_transform = ImplicitModelTransformd(opt['up_factor'], mode=mode)

        self.size_lr = opt['datasets']['patch_size']
        self.pad_transform_lr = mt.Identityd(keys=['L'])
        self.divisible_pad = 4 if opt['up_factor'] % 2 == 0 else 3  # Added adaptive divisible padding
        if opt['model_architecture'] == "MTVNet":
            center_size = opt['netG']['context_sizes'][-1]  # fixed assumption of level_ratio = 2

            pad_size = (self.size_lr - center_size) // 2  # pad half of the context on all sides
            if pad_size > 0:
                self.pad_transform_lr = mt.BorderPadd(keys=['L'], spatial_border=[pad_size, pad_size, pad_size], mode='constant')  # Pad here if net is MTVNet
            else:
                self.pad_transform_lr = mt.Identityd(keys=['L'])


    def get_transforms(self):

        transforms = mt.Compose(
            [
                mt.LoadImaged(keys=["H", "L"], dtype=torch.float32),
                mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                #mt.DivisiblePadd(keys=["H", "L"], k=self.divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
                #mt.Zoomd(keys=["L"],
                #         zoom=1 / self.up_factor,
                #         mode=self.downsampling_method,
                #         align_corners=True,
                #         keep_size=False,
                #         dtype=torch.float32),
                self.pad_transform_lr,
            ]
        )

        return transforms
