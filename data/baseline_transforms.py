import torch

import monai.transforms as mt
import monai.transforms.spatial.functional as msF

from data.train_transforms import GaussianblurImaged, ImplicitModelTransformd

class Resize_baseline_transformsV2():

    def __init__(self, opt, mode="train"):

        self.implicit = True if opt['model'] == "implicit" else False
        self.mode = mode
        self.up_factor = opt['up_factor']
        self.blur_sigma = opt['datasets']['blur_sigma']
        self.downsampling_method = opt['datasets']['downsampling_method']
        self.channel_dim = opt['datasets']['channel_dim']

        # Fields for crop_foreground
        if opt['datasets']['name'] == "KIRBY21":
            self.lr_foreground_threshold = 0.02  # Kirby21 is T2w and therefore we need lower threshold
        elif opt['datasets']['name'] == "IXI":
            self.lr_foreground_threshold = 0.05
        elif opt['datasets']['name'] == "BRATS2023":
            self.lr_foreground_threshold = 0.05
        elif opt['datasets']['name'] == "HCP_1200":
            self.lr_foreground_threshold = 0.05
        else:
            self.lr_foreground_threshold = 0.05

        if opt['datasets']['blur_method'] == '3d_gaussian_blur':
            self.blur_transform = GaussianblurImaged(keys=["L"], blur_sigma=self.blur_sigma)
        elif opt['datasets']['blur_method'] == 'monai_gaussian_blur':
            self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['datasets']['blur_sigma'])

        if opt['datasets']['norm_type'] == "scale_intensity":
            self.norm_transform = mt.ScaleIntensityd(keys=["H"], minv=0.0, maxv=1.0)
        elif opt['datasets']['norm_type'] == "znormalization":
            self.norm_transform = mt.NormalizeIntensityd(keys=["H"])

        self.resize_functional = Resize_functional(self.up_factor, self.downsampling_method, in_key='L', out_key='L')

        self.implicit_model_transform = ImplicitModelTransformd(opt['up_factor'], mode=mode)

        self.size_lr = opt['datasets']['patch_size']
        self.pad_transform_lr = mt.Identityd(keys=['L'])
        if opt['model_architecture'] == "MTVNet":
            center_size = opt['netG']['context_sizes'][-1]  # fixed assumption of level_ratio = 2

            pad_size = (self.size_lr - center_size) // 2  # pad half of the context on all sides
            if pad_size > 0:
                self.pad_transform_lr = mt.BorderPadd(keys=['L'], spatial_border=[pad_size, pad_size, pad_size], mode='constant')  # Pad here if net is MTVNet
            else:
                self.pad_transform_lr = mt.Identityd(keys=['L'])

        # Define foreground cropping / divisible padding
        #self.divisible_transform = mt.CropForegroundd(keys=["H"], source_key="H", margin=8, select_fn=self.foreground_threshold_func, k_divisible=4)
        self.divisible_transform = mt.DivisiblePadd(keys=["H"], k=4, mode="constant")  # Ensure HR and LR scans have even dimensions

    def foreground_threshold_func(self, img):
        # threshold foreground
        return img > self.lr_foreground_threshold  # This is actually the HR image

    def get_transforms(self):

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H"], replacement=0),  # Remove any NaNs
                self.norm_transform,
                self.divisible_transform,
                mt.CopyItemsd(keys=["H"], times=1, names=["L"]),
                mt.GaussianSmoothd(keys=["L"], sigma=self.blur_sigma),
                mt.Zoomd(keys=["L"],
                         zoom=1 / self.up_factor,
                         mode=self.downsampling_method,
                         align_corners=True,
                         keep_size=False),
                self.pad_transform_lr,
                # Skip cropping for baseline

            ]
        )

        return transforms

class Resize_functional():
    def __init__(self, up_factor, downsampling_method, **kwargs):
        self.in_key = kwargs['in_key']
        self.out_key = kwargs['out_key']

        # These are used for the resizing whole volumetric samples where the dimensions are not known
        self.up_factor = up_factor
        self.downsampling_method = downsampling_method

    def __call__(self, img_dict):
        # Read size of volumetric sample
        size_hr = img_dict[self.in_key].shape
        # infer size of LR volume from up_factor
        h_lr, w_lr, d_lr = size_hr[1] // self.up_factor, size_hr[2] // self.up_factor, size_hr[3] // self.up_factor

        img_dict[self.out_key] = msF.resize(img_dict[self.in_key],
                                            out_size=[h_lr, w_lr, d_lr],
                                            mode=self.downsampling_method,
                                            align_corners=True,
                                            dtype=None,
                                            input_ndim=3,
                                            anti_aliasing=False,
                                            anti_aliasing_sigma=None,
                                            lazy=False,  # True is bugged somehow
                                            transform_info=None)

        return img_dict


