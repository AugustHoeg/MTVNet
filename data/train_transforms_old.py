import math

import numpy as np
import torch
import monai.transforms as mt
from monai.transforms import Randomizable

from data.kspace import KspaceTruncd
from utils.utils_3D_image import test_3d_gaussian_blur
from utils.utils_ARSSR import make_coord


class CropBorderd(mt.Transform):
    def __init__(self, threshold, k_div=4, min_size=132):
        self.threshold = threshold
        self.k_div = k_div
        self.min_size = min_size

    def otsu_threshold(self, tensor):
        """
        Compute an Otsu threshold for a 3D tensor with values normalized to [0, 1].
        
        Parameters:
        tensor (torch.Tensor): 3D input tensor.
        
        Returns:
        float: Optimal threshold value.
        """
        # Flatten the tensor and calculate histogram with fixed bins in [0, 1]
        flat_tensor = tensor.flatten()
        hist = torch.histc(flat_tensor, bins=256, min=0.0, max=1.0)
        
        # Calculate probabilities for each bin
        total_voxels = flat_tensor.numel()
        prob = hist / total_voxels
        cumulative_sum = torch.cumsum(prob, dim=0)
        cumulative_mean = torch.cumsum(hist * torch.linspace(0, 1, steps=256), dim=0) / total_voxels
        
        # Global mean
        global_mean = cumulative_mean[-1]

        # Compute inter-class variance for each threshold
        inter_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum))
        inter_class_variance[torch.isnan(inter_class_variance)] = 0  # Handle NaNs

        # Get the bin with maximum inter-class variance
        optimal_threshold_index = inter_class_variance.argmax().item()
        optimal_threshold = optimal_threshold_index / 255.0  # Scale index to [0, 1]
        
        return optimal_threshold

    def __call__(self, img_dict):
        # Apply threshold to create mask and get non-background coordinates
        mask = img_dict['H'] > self.otsu_threshold(img_dict['H'])
        coords = mask.nonzero(as_tuple=False)
        
        # Get bounding box for the mask
        h_min, w_min, d_min = coords.min(dim=0).values[1:]
        h_max, w_max, d_max = coords.max(dim=0).values[1:] + 1  # +1 to include the max index

        # Round the coordinates to the nearest multiple of k_div
        h_min, w_min, d_min = (h_min // self.k_div) * self.k_div, (w_min // self.k_div) * self.k_div, (d_min // self.k_div) * self.k_div
        h_max, w_max, d_max = (h_max // self.k_div) * self.k_div, (w_max // self.k_div) * self.k_div, (d_max // self.k_div) * self.k_div

        # Ensure minimum cropping dimensions by extending the bounding box if necessary
        h_size, w_size, d_size = h_max - h_min, w_max - w_min, d_max - d_min
        if h_size < self.min_size:
            h_max = min(self.min_size, img_dict['H'].shape[1])  # Ensure within original bounds
            h_min = 0        
        if w_size < self.min_size:
            w_max = min(self.min_size, img_dict['H'].shape[2])
            w_min = 0        
        if d_size < self.min_size:
            d_max = min(self.min_size, img_dict['H'].shape[3])
            d_min = 0

        # Crop the tensor and update img_dict
        img_dict['H'] = img_dict['H'][:, h_min:h_max, w_min:w_max, d_min:d_max]
        return img_dict


class ImplicitModelTransformd():
    def __init__(self, up_factor, mode,  **kwargs):

        self.up_factor = up_factor
        self.mode = mode
        if mode == "train":
            self.sample_size = 8000
        else:
            self.sample_size = -1


    def __call__(self, img_dict):

        patch_hr = img_dict['H']

        if self.mode == "train":
            # compute the size of HR patch according to the scale
            hr_h, hr_w, hr_d = (torch.tensor([10, 10, 10], dtype=torch.int) * self.up_factor)
            # generate HR patch by cropping
            patch_hr = patch_hr[0, :hr_h, :hr_w, :hr_d]
            # simulated LR patch by down-sampling HR patch
            patch_lr = img_dict['L'][:, :hr_h//self.up_factor, :hr_w//self.up_factor, :hr_d//self.up_factor]
        else:
            # Take whole HR/LR patch
            patch_hr = patch_hr[0, :, :, :]
            patch_lr = img_dict['L'][:, :, :, :]

        # generate coordinate set
        xyz_hr = make_coord(patch_hr.shape, flatten=True)
        # randomly sample voxel coordinates
        if self.mode == "train":
            sample_indices = torch.randperm(len(xyz_hr))[:self.sample_size]  # sample without replacement
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]
        else:
            patch_hr = patch_hr.reshape(-1, 1)

        return {'L': patch_lr, 'H_xyz': xyz_hr, 'H': patch_hr}



class ImplicitModelTransformFastd:
    def __init__(self, up_factor, mode, **kwargs):
        self.up_factor = up_factor
        self.mode = mode
        self.sample_size = 8000 if mode == "train" else -1

        if self.mode == "train":
            # Precompute HR shape and HR coordinate grid for training
            self.hr_shape = (torch.tensor([10, 10, 10], dtype=torch.int) * up_factor).tolist()
            self.coord_grid = self._precompute_coord(self.hr_shape)  # Precompute coord grid only once

    def __call__(self, img_dict):
        patch_hr = img_dict['H']

        if self.mode == "train":
            hr_h, hr_w, hr_d = self.hr_shape

            # Crop the high-resolution patch
            patch_hr = patch_hr[0, :hr_h, :hr_w, :hr_d]

            # Simulate low-resolution patch
            patch_lr = img_dict['L'][:, :hr_h // self.up_factor, :hr_w // self.up_factor, :hr_d // self.up_factor]

            # Use the precomputed coordinate grid
            xyz_hr = self.coord_grid

            # Sample random indices once
            sample_indices = torch.randperm(xyz_hr.shape[0], device=patch_hr.device)[:self.sample_size]
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]
        else:
            # For testing, just reshape the entire patch
            patch_hr = patch_hr[0]
            patch_lr = img_dict['L']
            xyz_hr = self.make_coord(patch_hr.shape)
            patch_hr = patch_hr.reshape(-1, 1)

        return {'L': patch_lr, 'H_xyz': xyz_hr, 'H': patch_hr}

    @staticmethod
    def make_coord(shape):
        """
        Generate the coordinate grid for a given shape.
        """
        ranges = [-1, 1]
        coord_seqs = [torch.linspace(ranges[0] + (1 / (2 * n)), ranges[1] - (1 / (2 * n)), n, device='cuda') for n in shape]
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        return ret.view(-1, ret.shape[-1])

    def _precompute_coord(self, hr_shape):
        """
        Precompute and cache coordinate grid for training mode.
        """
        return self.make_coord(hr_shape)

class Resize_transformsV2():

    def __init__(self, opt, mode="train"):

        self.implicit = True if opt['model'] == "implicit" else False
        self.mode = mode
        self.size_hr = opt['datasets']['patch_size_hr']
        self.size_lr = opt['datasets']['patch_size']
        self.blur_sigma = opt['datasets']['blur_sigma']
        self.downsampling_method = opt['datasets']['downsampling_method']
        self.patches_per_batch = opt['datasets']['train']['dataset_params']['patches_per_batch']
        self.channel_dim = opt['datasets']['channel_dim']
        self.up_factor = opt['up_factor']

        # Define foreground cropping / divisible padding
        # self.divisible_transform = mt.CropForegroundd(keys=["H"], source_key="H", margin=0, select_fn=self.foreground_threshold_func, k_divisible=4)
        self.divisible_transform = mt.DivisiblePadd(keys=["H"], k=4, mode="constant")  # Ensure HR and LR scans have even dimensions
        # self.border_crop = CropBorderd(self.lr_foreground_threshold)

        # Normalization and scaling
        if opt['datasets']['norm_type'] == "scale_intensity":
            self.norm_transform = mt.ScaleIntensityd(keys=["H"], minv=0.0, maxv=1.0)
        elif opt['datasets']['norm_type'] == "znormalization":
            self.norm_transform = mt.NormalizeIntensityd(keys=["H"])

        # Padding transform
        self.pad_transform = mt.Identityd(keys=['H'])
        self.pad_size = 0
        if opt['model_architecture'] == "MTVNet":
            center_size = opt['netG']['context_sizes'][-1]  # fixed assumption of level_ratio = 2

            self.pad_size = (self.size_lr - center_size) // 2  # pad half of the context on all sides
            # pad_size = pad_size * opt['up_factor']  # Since we pad the HR image, multiply padding by the upscaling factor
            if self.pad_size > 0:
                self.pad_transform = mt.BorderPadd(keys=["L"], spatial_border=[self.pad_size, self.pad_size, self.pad_size], mode='constant')  # Pad here if net is MTVNet
            else:
                self.pad_transform = mt.Identityd(keys=["L"])

        # Degradation
        self.degradation_method = "kspace_trunc"
        if self.degradation_method == "resize":

            if opt['datasets']['blur_method'] == '3d_gaussian_blur':
                self.blur_transform = GaussianblurImaged(blur_sigma=self.blur_sigma, in_key="L", out_key="L")
            elif opt['datasets']['blur_method'] == 'monai_gaussian_blur':
                self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['datasets']['blur_sigma'])

            self.degradation = mt.Compose([
                mt.CopyItemsd(keys=["H"], times=1, names=["L"]),
                mt.GaussianSmoothd(keys=["L"], sigma=self.blur_sigma),
                mt.Zoomd(keys=["L"],
                         zoom=1 / self.up_factor,
                         mode=self.downsampling_method,
                         align_corners=True,
                         keep_size=False),
            ])
        elif self.degradation_method == "kspace_trunc":
            self.degradation = mt.Compose([
                mt.CopyItemsd(keys=["H"], times=1, names=["L"]),
                KspaceTruncd(keys=["L"], trunc_factor=3.0, norm_val=1.0, slice_dim=1),
            ])

        # Patch cropping
        self.border_hr = opt['up_factor'] * 2  # 2 for 1x, 4 for 2x and 8 for 4x
        self.border_lr = self.border_hr // opt['up_factor']

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

        if self.implicit:
            self.random_crop_pair = RandomCropPairImplicitd(self.size_lr, self.up_factor, self.lr_foreground_threshold, mode)
        else:
            crop_pair_mode = "crop_uniform"  # "crop_foreground"
            self.random_crop_pair = RandomCropPaird(self.size_lr, self.up_factor, self.lr_foreground_threshold, crop_pair_mode, self.pad_size)


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
                self.divisible_transform, #self.border_crop,
                #self.pad_transform,  # pad HR
                self.degradation,
                self.pad_transform,  # pad LR
                # Random transforms
                #RandomCropPaird(self.size_lr, self.up_factor)
                self.random_crop_pair

            ]
        )

        return transforms


class RandomCropPaird(Randomizable):

    def __init__(self, patch_size_lr, up_factor, lr_foreground_threshold, crop_pair_mode="crop_uniform", pad_size=0):
        super().__init__()
        self.size_lr = patch_size_lr
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * self.up_factor
        self.lr_foreground_threshold = lr_foreground_threshold
        self.threshhold_percentage = 0.10  # 10%
        self.pad_size = pad_size
        if pad_size > 0:
            self.center_size = self.size_lr - 2*self.pad_size
            self.size_lr = self.size_lr
            self.size_hr = self.size_hr - 2*self.up_factor*self.pad_size

        # Set crop function based on upscaling factor
        if crop_pair_mode == "crop_uniform":
            self.crop_func = self.crop_uniform
        else:
            self.crop_func = self.crop_foreground


    def __call__(self, img_dict: dict):
        return self.crop_func(img_dict)

    def randomize(self):
        self.sub_seed = self.R.randint(999999, dtype="uint32")

    def crop_uniform(self, img_dict: dict):

        # Get valid index range of LR image
        self.valid_range_lr = torch.tensor(img_dict['L'].shape[1:]) - self.size_lr

        # Sample uniform random indexes in valid range in LR
        i_lr, j_lr, k_lr = tuple(int(np.random.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range_lr)

        # Get corresponding position in HR
        i_hr, j_hr, k_hr = i_lr*self.up_factor, j_lr*self.up_factor, k_lr*self.up_factor

        # Crop corresponding HR and LR patch
        H = img_dict['H'][:, i_hr:i_hr + self.size_hr, j_hr:j_hr + self.size_hr, k_hr:k_hr + self.size_hr]
        L = img_dict['L'][:, i_lr:i_lr + self.size_lr, j_lr:j_lr + self.size_lr, k_lr:k_lr + self.size_lr]

        img_dict = {'H': H.float(), 'L': L.float()}

        return img_dict

    def crop_foreground(self, img_dict: dict):

        # Get valid index range of LR image
        self.valid_range_lr = torch.tensor(img_dict['L'].shape[1:]) - self.size_lr

        foreground_fraction = self.size_lr ** 3 * self.threshhold_percentage
        for _ in range(10):  # Maximum resample 10 times
            # Sample uniform random indexes in valid range in LR
            i_lr, j_lr, k_lr = tuple(int(np.random.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range_lr)

            # Check if sample has at least X percent foreground
            L = img_dict['L'][:, i_lr:i_lr + self.size_lr, j_lr:j_lr + self.size_lr, k_lr:k_lr + self.size_lr]
            if torch.sum(L > self.lr_foreground_threshold) > foreground_fraction:
                break

        # Get corresponding position in HR
        i_hr, j_hr, k_hr = i_lr * self.up_factor, j_lr * self.up_factor, k_lr * self.up_factor

        # Crop corresponding HR and LR patch
        H = img_dict['H'][:, i_hr:i_hr + self.size_hr, j_hr:j_hr + self.size_hr, k_hr:k_hr + self.size_hr]

        img_dict = {'H': H, 'L': L}

        return img_dict

class RandomCropPairImplicitd(Randomizable):

    def __init__(self, patch_size_lr, up_factor, lr_foreground_threshold, mode):
        super().__init__()
        self.size_lr = patch_size_lr
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * self.up_factor
        self.mode = mode
        self.sample_size = 8000 if mode == "train" else -1
        self.lr_foreground_threshold = lr_foreground_threshold

        if self.mode == "train":
            # Precompute HR shape and HR coordinate grid for training
            self.hr_shape = (torch.tensor([24, 24, 24], dtype=torch.int) * up_factor).tolist()
            self.lr_shape = (torch.tensor([24, 24, 24], dtype=torch.int)).tolist()
            self.coord_grid = self._precompute_coord(self.hr_shape)  # Precompute coord grid only once
    @staticmethod
    def make_coord(shape):
        """
        Generate the coordinate grid for a given shape.
        """
        ranges = [-1, 1]
        coord_seqs = [torch.linspace(ranges[0] + (1 / (2 * n)), ranges[1] - (1 / (2 * n)), n, device='cuda') for n in shape]
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        return ret.view(-1, ret.shape[-1])
    def _precompute_coord(self, hr_shape):
        """
        Precompute and cache coordinate grid for training mode.
        """
        return self.make_coord(hr_shape)

    def randomize(self):
        self.sub_seed = self.R.randint(999999, dtype="uint32")

    def __call__(self, img_dict: dict):

        # Get valid index range of LR image
        self.valid_range_lr = torch.tensor(img_dict['L'].shape[1:]) - self.size_lr
        
        # Sample uniform random indexes in valid range in LR
        i_lr, j_lr, k_lr = tuple(int(self.R.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range_lr)

        # Get corresponding position in HR
        i_hr, j_hr, k_hr = i_lr * self.up_factor, j_lr * self.up_factor, k_lr * self.up_factor

        if self.mode == "train":
            hr_h, hr_w, hr_d = self.hr_shape
            lr_h, lr_w, lr_d = self.lr_shape

            # Crop the high-resolution patch
            patch_hr = img_dict['H'][:, i_hr:i_hr + hr_h, j_hr:j_hr + hr_w, k_hr:k_hr + hr_d]
    
            # Simulate low-resolution patch
            patch_lr = img_dict['L'][:, i_lr:i_lr + lr_h, j_lr:j_lr + lr_w, k_lr:k_lr + lr_d]

            # Use the precomputed coordinate grid
            xyz_hr = self.coord_grid

            # Filter background points
            pos_indices = patch_hr > self.lr_foreground_threshold
            sorted_patch_hr = patch_hr[pos_indices]
            sorted_xyz_hr = xyz_hr[pos_indices.reshape(-1), :]

            # Check if we have fewer than 8000 points and add more if necessary
            if sorted_xyz_hr.shape[0] < self.sample_size:
                extra_indices = torch.randperm(xyz_hr.shape[0], device=patch_hr.device)[:self.sample_size - sorted_xyz_hr.shape[0]]
                sorted_xyz_hr = torch.cat([sorted_xyz_hr, xyz_hr[extra_indices]], dim=0)
                sorted_patch_hr = torch.cat([sorted_patch_hr, patch_hr.reshape(-1)[extra_indices]], dim=0)
    
            # Sample random indices once
            sample_indices = torch.randperm(sorted_xyz_hr.shape[0], device=patch_hr.device)[:self.sample_size]
            xyz_hr = sorted_xyz_hr[sample_indices]
            patch_hr = sorted_patch_hr.reshape(-1, 1)[sample_indices]
        else:
            # For testing, just reshape the entire patch
            patch_hr = img_dict['H'][:, i_hr:i_hr + self.size_hr, j_hr:j_hr + self.size_hr, k_hr:k_hr + self.size_hr]
            patch_hr = patch_hr[0].reshape(-1, 1)
            patch_lr = img_dict['L'][:, i_lr:i_lr + self.size_lr, j_lr:j_lr + self.size_lr, k_lr:k_lr + self.size_lr]
            xyz_hr = self.make_coord(patch_hr.shape)

        return {'L': patch_lr, 'H_xyz': xyz_hr, 'H': patch_hr}

class GaussianblurImaged():
    def __init__(self, blur_sigma, **kwargs):

        self.in_key = kwargs['in_key']
        self.out_key = kwargs['out_key']

        self.blur_sigma = blur_sigma
        self.radius = math.ceil(3 * self.blur_sigma)

    def __call__(self, img_dict):
        img_dict[self.out_key] = test_3d_gaussian_blur(img_dict[self.in_key].squeeze(), ks=int(2 * self.radius + 1), blur_sigma=self.blur_sigma)
        return img_dict

