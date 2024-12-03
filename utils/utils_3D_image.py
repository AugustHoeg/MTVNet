
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchio.transforms as tiotransforms
from torchvision.utils import make_grid


def crop_center(image, center_size):
    B, C, H, W, D = image.shape

    start_H = (H - center_size) // 2
    start_W = (W - center_size) // 2
    start_D = (D - center_size) // 2

    # Crop the image
    image = image[:, :, start_H:start_H + center_size, start_W:start_W + center_size, start_D:start_D + center_size]

    return image


def crop_context(image, L, level_ratio):
    """
    Crops a 3D image tensor according to the specified rules.

    Parameters:
        image (torch.Tensor): The input 3D image tensor of shape (H, W, D).
        L (int): The cropping level.
        level_ratio (int): The ratio used to calculate the new dimensions. Can be 2 or 3.

    Returns:
        torch.Tensor: The cropped 3D image tensor.
    """
    if L == 1:
        return image
    if L < 1:
        raise ValueError("L must be at least 1")
    if level_ratio not in [2, 3]:
        raise ValueError("level_ratio must be either 2 or 3")

    B, C, H, W, D = image.shape

    # Compute new dimensions based on the level_ratio
    if level_ratio == 2:
        new_H = H // (level_ratio ** (L - 1))
        new_W = W // (level_ratio ** (L - 1))
        new_D = D // (level_ratio ** (L - 1))
    elif level_ratio == 3:
        new_H = int(H * (2 / level_ratio) ** (L - 1))
        new_W = int(W * (2 / level_ratio) ** (L - 1))
        new_D = int(D * (2 / level_ratio) ** (L - 1))

    # Compute the starting indices to crop the center
    start_H = (H - new_H) // 2
    start_W = (W - new_W) // 2
    start_D = (D - new_D) // 2

    # Crop the image
    image = image[:, :, start_H:start_H + new_H, start_W:start_W + new_W, start_D:start_D + new_D]

    return image


class ImageComparisonTool():
    def __init__(self, patch_size_hr, upscaling_methods, unnorm=True, div_max=False, out_dtype=np.uint8):
        self.patch_size_hr = patch_size_hr
        self.unnorm = unnorm
        self.div_max = div_max
        self.out_dtype = out_dtype
        self.upscaling_methods = upscaling_methods

        self.upscale_func_dict = {}
        for method in upscaling_methods:
            self.upscale_func_dict[method] = self.get_upscaling_func(method=method, size=patch_size_hr)


    def get_upscaling_func(self, method="tio_linear", size=None):

        if method == "tio_linear":
            return tiotransforms.Resize(target_shape=size, image_interpolation='LINEAR')
        elif method == "tio_nearest":
            return tiotransforms.Resize(target_shape=size, image_interpolation='NEAREST')
        elif method == "tio_bspline":
            return tiotransforms.Resize(target_shape=size, image_interpolation='BSPLINE')
        else:
            raise NotImplementedError('Upsampling method %s not implemented.' % method)


    def get_comparison_image(self, img_dict, slice_idx=None):
        if slice_idx is None:
            slice_idx = img_dict['H'].shape[-1] // 2

        # Upscale LR volumes and extract slice
        img_list = []
        for key, func in self.upscale_func_dict.items():
            if img_dict['H'].shape != img_dict['L'].shape:  # upscale LR volume to match HR
                up_lr_slice = func(img_dict['L'].cpu())[:, :, :, slice_idx]

            else:
                up_lr_slice = img_dict['L'].cpu()[:, :, :, slice_idx]

            img_list.append(up_lr_slice)

        hr_slice = img_dict['H'][:, :, :, slice_idx].cpu()  # C, H, W, D -> C, H, W
        sr_slice = img_dict['E'][:, :, :, slice_idx].cpu()

        img_list.append(sr_slice)
        img_list.append(hr_slice)

        row = torch.stack(img_list)
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)  # make grid, then permute to H, W, C because WandB assumes channel last
        grid_image = self.unnorm_and_rescale(grid, self.out_dtype)

        return grid_image

    def unnorm_and_rescale(self, img, out_dtype=np.uint8):

        if self.unnorm:
            img = (img/2) + 0.5  # unnormalize from [-1; 1] to [0; 1]

        if self.div_max:
            img = img / torch.max(img)  # Divide by max to ensure output is between 0 and 1
            img[img < 0] = 0

        img = torch.clamp(img, min=0.0, max=1.0)  # Clip values
        if out_dtype == torch.uint8:
            img = torch.squeeze((torch.round(img * 255)).type(torch.uint8))  # Convert to uint8
        elif out_dtype == np.uint8:
            img = (np.round(img.numpy() * 255)).astype(np.uint8).squeeze()  # Convert to numpy uint8
        elif out_dtype == np.uint16:
            img = (np.round(img.numpy() * 65535)).astype(np.uint16).squeeze()  # Convert to numpy uint16 (unsupported in torch)

        return img


def rescale255(images):

    images = images / np.max(images)  # Ensure output is between 0 and 1
    images[images < 0] = 0
    images = (images * 255).astype(np.uint8)  # Convert to uint8 in range [0; 255]

    return images

def unnormalize_image(img):
    """
    If the input images are normalized, which is generally recommended, they should be unnormalized before visualization
    This function unnormalizes an image by assuming zero mean and unit standard deviation.
    :param img: Input image (normalized)
    :return: unnormalized output image
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if len(img.size()) > 2:
        return np.transpose(npimg, (1, 2, 0))
    else:
        return npimg

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

def toggle_grad(model, on_or_off):
    # https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py#L674
    for param in model.parameters():
        param.requires_grad = on_or_off

def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

def ICNR3D(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_cubed = upscale_factor * upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_cubed == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by upscale_factor^3: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_cubed, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_cubed, dim=0)

def deconvICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    kernel_size = torch.tensor(tensor.shape[2:])
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[-1] % upscale_factor == 0, \
        ("The size of the kernel: "
         f"tensor.shape[0] = {tensor.shape[-1]}"
         " is not divisible by the upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel_size = kernel_size // upscale_factor
    sub_kernel = torch.empty(*tensor.shape[0:2], *sub_kernel_size)  # assumes 3D kernel
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return torch.nn.functional.interpolate(sub_kernel, mode='nearest', scale_factor=upscale_factor)


def read_parameters(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                parameters[key.strip()] = value.strip()
    return parameters

def gauss_dev_1D(t):
    s = np.sqrt(t)
    x = np.array(range(int(-3*s), int(3*s+1)))
    return -(x/t*np.sqrt(2*np.pi*t))*np.exp(-x*x/2*t)

def get_gaussian_kernel(sigma, also_dg=False, also_ddg=False, radius=None):
    # order only 0 or 1

    if radius is None:
        radius = max(int(4 * sigma + 0.5), 1)  # similar to scipy _gaussian_kernel1d but never smaller than 1
    x = torch.arange(-radius, radius + 1)

    sigma2 = sigma * sigma
    phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if also_dg:
        return phi_x, phi_x * -x / sigma2
    elif also_ddg:
        return phi_x, phi_x * -x / sigma2, phi_x * ((x**2/sigma2**2) - (1/sigma2))
    else:
        return phi_x

def test_3d_gaussian_blur(vol, ks, blur_sigma):

    t = blur_sigma**2
    radius = math.ceil(3 * blur_sigma)
    x = np.arange(-radius, radius + 1)
    if vol.dtype == torch.double:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).double()
    elif vol.dtype == torch.float:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).float()
    elif vol.dtype == torch.short:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).short()
    elif vol.dtype == torch.bfloat16:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).bfloat16()

    # Separable 1D convolution in 3 directions
    k1d = k.view(1, 1, -1)
    for _ in range(3):
        vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)
        vol = vol.permute(2, 0, 1)

    vol = vol.reshape(1, *vol.shape)
    return vol



