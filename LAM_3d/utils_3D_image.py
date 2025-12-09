import glob
import os
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
#from dataset import EM_DATA
#from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm_notebook as tqdm
import torch
import torch.nn as nn
import nibabel as nib
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
#matplotlib inline
#from PIL import Image
import cv2 as cv
import PIL.Image as Image
import torchio as tio
import torchio.transforms as tiotransforms

#import config

#import nifti2dicom as nii2dcm
import SimpleITK as sitk
import math

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
        #TIO_UPSIZE_NEAREST = tiotransforms.Resize(target_shape=(patch_size_hr), image_interpolation='NEAREST')
        #TIO_UPSIZE_LINEAR = tiotransforms.Resize(target_shape=(patch_size_hr), image_interpolation='LINEAR')
        #TIO_UPSIZE_BSPLINE = tiotransforms.Resize(target_shape=(patch_size_hr), image_interpolation='BSPLINE')

        if method == "tio_linear":
            return tiotransforms.Resize(target_shape=size, image_interpolation='LINEAR')
        elif method == "tio_nearest":
            return tiotransforms.Resize(target_shape=size, image_interpolation='NEAREST')
        elif method == "tio_bspline":
            return tiotransforms.Resize(target_shape=size, image_interpolation='BSPLINE')
        else:
            raise NotImplementedError('Upsampling method %s not implemented.' % method)


    def get_comparison_image(self, img_dict, slice_idx=None):
        # TODO make this function support parsing multiple grid_images when len(slice_idx) > 1 to improve efficiency.
        if slice_idx is None:
            slice_idx = img_dict['H'].shape[-1] // 2
            #slice_idx = self.patch_size_hr // 2  # Assumes volumetric patch is cubic
            #slice_idx = 48

        # Upscale LR volumes and extract slice
        img_list = []
        for key, func in self.upscale_func_dict.items():
            if img_dict['H'].shape != img_dict['L'].shape:  # TODO: Fix hardcoding here
                up_lr_slice = func(img_dict['L'].cpu())[:, :, :, slice_idx]

            else:
                up_lr_slice = img_dict['L'].cpu()[:, :, :, slice_idx]

            img_list.append(up_lr_slice)

        hr_slice = img_dict['H'][:, :, :, slice_idx].cpu() # C, H, W, D -> C, H, W
        sr_slice = img_dict['E'][:, :, :, slice_idx].cpu()

        #print("Unnorm and rescale image slices here?")

        img_list.append(sr_slice)
        img_list.append(hr_slice)

        row = torch.stack(img_list)
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)  # make grid, then permute to H, W, C because WandB assumes channel last
        # uint8_image = utils_image.tensor2uint(grid)  # Clamps to [0; 1], then converts to uint8 image in range [0; 255]
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

    #i_min = np.min(images)
    #i_max = np.max(images)
    if False:
        for i in range(3):
            i_min = np.min(images[:,:,i])
            i_max = np.max(images[:,:,i])
            images[:,:,i] = (images[:,:,i] - i_min) / (i_max - i_min)

    #images[images > 1] = 1

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
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    if len(img.size()) > 2:
        return np.transpose(npimg, (1, 2, 0))
    else:
        return npimg

# def create_output_folders():
#     # create output folders
#     if os.path.exists(os.path.join("saved_models/" + config.OUT_DIR_NAME, config.EXPERIMENT_NAME)):
#         print("Error: Directory", config.EXPERIMENT_NAME, "already exits. Please choose another experiment ID.")
#         print("Stopping execution...")
#         raise SystemExit(0)

#     os.mkdir(os.path.join("saved_models/" + config.OUT_DIR_NAME, config.EXPERIMENT_NAME))
#     os.mkdir(os.path.join("saved_loss_dicts/" + config.OUT_DIR_NAME, config.EXPERIMENT_NAME))
#     new_model_outputs_path = os.path.join("model_outputs/" + config.OUT_DIR_NAME, config.EXPERIMENT_NAME)
#     os.mkdir(new_model_outputs_path)
#     os.mkdir(os.path.join(new_model_outputs_path, "test"))
#     os.mkdir(os.path.join(new_model_outputs_path, "train"))
#     os.mkdir(os.path.join(new_model_outputs_path, "loss_plots"))

#     # Example usage
#     create_parameter_file("Epochs: %d" % config.EPOCHS,
#                           "Patch size HR: %d" % config.PATCH_SIZE_HR,
#                           "Patch size LR: %d" % config.PATCH_SIZE,
#                           "Upscaling factor: %d" % config.UP_FACTOR,
#                           "Batch size: %d" % config.BATCH_SIZE,
#                           "Learning rate: %0.4f" % config.LEARNING_RATE,
#                           "K factor: %d" % config.K_FACTOR,
#                           config.LOSS_WEIGHTS)

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

def gdx(vol, g, gd, prepend_one, padding):
    #### TESTING HERE ####
    if padding == "valid":
        k_test = gd[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian dev
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
        k_test = g[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
        k_test = g[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
    else:
        k_test = gd[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian dev
        vol = F.conv3d(vol, k_test, stride=1, padding=(len(gd) // 2, 0, 0))
        k_test = g[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding=(0, len(g) // 2, 0))
        k_test = g[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding=(0, 0, len(g) // 2))

    if prepend_one:
        return vol.unsqueeze(0)
    else:
        return vol

def gdy(vol, g, gd, prepend_one, padding):
    #### TESTING HERE ####
    if padding == "valid":
        k_test = gd[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian dev
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
        k_test = g[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
        k_test = g[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
    else:
        k_test = gd[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian dev
        vol = F.conv3d(vol, k_test, stride=1, padding=(0, len(gd) // 2, 0))
        k_test = g[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding=(len(g) // 2, 0, 0))
        k_test = g[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding=(0, 0, len(g) // 2))

    if prepend_one:
        return vol.unsqueeze(0)
    else:
        return vol

def gdz(vol, g, gd, prepend_one, padding):
    #### TESTING HERE ####
    if padding == "valid":
        k_test = gd[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian dev
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
        k_test = g[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
        k_test = g[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding="valid")
    else:
        k_test = gd[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian dev
        vol = F.conv3d(vol, k_test, stride=1, padding=(0, 0, len(gd) // 2))
        k_test = g[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding=(len(g) // 2, 0, 0))
        k_test = g[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol = F.conv3d(vol, k_test, stride=1, padding=(0, len(g) // 2, 0))

    if prepend_one:
        return vol.unsqueeze(0)
    else:
        return vol

def laplacian3D(vol, g, gdd, prepend_one, padding):

    Lxx = gdx(vol.unsqueeze(0), g, gdd, prepend_one=False, padding=padding)
    Lyy = gdy(vol.unsqueeze(0), g, gdd, prepend_one=False, padding=padding)
    Lzz = gdz(vol.unsqueeze(0), g, gdd, prepend_one=False, padding=padding)

    L = Lxx + Lyy + Lzz
    if prepend_one:
        return L.unsqueeze(0)
    else:
        return L


def gauss_dev_x(vol, blur_sigma, prepend_one, padding):

    t = blur_sigma ** 2
    radius = math.ceil(3.0 * blur_sigma)
    ks = int(2 * radius + 1)
    x = np.arange(-radius, radius + 1)
    if vol.dtype == torch.double:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).double()
        k_dev = torch.from_numpy(gauss_dev_1D(t)).double()
    else:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).float()
        k_dev = torch.from_numpy(gauss_dev_1D(t)).float()
    # k = torch.from_numpy(cv.getGaussianKernel(ks, blur_sigma)).squeeze().float()

    # Separable 1D convolution in 3 directions
    k1d = k.view(1, 1, -1)
    if vol.device.type == "cuda":
        #k1d = k1d.to(DCSRN_config.DEVICE)
        k = k.to(DCSRN_config.DEVICE)
        k_dev = k_dev.to(DCSRN_config.DEVICE)

    k1d_dev = k_dev.view(1, 1, -1)
    vol_in_test = vol[None, None, ...]  ## make into [1,1,h,w,d]
    #### TESTING HERE ####
    if padding == "valid":
        k_test = k_dev[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian dev
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")

        k_test = k[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")

        k_test = k[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")
    else:
        k_test = k_dev[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian dev
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(len(k_dev) // 2, 0, 0))

        k_test = k[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(0, len(k) // 2, 0))

        k_test = k[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(0, 0, len(k) // 2))

    vol = vol_in_test[0, 0, :, :, :]

    # vol_in = vol[None, None, ...]
    # # k2d = torch.einsum('i,j->ij', k, k)
    # # k2d = k2d / k2d.sum() # not necessary if kernel already sums to zero, check:
    # # print(f'{k2d.sum()=}')
    # for i in range(3):
    #     if i == 0:
    #         kernel = k_dev[None, None, :, None, None]
    #     else:
    #         kernel = k[None, None, :, None, None]
    #     vol_in = vol_in.permute(0, 1, 4, 2, 3)
    #     vol_in = F.conv3d(vol_in, kernel, stride=1, padding=(len(k) // 2, 0, 0))

    ### THIS IS ALSO FINE BUT MAYBE THE DIRECTIONS HAVE BEEN SWAPPED ###
    ## patch is (x, y, z)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d_dev, padding=len(k_dev) // 2).view(*vol.shape)  # gaussian dev in x
    #vol = vol.permute(1, 2, 0)  # patch is now (y, z, x)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=len(k) // 2).view(*vol.shape)  # gaussian in y
    #vol = vol.permute(1, 2, 0)  # patch is now (z, x, y)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=len(k) // 2).view(*vol.shape)  # gaussian in z
    #vol = vol.permute(1, 2, 0)  # patch is now (x, y, z)

    #vol = vol.permute(1, 2, 0)  # patch is now (y, z, x)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)  # gaussian in y
    #vol = vol.permute(1, 2, 0)  # patch is now (z, x, y)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)  # gaussian in z
    #vol = vol.permute(1, 2, 0)  # patch is now (x, y, z)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d_dev, padding=ks // 2).view(*vol.shape)  # gaussian dev in x

    if prepend_one:
        return vol.reshape(1, *vol.shape)
    else:
        return vol

def gauss_dev_y(vol, blur_sigma, prepend_one, padding):

    t = blur_sigma ** 2
    radius = math.ceil(3 * blur_sigma)
    ks = int(2 * radius + 1)
    x = np.arange(-radius, radius + 1)
    if vol.dtype == torch.double:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).double()
        k_dev = torch.from_numpy(gauss_dev_1D(t)).double()
    else:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).float()
        k_dev = torch.from_numpy(gauss_dev_1D(t)).float()

    # Separable 1D convolution in 3 directions
    k1d = k.view(1, 1, -1)
    if vol.device.type == "cuda":
        # k1d = k1d.to(DCSRN_config.DEVICE)
        k = k.to(DCSRN_config.DEVICE)
        k_dev = k_dev.to(DCSRN_config.DEVICE)

    #k_dev = torch.from_numpy(gauss_dev_1D(t)).float()

    k1d_dev = k_dev.view(1, 1, -1)

    #### TESTING HERE ####
    if padding == "valid":
        vol_in_test = vol[None, None, ...]  ## make into [1,1,h,w,d]
        k_test = k_dev[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian dev
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")

        k_test = k[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")

        k_test = k[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")
    else:
        vol_in_test = vol[None, None, ...]  ## make into [1,1,h,w,d]
        k_test = k_dev[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian dev
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(0, len(k_dev) // 2, 0))

        k_test = k[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(len(k) // 2, 0, 0))

        k_test = k[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(0, 0, len(k) // 2))

    vol = vol_in_test[0,0,:,:,:]

    ## patch is (x, y, z)
    #vol = vol.permute(1, 2, 0)  # patch is now (y, z, x)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d_dev, padding=len(k_dev) // 2).view(*vol.shape)  # gaussian dev in y
    #vol = vol.permute(1, 2, 0)  # patch is now (z, x, y)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=len(k) // 2).view(*vol.shape)  # gaussian in z
    #vol = vol.permute(1, 2, 0)  # patch is now (x, y, z)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=len(k) // 2).view(*vol.shape)  # gaussian in x

    #vol = vol.permute(2, 0, 1)  # patch is now (x, y, z)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)  # gaussian in x
    #vol = vol.permute(2, 0, 1)  # patch is now (z, x, y)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)  # gaussian in z
    #vol = vol.permute(2, 0, 1)  # patch is now (y, z, x)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d_dev, padding=ks // 2).view(*vol.shape)  # gaussian dev in y
    #vol = vol.permute(2, 0, 1)  # patch is now (x, y, z)

    if prepend_one:
        return vol.reshape(1, *vol.shape)
    else:
        return vol

def gauss_dev_z(vol, blur_sigma, prepend_one, padding):

    t = blur_sigma ** 2
    radius = math.ceil(3 * blur_sigma)
    ks = int(2 * radius + 1)
    x = np.arange(-radius, radius + 1)
    if vol.dtype == torch.double:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).double()
        k_dev = torch.from_numpy(gauss_dev_1D(t)).double()
    else:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).float()
        k_dev = torch.from_numpy(gauss_dev_1D(t)).float()

    # Separable 1D convolution in 3 directions
    k1d = k.view(1, 1, -1)
    if vol.device.type == "cuda":
        # k1d = k1d.to(DCSRN_config.DEVICE)
        k = k.to(DCSRN_config.DEVICE)
        k_dev = k_dev.to(DCSRN_config.DEVICE)

    #k_dev = torch.from_numpy(gauss_dev_1D(t)).float()

    k1d_dev = k_dev.view(1, 1, -1)

    #### TESTING HERE ####
    if padding == "valid":
        vol_in_test = vol[None, None, ...]  ## make into [1,1,h,w,d]
        k_test = k_dev[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian dev
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")

        k_test = k[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")

        k_test = k[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding="valid")
    else:
        vol_in_test = vol[None, None, ...]  ## make into [1,1,h,w,d]
        k_test = k_dev[None, None, None, None, :]  ## make intro [1,1,1,1,kernel] representing z gaussian dev
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(0, 0, len(k_dev) // 2))

        k_test = k[None, None, :, None, None]  ## make intro [1,1,kernel_dev,1,1] representing x gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(len(k) // 2, 0, 0))

        k_test = k[None, None, None, :, None]  ## make intro [1,1,1,kernel,1] representing y gaussian
        vol_in_test = F.conv3d(vol_in_test, k_test, stride=1, padding=(0, len(k) // 2, 0))

    vol = vol_in_test[0, 0, :, :, :]

    ## patch is (x, y, z)
    #vol = vol.permute(2, 0, 1)  # patch is now (z, x, y)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d_dev, padding=len(k_dev) // 2).view(*vol.shape)  # gaussian dev in z
    #vol = vol.permute(2, 0, 1)  # patch is now (y, z, x)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=len(k) // 2).view(*vol.shape)  # gaussian in y
    #vol = vol.permute(2, 0, 1)  # patch is now (x, y, z)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=len(k) // 2).view(*vol.shape)  # gaussian in x

    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)  # gaussian in x
    #vol = vol.permute(1, 2, 0)  # patch is now (y, z, x)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)  # gaussian in y
    #vol = vol.permute(1, 2, 0)  # patch is now (z, x, y)
    #vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d_dev, padding=ks // 2).view(*vol.shape)  # gaussian dev in z
    #vol = vol.permute(1, 2, 0)  # patch is now (x, y, z)

    if prepend_one:
        return vol.reshape(1, *vol.shape)
    else:
        return vol

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
    # k = torch.from_numpy(cv.getGaussianKernel(ks, blur_sigma)).squeeze().float()

    # Separable 1D convolution in 3 directions
    k1d = k.view(1, 1, -1)
    for _ in range(3):
        vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)
        vol = vol.permute(2, 0, 1)
    #print((vol_3d- vol).abs().max()) # something ~1e-7
    #print(torch.allclose(vol_3d, vol, atol=1e-6))
    vol = vol.reshape(1, *vol.shape)
    return vol

def downscaleNsave(subject_hr, data_path, lo_res_name="clinical_femur01_aligned_lo_res.nii"):

    raw_image_hr = subject_hr['one_image']['data']  #

    _, dim_x, dim_y, dim_z = raw_image_hr.shape
    resize = tiotransforms.Resize(target_shape=(dim_x // 4, dim_y // 4, dim_z // 4), image_interpolation='LINEAR')
    lo_res = resize(subject_hr)

    lo_res['one_image'].save(os.path.join(data_path, lo_res_name))
    lo_res['one_image'].save(os.path.join(data_path, (lo_res_name+".gz")))

    return lo_res

def load_patch_subjects(data_path, crop_start, crop_end):

    full_dicom_list = sorted(glob.glob(data_path + "/DICOM/hi_res/slice*.dcm"))
    n_img = len(full_dicom_list)

    # Define HR patch index limits
    low_idx = crop_start
    high_idx = n_img - DCSRN_config.PATCH_SIZE_HR - crop_end

    start_idx = np.random.randint(low_idx, high_idx, 1, dtype=int)
    end_idx = start_idx + DCSRN_config.PATCH_SIZE_HR
    idx_list = np.arange(start_idx, end_idx, 1, dtype=int)
    dicom_list_hr = np.array(full_dicom_list)[idx_list]

    idx_list_lr = idx_list[0:-1:4]
    dicom_list_lr = np.array(full_dicom_list)[idx_list_lr]

    hr_image = tio.ScalarImage(dicom_list_hr)
    lr_image = tio.ScalarImage(dicom_list_lr)

    sampler = tio.data.UniformSampler(DCSRN_config.PATCH_SIZE_HR)
    hr_patch = list(sampler(tio.Subject(one_image=tio.ScalarImage(tensor=hr_image.data.T)), 1))[0]
    area = hr_patch.location // 4
    #crop = tio.transforms.Crop(cropping=hr_patch.history[0].cropping)


    print("Done")


def load_patch_subjects_V2(data_path, crop_start, crop_end):
    full_dicom_list = sorted(glob.glob(data_path + "/DICOM/hi_res/slice*.dcm"))
    n_img = len(full_dicom_list)

    # Define HR patch index limits
    low_idx = crop_start
    high_idx = n_img - DCSRN_config.PATCH_SIZE_HR - crop_end

    start_idx = np.random.randint(low_idx, high_idx, 1, dtype=int)
    end_idx = start_idx + DCSRN_config.PATCH_SIZE_HR
    idx_list = np.arange(start_idx, end_idx, 1, dtype=int)
    dicom_list_hr = np.array(full_dicom_list)[idx_list]

    # Read dicom image series from hr stack
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_list_hr)
    sitk_image_hr = reader.Execute()
    hr_image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image_hr))

    transform_z_norm = transforms.Compose([tio.ZNormalization()])
    #hr_image = transform_z_norm(hr_image)

    sampler = tio.data.UniformSampler(DCSRN_config.PATCH_SIZE_HR)
    hr_patch = list(sampler(hr_image, num_patches=1))[0]
    hr_image = transform_z_norm(hr_image)
    crop = tio.transforms.Crop(cropping=hr_patch.history[-1].cropping)

    dicom_list_lr = [string.replace('hi', 'lo') for string in dicom_list_hr]
    reader.SetFileNames(dicom_list_lr)
    sitk_image_lr = reader.Execute()
    lr_image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image_lr))

    #lr_image = transform_z_norm(lr_image)
    lr_patch = crop(lr_image)
    lr_patch = transform_z_norm(lr_patch)

    if DCSRN_config.UP_FACTOR != 1:
        resize = tiotransforms.Resize(target_shape=(DCSRN_config.PATCH_SIZE_HR//DCSRN_config.UP_FACTOR),image_interpolation='LINEAR')
        lr_patch_downscaled = resize(lr_patch)
        return hr_patch, lr_patch, hr_image, lr_image, lr_patch_downscaled

    return hr_patch, lr_patch, hr_image, lr_image, lr_image

def load_patch_subjects_V3(data_path):
    full_dicom_list = sorted(glob.glob(data_path + "/DICOM/hi_res/slice*.dcm"))
    n_img = len(full_dicom_list)

    femur_id = "01_b_1"
    if femur_id == "01_b_1":
        femur_mean = torch.tensor(-701.5400, dtype=torch.float32)
        femur_std = torch.tensor(610.5200, dtype=torch.float32)
        clinical_femur_mean = torch.tensor(7424.5500, dtype=torch.float32)
        clinical_femur_std = torch.tensor(4048.8500, dtype=torch.float32)
        crop_start = 80
        crop_end = 20

    # Define HR patch index limits
    low_idx = crop_start
    high_idx = n_img - DCSRN_config.PATCH_SIZE_HR - crop_end

    start_idx = np.random.randint(low_idx, high_idx, 1, dtype=int)
    end_idx = start_idx + DCSRN_config.PATCH_SIZE_HR
    idx_list = np.arange(start_idx, end_idx, 1, dtype=int)
    dicom_list_hr = np.array(full_dicom_list)[idx_list]

    # Read dicom image series from hr stack
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_list_hr)
    sitk_image_hr = reader.Execute()

    # Read dicom image series from lr stack
    dicom_list_lr = [string.replace('hi', 'lo') for string in dicom_list_hr]
    reader.SetFileNames(dicom_list_lr)
    sitk_image_lr = reader.Execute()

    # Define sampling method
    sitk_image_segmentation = None
    if DCSRN_config.SAMPLING_METHOD == "uniform":
        sampler = tio.data.UniformSampler(DCSRN_config.PATCH_SIZE_HR)
    elif DCSRN_config.SAMPLING_METHOD == "weighted":
        dicom_list_segmentation = [string.replace('hi_res', 'seg') for string in dicom_list_hr]
        reader.SetFileNames(dicom_list_segmentation)
        sitk_image_segmentation = reader.Execute()
        #segmentation = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image_segmentation))
        sampler = tio.data.WeightedSampler(DCSRN_config.PATCH_SIZE_HR, probability_map="sampling_map")
    else:
        print("Sampling method undefined: Defaulting to uniform sampler")
        sampler = tio.data.UniformSampler(DCSRN_config.PATCH_SIZE_HR)

    # Sample and normalize a patch
    hr_patch, crop = get_normalized_patch(sitk_image_hr, sampler, None, DCSRN_config.NORMALIZATION_METHOD, femur_mean, femur_std, sitk_image_segmentation)
    lr_patch, _ = get_normalized_patch(sitk_image_lr, sampler, crop, DCSRN_config.NORMALIZATION_METHOD, clinical_femur_mean, clinical_femur_std, sitk_image_segmentation)

    # Resize lr patch if needed
    if DCSRN_config.UP_FACTOR != 1:
        resize = tiotransforms.Resize(target_shape=(DCSRN_config.PATCH_SIZE_HR//DCSRN_config.UP_FACTOR),image_interpolation='LINEAR')
        lr_patch = resize(lr_patch)

    return hr_patch, lr_patch


def get_normalized_patch(sitk_image, sampler, crop, normalization_method, femur_mean, femur_std, sitk_segmentation):
    if normalization_method == "scan_wise":  # Normalize using precomputed mean and standard deviation
        sitk_image = (sitk_image-femur_mean)/femur_std
        if sitk_segmentation == None:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image))
        else:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image), sampling_map=tio.LabelMap.from_sitk(sitk_segmentation))
        if crop == None:
            patch = list(sampler(image, num_patches=1))[0]
            crop = tio.transforms.Crop(cropping=patch.history[-1].cropping)  # -1 of znorm then crop, or 0 if crop then znorm
        else:
            patch = crop(image)
    elif normalization_method == "block_wise":  # Normalize based on the sampled block
        if sitk_segmentation == None:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image))
        else:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image), sampling_map=tio.LabelMap.from_sitk(sitk_segmentation))
        transform_z_norm = transforms.Compose([tio.ZNormalization()])
        image = transform_z_norm(image)  # Znorm before cropping
        if crop == None:
            patch = list(sampler(image, num_patches=1))[0]
            crop = tio.transforms.Crop(cropping=patch.history[-1].cropping)  # -1 of znorm then crop, or 0 if crop then znorm
        else:
            patch = crop(image)
    elif normalization_method == "patch_wise":  # Normalize based on the 3D patch
        if sitk_segmentation == None:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image))
        else:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image), sampling_map=tio.LabelMap.from_sitk(sitk_segmentation))
        if crop == None:
            patch = list(sampler(image, num_patches=1))[0]
            crop = tio.transforms.Crop(cropping=patch.history[-1].cropping)
        else:
            patch = crop(image)
        transform_z_norm = transforms.Compose([tio.ZNormalization()])
        patch = transform_z_norm(patch)  # Znorm after cropping
    else:  # No normalization
        if sitk_segmentation == None:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image))
        else:
            image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image), sampling_map=tio.LabelMap.from_sitk(sitk_segmentation))
        if crop == None:
            patch = list(sampler(image, num_patches=1))[0]
            crop = tio.transforms.Crop(cropping=patch.history[-1].cropping)  # -1 of znorm then crop, or 0 if crop then znorm
        else:
            patch = crop(image)

    return patch, crop


def load_patch_transform(subject):
    #full_dicom_list = sorted(glob.glob(data_path + "/DICOM/hi_res/slice*.dcm"))
    #n_img = len(full_dicom_list)
    # In case of femur01
    crop_start = 80
    crop_end = 320

    # Define HR patch index limits
    #low_idx = crop_start
    #high_idx = n_img - DCSRN_config.PATCH_SIZE_HR - crop_end

    #start_idx = np.random.randint(low_idx, high_idx, 1, dtype=int)
    #end_idx = start_idx + DCSRN_config.PATCH_SIZE_HR
    #idx_list = np.arange(start_idx, end_idx, 1, dtype=int)
    #dicom_list_hr = np.array(full_dicom_list)[idx_list]

    # Read dicom image series from hr stack
    #reader = sitk.ImageSeriesReader()
    #reader.SetFileNames(dicom_list_hr)
    #sitk_image_hr = reader.Execute()
    #hr_image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image_hr))

    #transform_z_norm = transforms.Compose([tio.ZNormalization()])
    #hr_image = transform_z_norm(hr_image)

    #sampler = tio.data.UniformSampler(DCSRN_config.PATCH_SIZE_HR)
    #hr_patch = list(sampler(hr_image, num_patches=1))[0]
    #crop = tio.transforms.Crop(cropping=hr_patch.history[1].cropping)

    #dicom_list_lr = [string.replace('hi', 'lo') for string in dicom_list_hr]
    #reader.SetFileNames(dicom_list_lr)
    #sitk_image_lr = reader.Execute()
    #lr_image = tio.Subject(one_image=tio.ScalarImage.from_sitk(sitk_image_lr))

    #lr_image = transform_z_norm(lr_image)

    #lr_patch = crop(lr_image)

    #if DCSRN_config.UP_FACTOR != 1:
    #    resize = tiotransforms.Resize(target_shape=(DCSRN_config.PATCH_SIZE_HR//DCSRN_config.UP_FACTOR),image_interpolation='LINEAR')
    #    lr_patch_downscaled = resize(lr_patch)

    #return hr_patch, lr_patch, hr_image, lr_image, lr_patch_downscaled

    return subject




def plot_8(image_patch_hr, image_patch_lr):

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 4, 1)
    plt.imshow(image_patch_lr[0, :, :, 20], cmap='gray')
    plt.subplot(2, 4, 5)
    plt.imshow(image_patch_hr[0, :, :, 20], cmap='gray')

    plt.subplot(2, 4, 2)
    plt.imshow(image_patch_lr[0, :, :, 40], cmap='gray')
    plt.subplot(2, 4, 6)
    plt.imshow(image_patch_hr[0, :, :, 40], cmap='gray')

    plt.subplot(2, 4, 3)
    plt.imshow(image_patch_lr[0, :, :, 60], cmap='gray')
    plt.subplot(2, 4, 7)
    plt.imshow(image_patch_hr[0, :, :, 60], cmap='gray')

    plt.subplot(2, 4, 4)
    plt.imshow(image_patch_lr[0, :, :, -1], cmap='gray')
    plt.subplot(2, 4, 8)
    plt.imshow(image_patch_hr[0, :, :, -1], cmap='gray')


def plot_ensemble(plot_whole_images=False):

    for i in range(3):
        hr_patch, lr_patch, hr_image, lr_image, lr_patch_downscaled = load_patch_subjects_V2(
            data_path=data_path + "/resliced_femur01/", crop_start=320, crop_end=80)
        # hr_patch, lr_patch, hr_image, lr_image, lr_patch_downscaled = load_patch_subjects_V2(data_path=data_path+"/femur_01/", crop_start=320, crop_end=80)
        # hr_patch = hr_image['one_image']['data']
        image_patch_lr = lr_image['one_image']['data']
        image_patch_hr = hr_image['one_image']['data']
        patch_lr = lr_patch['one_image']['data']
        patch_hr = hr_patch['one_image']['data']

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        from matplotlib import rcParams
        rcParams.update({'font.size': 12})
        if plot_whole_images:
            plt.figure(figsize=(5, 5))
            plt.title("Low-resolution")
            plt.xticks([]), plt.yticks([])
            plt.imshow(image_patch_lr[0, :, :, DCSRN_config.PATCH_SIZE_HR//2], cmap='gray')

            plt.figure(figsize=(5, 5))
            plt.title("High-resolution")
            plt.xticks([]), plt.yticks([])
            plt.imshow(image_patch_hr[0, :, :, DCSRN_config.PATCH_SIZE_HR//2], cmap='gray')

        plt.figure(figsize=(5, 5))
        plt.title("Low-resolution")
        plt.xticks([]), plt.yticks([])
        plt.imshow(patch_lr[0, :, :, DCSRN_config.PATCH_SIZE_HR // 2], cmap='gray')

        plt.figure(figsize=(5, 5))
        plt.title("High-resolution")
        plt.xticks([]), plt.yticks([])
        plt.imshow(patch_hr[0, :, :, DCSRN_config.PATCH_SIZE_HR // 2], cmap='gray')

    plt.show()



if __name__ == "__main__":

    upscale_factor = 2
    input = torch.randn(4, 8, 5, 5, 5)
    pixelshuffle = nn.PixelShuffle(upscale_factor)
    conv = nn.ConvTranspose3d(8, 8, 6, 2, 2, bias=0)
    output_no_ICNR = conv(input)
    weight = ICNR(conv.weight, initializer=nn.init.normal_,upscale_factor=upscale_factor, mean=0.0, std=0.02)
    conv.weight.data.copy_(weight)
    output = conv(input)
    #output = pixelshuffle(output)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(output_no_ICNR.detach().numpy()[0, 0, 5, :, :])
    plt.subplot(1, 2, 2)
    plt.imshow(output.detach().numpy()[0, 0, 5, :, :])
    plt.show()

    from dataset_DCSRN import nvg_func

    dataset_path = "3D_datasets/datasets/2022_QIM_52_Bone"
    test_path = "train"

    data_path = os.path.join(dataset_path, test_path)
    file_paths = glob.glob(os.path.join(data_path, "*.nii"))

    print(file_paths)

    #subject_hr = tio.Subject(one_image=tio.ScalarImage(os.path.join(data_path, "resliced_femur01/resliced_clinical_femur01_aligned_with_femur01.nii")))


    subject_hr = tio.Subject(one_image=tio.ScalarImage(
        os.path.join(data_path, "raw/cropped_high_res/cropped_2_femur_74_c 1_int16.nii")))
    # "C:\Users\augus\OneDrive\Dokumenter\GitHub\Vedrana_master_project\3D_datasets\datasets\2022_QIM_52_Bone\train\raw\cropped_high_res"
    #femur_01_b_1 = tio.Subject(
    #    hi_res=tio.ScalarImage(os.path.join(data_path + "/raw/cropped_high_res/", "cropped_femur_01_b 1_int16.nii")),
    #    lo_res=tio.ScalarImage(os.path.join(data_path + "/raw/resliced/01_b_1/", "aligned_resliced_clinical_femur_01_b_1.nii.gz")),
    #    seg=tio.LabelMap(os.path.join(data_path + "/raw/resliced/01_b_1/", "aligned_resliced_femur_01_b_1_segmentation_1.nii.gz")),
    #    name="femur_01_b_1")


    if False:
        lo_res = downscaleNsave(subject_hr, data_path, "femur_74_c_1_downscaled.nii")
        print("Done")

    if False:
        subject_4x_downscale = tio.Subject(one_image=tio.ScalarImage(os.path.join(data_path, "resliced_femur01/clinical_femur01_aligned_lo_res.nii")))
        print(np.array(subject_4x_downscale.shape)*4)

    if False:
        subject_lr = tio.Subject(one_image=tio.ScalarImage(os.path.join(data_path, "clinical_femur01.nii")))
        raw_image_lr = subject_lr['one_image']['data']  #

    if False:
        #in_dir = os.path.join(data_path, "femur01.nii")
        in_dir = os.path.join(data_path, "resliced_femur01/resliced_clinical_femur01_aligned_with_femur01.nii")
        out_dir = os.path.join(data_path, "resliced_femur01/DICOM/lo_res/")
        nii2dcm.nifti2dicom_1file(in_dir, out_dir)

    if False:
        in_dir = os.path.join(data_path, "resliced_femur01/clinical_femur01_aligned_lo_res.nii")
        out_dir = os.path.join(data_path, "resliced_femur01/DICOM/lo_res/")
        nii2dcm.nifti2dicom_1file(in_dir, out_dir)

    if False:
        sampler = tio.data.UniformSampler(DCSRN_config.PATCH_SIZE_HR)
        # t1_image = tio.ScalarImage(image_path)
        test_hr = list(sampler(subject_hr, 1))[0]
        test_lr = list(sampler(subject_lr, 1))[0]
        print("uniform sample time test")

    if False:
        full_dicom_list = sorted(glob.glob(os.path.join(data_path, "resliced_femur01/DICOM/lo_res/slice*.dcm")))
        n_img = len(full_dicom_list)
        low = 80
        high = n_img-40-20
        start_idx = np.random.randint(low, high, 1, dtype=int)
        end_idx = start_idx+DCSRN_config.PATCH_SIZE
        idx_list = np.arange(start_idx, end_idx, 1, dtype=int)
        dicom_list = np.array(full_dicom_list)[idx_list]
        subject_patch_lr = tio.Subject(one_image=tio.ScalarImage(dicom_list))
        image_patch_lr = subject_patch_lr['one_image']['data']
        print("patch dicom test: This is much faster than ")

    if False:
        idx_list = np.arange(start_idx*4, end_idx*4, 4, dtype=int)
        full_dicom_list = sorted(glob.glob(os.path.join(data_path, "resliced_femur01/DICOM/hi_res/slice*.dcm")))
        dicom_list = np.array(full_dicom_list)[idx_list]
        subject_patch_hr = tio.Subject(one_image=tio.ScalarImage(dicom_list))
        image_patch_hr = subject_patch_hr['one_image']['data']
        print("patch dicom test: This is much faster than ")
        print("Start idx", start_idx, ", end idx", end_idx)

    if True:
        #from structure_tensor import sample_patch
        #sample_patch(80, False, femur_01_b_1)
        hr_patch, lr_patch, hr_image, lr_image, lr_patch_downscaled = load_patch_subjects_V2(data_path=data_path + "/resliced_femur01/", crop_start=320, crop_end=80)
        #hr_patch, lr_patch, hr_image, lr_image, lr_patch_downscaled = load_patch_subjects_V2(data_path=data_path+"/femur_01/", crop_start=320, crop_end=80)
        #hr_patch = hr_image['one_image']['data']
        image_patch_lr = lr_image['one_image']['data']
        print("Test")

    #plot_8(hr_image['one_image']['data'], lr_image['one_image']['data'])

    #plot_8(hr_patch['one_image']['data'], lr_patch['one_image']['data'])

    #plt.figure()
    #plt.subplot(1, 3, 1)
    #plt.imshow(lr_patch_downscaled['one_image']['data'][0, :, :, 10], cmap='gray')
    #plt.subplot(1, 3, 2)
    #plt.imshow(lr_patch['one_image']['data'][0, :, :, 10 * 4], cmap='gray')
    #plt.subplot(1, 3, 3)
    #plt.imshow(hr_patch['one_image']['data'][0, :, :, 10 * 4], cmap='gray')

    plot_ensemble()

        #image = list(sampler(image, 1))[0]
        #image = image.one_image.data


    if False:
        subject_lr = tio.Subject(one_image=tio.ScalarImage(file_paths[0]))
        raw_image_lr = subject_lr['one_image']['data']  #

        _, dim_x, dim_y, dim_z = raw_image_lr.shape
        plt.figure(figsize=(16,9))
        plt.imshow(raw_image_lr[0,:,:,dim_z//2], cmap="gray")
        plt.figure(figsize=(16,9))
        plt.imshow(raw_image_lr[0, :, dim_y//2, :], cmap="gray")
        plt.figure(figsize=(16,9))
        plt.imshow(raw_image_lr[0, dim_x//2, :, :], cmap="gray")

        #img = nib.load(file_paths[2])
        subject_hr = tio.Subject(one_image=tio.ScalarImage(file_paths[2]))
        raw_image_hr = subject_hr['one_image']['data']  #

        _, dim_x, dim_y, dim_z = raw_image_hr.shape
        plt.figure(figsize=(16, 9))
        plt.imshow(raw_image_hr[0, :, :, dim_z // 2], cmap="gray")
        plt.figure(figsize=(16, 9))
        plt.imshow(raw_image_hr[0, :, dim_y // 2, :], cmap="gray")
        plt.figure(figsize=(16, 9))
        plt.imshow(raw_image_hr[0, dim_x // 2, :, :], cmap="gray")


    plt.show()

    print("Done")

    #torch.save(raw_image, 'raw_image.pt')

    #subject_hr = tio.Subject(one_image=tio.ScalarImage(tensor=torch.zeros(1, dim_x * 4, dim_y * 4, dim_z * 4)))
