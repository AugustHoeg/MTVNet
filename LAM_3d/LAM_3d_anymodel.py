import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.BackProp import attribution_objective, Path_gradient, Path_gradient_ArSSR, make_coord
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.attributes import attr_grad
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.utils import vis_saliency, grad_abs_norm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run LAM_3d with specified model.")
parser.add_argument("--model_name", type=str, required=False,
                    help="Name of model to use (e.g., EDDSR, mDCSRN, MFER, RRDBNet3D, SuperFormer).")
parser.add_argument("--cube_no", type=str, required=False,
                    help="ID of cube for LAM analysis")
parser.add_argument("--h", type=int, required=False,
                    help="Height of LAM ROI")
parser.add_argument("--w", type=int, required=False,
                    help="Width of LAM ROI")
parser.add_argument("--d", type=int, required=False,
                    help="Depth of LAM ROI")
parser.add_argument("--window_size", type=int, required=False,
                    help="Size of LAM ROI.")

args = parser.parse_args()

seed_value = 8339
torch.manual_seed(seed_value)
np.random.seed(seed_value)

model_name = args.model_name
cube_no = args.cube_no
h = args.h
w = args.w
d = args.d
window_size = args.window_size

# Default arguments
if cube_no is None:
    cube_no = '002'
if window_size is None:
    window_size = 48  # Define window_size of D
if model_name is None:
    model_name = "mDCSRN"
if h is None:
    h = 50  # The y coordinate of your select patch
if w is None:
    w = 40  # The x coordinate of your select patch
if d is None:
    d = 40  # The z coordinate of your select patch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
up_factor = 4
upscale_dict = {4: '32', 2: '64', 1: '128'}
input_size = 32

# Load and initialize the specified model
if model_name == "EDDSR":
    from ModelZoo.EDDSR import EDDSR_xs, load_network
    model = EDDSR_xs(up_factor=up_factor).to(device)

elif model_name == "ArSSR":
    from ModelZoo.ArSSR import ArSSR, load_network
    model = ArSSR(encoder_name="ResCNN",
                  feature_dim=128,
                  decoder_depth=8,
                  decoder_width=256).to(device)

elif model_name == "mDCSRN":
    from ModelZoo.mDCSRN_GAN import MultiLevelDenseNet, load_network
    model = MultiLevelDenseNet(up_factor=up_factor,
                               in_c=1,
                               k_factor=12,
                               k_size=3,
                               num_dense_blocks=8,
                               upsample_method="pixelshuffle3D",
                               use_checkpoint=True).to(device)
elif model_name == "MFER":
    from ModelZoo.MFER_official import MFER_xs, load_network
    model = MFER_xs(up_factor=up_factor).to(device)

elif model_name == "RRDBNet3D":
    from ModelZoo.RRDBNet3D_official import RRDBNet, load_network
    model = RRDBNet(up_factor=up_factor,
                    in_nc=1,
                    out_nc=1,
                    nf=64,
                    nb=12,
                    gc=32).to(device)

elif model_name == "SuperFormer":
    from ModelZoo.SuperFormer import SuperFormer, load_network
    patch_size = int(upscale_dict[up_factor])
    model = SuperFormer(img_size=patch_size, patch_size=2, in_chans=1,
                        embed_dim=252, depths=[6, 6, 6], num_heads=[6, 6, 6],
                        window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, rpb=True, patch_norm=True,
                        use_checkpoint=True, upscale=up_factor, img_range=1.,
                        upsampler="pixelshuffle3D", resi_connection='1conv',
                        output_type="direct", num_feat=126, init_type="default").to(device)
elif model_name == "MTVNet":
    from ModelZoo.AugustNet import AugustNet, load_network
    input_size = 128
    pred_size = 32
    model = AugustNet(
        input_size=(128, 128, 128),
        up_factor=4,
        num_levels=3,
        context_sizes=[128, 64, 32],
        num_blks=[1, 2, 3],
        blk_layers=[6, 6, 6],
        in_chans=1,
        shallow_feats=128,
        pre_up_feats=[64, 64],
        ct_embed_dims=[128, 128, 128],
        embed_dims=[128, 128, 128],
        ct_size=4,
        ct_pool_method="conv",
        patch_sizes=[8, 4, 2],
        num_heads=4,
        attn_window_sizes=[8, 8, 8],
        enable_shift=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        token_upsample_method="deconv_nn_resize",
        upsample_method="pixelshuffle3D",
        use_checkpoint=True,
        layer_type="fastervit",
        enable_ape_ct=True,
        enable_ape_x=False,
        enable_ct_rpb=True,
        enable_conv_skip=True,
        enable_long_skip=True,
        patch_pe_method="window_relative"
    ).to(device)

else:
    raise ValueError(f"Unknown model_name '{model_name}'.")

print("Working dir:", os.getcwd())
# Load state_dict
load_network(f'saved_models/{model_name}/Synthetic_2022_QIM_52_Bone_4x/100000_G.h5', model, strict=False, param_key='params')


# %% Load test image
img_lr = np.load(f'saved_image_cubes/Synthetic_2022_QIM_52_Bone_4x/LR/cube_{input_size}_{cube_no}.npy')
img_hr = np.load(f'saved_image_cubes/Synthetic_2022_QIM_52_Bone_4x/HR/cube_{cube_no}.npy')
tensor_lr = torch.from_numpy(img_lr) ; tensor_hr = torch.from_numpy(img_hr)
cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 3) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 3)

print(img_hr.shape)
print(img_lr.shape)

# %% Show image
if model_name == "MTVNet":
    z_idx_lr = (2 * d + window_size) // (2 * up_factor) + (input_size - pred_size) // 2
else:
    z_idx_lr = d // up_factor + window_size // (2 * up_factor)
z_idx_hr = d+window_size//2
plt.imshow(cv2_hr[:,:,d+window_size//2,:],cmap='gray')
plt.imshow(cv2_lr[:,:,z_idx_lr,:],cmap='gray')

# %% Draw rectangle on slice
pil_hr = Image.fromarray((img_hr[0,:,:,z_idx_hr] * 255).astype(np.uint8))
pil_lr = Image.fromarray((img_lr[0,:,:,z_idx_lr] * 255).astype(np.uint8))
draw_img_slice = pil_to_cv2(pil_hr)
cv2.rectangle(draw_img_slice, (w, h), (w + window_size, h + window_size), (0, 0, 255), 1)
position_pil = cv2_to_pil(draw_img_slice)

# %% Calculate LAM
sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.1
attr_objective = attribution_objective(attr_grad, h, w, d, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
if model_name == "ArSSR":
    xyz_hr = make_coord(list(tensor_hr.shape[1:])).unsqueeze(0)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient_ArSSR(tensor_lr.numpy(), xyz_hr, model, attr_objective, gaus_blur_path_func, cuda=True)
else:
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
abs_normed_grad_numpy = grad_abs_norm(grad_numpy)

cube_dir = f"cube_{cube_no}_win{window_size}_h{h}-w{w}-d{d}"
if not os.path.exists("Results/" + cube_dir):
    os.makedirs("Results/" + cube_dir)
    os.makedirs("Results/" + cube_dir + "/lam_out")

np.save(f'Results/{cube_dir}/lam_out/angn_{model_name}_{cube_dir}.npy',abs_normed_grad_numpy)

# %% Make visualizations
if model_name == "MTVNet":
    result_im = Image.fromarray((result[0, 0, :, :, z_idx_hr] * 255).astype(np.uint8))
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy[48:-48, 48:-48, z_idx_lr], zoomin=1)
    angn_mean = np.mean(abs_normed_grad_numpy[48:-48, 48:-48, 48:-48], axis=2)
    angn_mean = (angn_mean - np.min(angn_mean)) / (np.max(angn_mean) - np.min(angn_mean))
    saliency_image_abs_mean = vis_saliency(angn_mean, zoomin=1)
    saliency_image_abs_zoom = vis_saliency(abs_normed_grad_numpy[48:-48, 48:-48, z_idx_lr], zoomin=4)
    blend_abs_and_sr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(result_im) * alpha)
    blend_abs_and_hr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)

    gini_index = gini(abs_normed_grad_numpy[48:-48, 48:-48, z_idx_lr])
else:
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy[:,:,z_idx_lr], zoomin=up_factor)
    angn_mean = np.mean(abs_normed_grad_numpy,axis=2)
    angn_mean = (angn_mean - np.min(angn_mean)) / (np.max(angn_mean) - np.min(angn_mean))
    saliency_image_abs_mean = vis_saliency(angn_mean, zoomin=up_factor)
    blend_abs_and_hr = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)

    gini_index = gini(abs_normed_grad_numpy[:, :, z_idx_lr])

diffusion_index = (1 - gini_index) * 100
print(f"The DI of this case is {diffusion_index}")

gini_index_mean = gini(angn_mean)
diffusion_index_mean = (1 - gini_index_mean) * 100
print(f"The DI_mean of this case is {diffusion_index_mean}")

# %% Show LAM
fig, axs = plt.subplots(1,3,figsize=(14,4))
axs[0].imshow(position_pil)
axs[1].imshow(saliency_image_abs)
axs[2].imshow(saliency_image_abs_mean)

# %% Save results
plt.figure()
plt.imshow(position_pil)
plt.axis('off')
plt.savefig(f'Results/{cube_dir}/selection_{cube_no}_h{h}-w{w}-d{d}.png', bbox_inches='tight', pad_inches=0)

plt.figure()
plt.imshow(saliency_image_abs)
plt.axis('off')
plt.savefig(f'Results/{cube_dir}/{model_name}_{cube_no}_h{h}-w{w}-d{d}.png', bbox_inches='tight', pad_inches=0)

plt.figure()
plt.imshow(saliency_image_abs_mean)
plt.axis('off')
plt.savefig(f'Results/{cube_dir}/{model_name}_{cube_no}_h{h}-w{w}-d{d}_mean.png', bbox_inches='tight', pad_inches=0)

plt.figure()
plt.imshow(blend_abs_and_hr)
plt.axis('off')
plt.savefig(f'Results/{cube_dir}/{model_name}_{cube_no}_h{h}-w{w}-d{d}_overlay.png', bbox_inches='tight', pad_inches=0)

# %% Write to file
with open(f'Results/{cube_dir}/LAM_DI.txt', 'a') as f:
    f.write(f'Diffusion index for {model_name}: {diffusion_index} ({cube_no}; selection: h{h}-w{w}-d{d})\n')
    f.write(f'Diffusion index (MEAN) for {model_name}: {diffusion_index_mean} ({cube_no}; selection: h{h}-w{w}-d{d})\n')

# %%
