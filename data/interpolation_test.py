
import matplotlib.pyplot as plt
import torch
import nibabel as nib

import monai.transforms as mt
import torchio.transforms as tiotransforms

if __name__ == "__main__":

    #img = 1*torch.randn(1,64,64,64)
    #img = torch.clamp(img, min=0.0, max=1.0)
    img = nib.load('139839_3T_T1w_MPR1.nii')
    #img = nib.load('BraTS-GLI-00002-000-t1c.nii.gz')

    #img = nib.load('IXI002-Guys-0828-T1.nii')
    data = torch.Tensor(img.get_fdata()).unsqueeze(0)
    #data = data[:, :64, :64, :64]

    dim1 = data.shape[1]
    dim2 = data.shape[2]
    dim3 = data.shape[3]

    resize_func_2x = mt.Resized(keys='I', spatial_size=[dim1//2, dim2//2, dim3//2], mode='linear', anti_aliasing=True, anti_aliasing_sigma=1.0)
    resize_func_4x = mt.Resized(keys='I', spatial_size=[dim1//4, dim2//4, dim3//4], mode='linear', anti_aliasing=True, anti_aliasing_sigma=1.0)
    resize_func_8x = mt.Resized(keys='I', spatial_size=[dim1//8, dim2//8, dim3//8], mode='linear', anti_aliasing=True, anti_aliasing_sigma=1.0)

    print(data.shape)
    #data = data.permute(0, 3, 1, 2)  # z-direction first
    print(data.shape)
    print("Max and Min data", torch.max(data), torch.min(data))
    data = data / torch.max(data)

    i_dict = {'I': data}
    data_lr_2x = resize_func_2x(i_dict)['I']
    data_lr_4x = resize_func_4x(i_dict)['I']
    data_lr_8x = resize_func_8x(i_dict)['I']

    upsample_nearest = tiotransforms.Resize(target_shape=(dim1, dim2, dim3), image_interpolation='NEAREST')
    data_up_2x = upsample_nearest(data_lr_2x)
    data_up_4x = upsample_nearest(data_lr_4x)
    data_up_8x = upsample_nearest(data_lr_8x)

    norm_val = 1.0

    plt.figure(figsize=(18,5))
    plt.subplot(1, 4, 1)
    plt.imshow(data[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("Reference")

    plt.subplot(1, 4, 2)
    plt.imshow(data_up_2x[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("2x downsampling")

    plt.subplot(1, 4, 3)
    plt.imshow(data_up_4x[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("4x downsampling")

    plt.subplot(1, 4, 4)
    plt.imshow(data_up_8x[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("8x downsampling")

    # plt.subplot(1, 4, 3)
    # plt.imshow(out_tf_3[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    # plt.title("k-space truncation: %d%%" % 67)
    #
    # plt.subplot(1, 4, 4)
    # plt.imshow(out_tf_2p5[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    # plt.title("k-space truncation: %d%%" % 80)

    #plt.figure()
    #plot_histograms(out_tf_4, data)

    plt.show()


    print("Done")