import os
import nibabel as nib
import numpy as np
from monai.data import create_test_image_3d

dir = "../testimages"
num_img = 2
print(f"generating synthetic data to {dir} (this may take a while)")

for i in range(num_img):
    im, seg = create_test_image_3d(1000, 1000, 1000, num_seg_classes=1, channel_dim=-1)
    print(np.max(im), np.min(im))
    print(np.max(seg), np.min(seg))
    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(dir, f"img_HR{i+3:d}.nii"))
    print("Image %d" % i)

