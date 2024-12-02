import os
import glob
from data.train_transforms import Resize_transformsV2
from data.baseline_transforms import Resize_baseline_transformsV2
class Dataset_KIRBY21():
    def __init__(self, opt):
        self.opt = opt
        self.patch_size_hr = opt['datasets']['patch_size_hr']
        self.patch_size_lr = opt['datasets']['patch_size']
        self.degradation_type = opt['datasets']['degradation_type']

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/Kirby21/"
            self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train", "KKI2009-*/KKI2009-*-T2w.nii")))
            self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test", "KKI2009-*/KKI2009-*-T2w.nii")))
        elif opt['cluster'] == "TITANS":
            self.data_path = "/scratch/aulho/Python/3D_datasets/datasets/Kirby21/"
            self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train", "KKI2009-*-T2w.nii")))
            self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test", "KKI2009-*-T2w.nii")))
        else:  # Default is opt['cluster'] = "DTU_HPC"
            self.data_path = "../3D_datasets/datasets/Kirby21/"
            self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train", "KKI2009-*-T2w.nii")))
            self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test", "KKI2009-*-T2w.nii")))

    def get_file_paths(self):

        train_files = [{"H": img_HR} for img_HR in self.HR_train]
        test_files = [{"H": img_HR} for img_HR in self.HR_test]

        return train_files, test_files

    def get_transforms(self, mode):

        self.mode = mode

        # Define transforms for KIRBY21
        data_trans = Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms()

        return transforms

    def get_baseline_transforms(self, mode):

        self.mode = mode

        data_trans = Resize_baseline_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms()

        return transforms
