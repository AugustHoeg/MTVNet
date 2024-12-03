import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is the Project Root directory

# Flag for selecting model architecture if configuration file is not provided
MODEL_ARCHITECTURE = "SuperFormer"  # "ArSSR" "mDCSRN_GAN" "mDCSRN" "SuperFormer" "ESRGAN3D" "RRDBNet3D" "RCAN3D" "EDDSR" "MFER" "MTVNet"
