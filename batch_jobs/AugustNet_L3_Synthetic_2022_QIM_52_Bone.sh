#!/bin/bash
# Simple init script for Python on DTU HPC
# Patrick M. Jensen, patmjen@dtu.dk, 2022
# Modified by August Leander Hoeg, s173944@dtu.dk, 2023.

# Configuration
# This is what you should change for your setup
VENV_NAME=venv         # Name of your virtualenv (default: venv)
VENV_DIR=.             # Where to store your virtualenv (default: current directory)
PYTHON_VERSION=3.11.9  # Python version (default: 3.9.14)
CUDA_VERSION=12.1      # CUDA version (default: 11.6)
/
#BSUB -q gpua100
#BSUB -J August_XtremeCT_single_AugustNet_new_L3
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 72:00
# request 40GB of system-memory rusage=40
#BSUB -R "select[gpu40gb]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -u "august.hoeg@gmail.com"
#BSUB -B
#BSUB -N
#BSUB -oo batch_outputs/output_august_XtremeCT_%J.out
#BSUB -eo batch_errors/error_august_XtremeCT_%J.out

# Exits if any errors occur at any point (non-zero exit code)
set -e

# Load modules
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "numpy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "scipy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "matplotlib/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "pandas/")
#module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "cv2/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "os/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "glob/")
module load cuda/$CUDA_VERSION
CUDNN_MOD=$(module avail -o modulepath -t cudnn | grep "cuda-${CUDA_VERSION}" 2>&1 | sort | tail -n1)
if [[ ${CUDNN_MOD} ]]
then
    module load ${CUDNN_MOD}
fi

# Create virtualenv if needed and activate it
if [ ! -d "${VENV_DIR}/${VENV_NAME}" ]
then
    echo INFO: Did not find virtualenv. Creating...
    virtualenv "${VENV_DIR}/${VENV_NAME}"
fi
source "${VENV_DIR}/${VENV_NAME}/bin/activate"

echo "About to run scripts"

python -u train.py --options_file train_AugustNet_L3_Synthetic_2022_QIM_52_Bone.json --cluster "DTU_HPC"

python -u test_simpleV3.py --options_file train_AugustNet_L3_Synthetic_2022_QIM_52_Bone.json --cluster "DTU_HPC"

echo "Finished scripts"
