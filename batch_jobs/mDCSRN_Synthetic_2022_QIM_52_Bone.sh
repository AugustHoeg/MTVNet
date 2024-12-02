#!/bin/bash	

#SBATCH --job-name=Titans_job
#SBATCH --output=batch_outputs/output_job-%J.out
#SBATCH --error=batch_errors/error_job-%J.out
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=128gb
#SBATCH --gres=gpu:Ampere:1
#SBATCH --mail-user=august.hoeg@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
source activate venv

## Define script here
python -u train.py --options_file train_mDCSRN_Synthetic_2022_QIM_52_Bone.json --cluster "TITANS"

python -u test_simpleV3.py --options_file train_mDCSRN_Synthetic_2022_QIM_52_Bone.json --cluster "TITANS"

echo "Done: $(date +%F-%R:%S)"
