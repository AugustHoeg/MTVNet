#!/bin/bash
python -u test_simpleV3.py --experiment_id "AugustNet_ps32_4x_ID007081_L1_100K" --cluster "DTU_HPC"


python -u test_simpleV3.py --experiment_id "SuperFormer_ps32_4x_ID000251_L1_100K" --cluster "DTU_HPC"


python -u test_simpleV3.py --experiment_id "mDCSRN_ps32_4x_ID000280_L1_100K" --cluster "DTU_HPC"


python -u test_simpleV3.py --experiment_id "MFER_ps32_4x_ID000270_L1_100K" --cluster "DTU_HPC"


#python -u test.py --experiment_id "" --cluster "DTU_HPC"


#python -u test.py --experiment_id "" --cluster "DTU_HPC"

