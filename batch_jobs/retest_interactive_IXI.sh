#!/bin/bash
python -u test_simpleV3.py --experiment_id "AugustNet_ps32_4x_ID007082_L1_100K" --cluster "DTU_HPC"


python -u test_simpleV3.py --experiment_id "SuperFormer_ps32_4x_ID000252_L1_100K" --cluster "DTU_HPC"


python -u test_simpleV3.py --experiment_id "mDCSRN_ps32_4x_ID000281_L1_100K" --cluster "DTU_HPC"


python -u test_simpleV3.py --experiment_id "MFER_ps32_4x_ID000271_L1_100K" --cluster "DTU_HPC"


python -u test_simpleV3.py --experiment_id "EDDSR_ps32_4x_ID000261_L1_100K" --cluster "DTU_HPC"


#python -u test_simpleV3.py --experiment_id "" --cluster "DTU_HPC"

