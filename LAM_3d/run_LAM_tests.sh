#!/bin/bash

python -m LAM_3d_anymodel --model_name "EDDSR" --window_size 48 --cube_no "002" --h 50 --w 40 --d 40

python -m LAM_3d_anymodel --model_name "mDCSRN" --window_size 48 --cube_no "002" --h 50 --w 40 --d 40

python -m LAM_3d_anymodel --model_name "MFER" --window_size 48 --cube_no "002" --h 50 --w 40 --d 40

python -m LAM_3d_anymodel --model_name "SuperFormer" --window_size 48 --cube_no "002" --h 50 --w 40 --d 40

python -m LAM_3d_anymodel --model_name "RRDBNet3D" --window_size 48 --cube_no "002" --h 50 --w 40 --d 40

python -m LAM_3d_anymodel --model_name "ArSSR" --window_size 48 --cube_no "002" --h 50 --w 40 --d 40

python -m LAM_3d_anymodel --model_name "MTVNet" --window_size 48 --cube_no "002" --h 50 --w 40 --d 40
