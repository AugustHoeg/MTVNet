
# MTVNet 

### MTVNet: Mapping using Transformers for Volumes – Network for Super-Resolution with Long-Range Interactions [[Paper Link]](https://arxiv.org)
[August Høeg]([https://github.com/AugustHoeg](https://github.com/AugustHoeg)), [Sophia] and []

## Updates
- ✅ 2022-05-09: Release the first version of the paper at Arxiv.
- ✅ 2022-05-20: Release the codes, models and results of HAT.
- **(To do)** Add ...

## Overview
<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Performance_comparison.png" width="600"/>

**Benchmark results on SRx4 without ImageNet pretraining. Mulit-Adds are calculated for a 64x64 input.**
| Model | Params(M) | Multi-Adds(G) | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| [SwinIR](https://github.com/JingyunLiang/SwinIR) |   11.9    | 53.6 | 32.92 | 29.09 | 27.92 | 27.45 | 32.03 |
| HAT-S |   9.6    | 54.9 | 32.92 | 29.15 | 27.97 | 27.87 | 32.35 |
| HAT |   20.8    | 102.4 | 33.04 | 29.23 | 28.00 | 27.97 | 32.48 |

## Real-World SR Results
**Note that:**
- The default settings in the training configs (almost the same as Real-ESRGAN) are for training **Real_HAT_GAN_SRx4_sharper**.
- **Real_HAT_GAN_SRx4** is trained using similar settings without USM the ground truth.
- **Real_HAT_GAN_SRx4** would have better fidelity.
- **Real_HAT_GAN_SRx4_sharper** would have better perceptual quality.

**Results produced by** Real_HAT_GAN_SRx4_sharper.pth.

<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Visual_Results.png" width="800"/>

**Comparison with the state-of-the-art Real-SR methods.**

<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Comparison.png" width="800"/>

## Citations
#### BibTeX

    @InProceedings{chen2023activating,
        author    = {Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
        title     = {Activating More Pixels in Image Super-Resolution Transformer},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {22367-22377}
    }

    @article{chen2023hat,
      title={HAT: Hybrid Attention Transformer for Image Restoration},
      author={Chen, Xiangyu and Wang, Xintao and Zhang, Wenlong and Kong, Xiangtao and Qiao, Yu and Zhou, Jiantao and Dong, Chao},
      journal={arXiv preprint arXiv:2309.05239},
      year={2023}
    }

## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
python setup.py develop
```

## How To Test

Without implementing the codes, [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) is a nice tool to run our models.

Otherwise, 
- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1u2r4Lc2_EEeQqra2-w85Xg) (access code: qyrl).  
- Then run the following codes (taking `HAT_SRx4_ImageNet-pretrain.pth` as an example):
```
python hat/test.py -opt options/test/HAT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.  

- Refer to `./options/test/HAT_SRx4_ImageNet-LR.yml` for **inference** without the ground truth image.

**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/HAT_tile_example.yml`.**

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch
```
- Note that the default batch size per gpu is 4, which will cost about 20G memory for each GPU.  

The training logs and weights will be saved in the `./experiments` folder.

## Results
The inference results on benchmark datasets are available at
[Google Drive](https://drive.google.com/drive/folders/1t2RdesqRVN7L6vCptneNRcpwZAo-Ub3L?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1CQtLpty-KyZuqcSznHT_Zw) (access code: 63p5).


## Contact
If you have any question, please email chxy95@gmail.com or join in the [Wechat group of BasicSR](https://github.com/XPixelGroup/BasicSR#-contact) to discuss with the authors.
