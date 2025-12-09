import time

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
import math

import config
from models.models_3D import SRBlock3D
from utils.utils_3D_image import ICNR, numel

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, **kwargs):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.use_act:
            return self.act(self.conv(x))
        else:
            return self.conv(x)


class UpsampleBlock3D(nn.Module):
    def __init__(self, in_c, n, k_size=3, pad=1, upsample_method='nearest'):
        super().__init__()
        self.upsample_method = upsample_method

        if self.upsample_method == 'deconv_nn_resize':
            # self.deconv0 = nn.ConvTranspose3d(in_c, n, kernel_size=k_size, stride=2, padding=pad, bias=0)
            self.deconv0 = nn.ConvTranspose3d(in_c, n, kernel_size=k_size, stride=2, padding=pad)
            self.act0 = nn.LeakyReLU()  # ESRGAN uses Leaky ReLU instead of PReLU
            #self.act0 = nn.PReLU(num_parameters=n)

            # ICNR is an initialization method for sub-pixel convolution which removes checkerboarding
            # From the paper: Checkerboard artifact free sub-pixel convolution.
            # https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
            weight = ICNR(self.deconv0.weight, initializer=nn.init.normal_, upscale_factor=2, mean=0.0, std=0.02)
            self.deconv0.weight.data.copy_(weight)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode=upsample_method)  # Supports 3D natively
            self.conv = nn.Conv3d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True)
            self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        if self.upsample_method == 'deconv_nn_resize':
            # Taken from this paper, rightmost method in figure 3: https://arxiv.org/pdf/1812.11440.pdf
            x = self.deconv0(x)
            out = self.act0(x)
        else:
            out = self.act(self.conv(self.upsample(x)))

        return out


# class DenseResidualUnit(nn.Module):
#     def __init__(self, in_channels, channels=32, residual_beta=0.2):
#         super().__init__()
#         self.conv = nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1)
#         self.act = nn.LeakyReLU()
#         self.residual_beta = residual_beta
#
#     def forward(self, x):
#         self.act(self.conv(x))*self.residual_beta


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()

        self.residual_beta = residual_beta
        self.blk1 = ConvBlock3D(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True, use_act=True)
        self.blk2 = ConvBlock3D(in_channels + 32, 32, kernel_size=3, stride=1, padding=1, bias=True, use_act=True)
        self.blk3 = ConvBlock3D(in_channels + 2*32, 32, kernel_size=3, stride=1, padding=1, bias=True, use_act=True)
        self.blk4 = ConvBlock3D(in_channels + 3*32, 32, kernel_size=3, stride=1, padding=1, bias=True, use_act=True)

        self.conv = ConvBlock3D(in_channels + 4*32, 64, kernel_size=3, stride=1, padding=1, bias=True, use_act=False)
        #self.conv = nn.Conv3d(4*32, 64, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x):

        #x0 = self.blk1(x)
        #skip0 = torch.cat([x, x0], 1)

        #x1 = self.blk2(skip0)
        #skip1 = torch.cat([skip0, x1], 1)

        #x2 = self.blk3(skip1)
        #skip2 = torch.cat([skip1, x2], 1)

        #x3 = self.blk4(skip2)
        #final_skip = torch.cat([skip2, x3], 1)

        #out = self.conv(final_skip)

        x0 = self.blk1(x)
        x1 = self.blk2(torch.cat([x, x0], 1))
        x2 = self.blk3(torch.cat([x, x0, x1], 1))
        x3 = self.blk4(torch.cat([x, x0, x1, x2], 1))
        out = self.conv(torch.cat([x, x0, x1, x2, x3], 1))

        return out

class RRDB3D(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2, use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(3):
            self.blocks.append(
                DenseResidualBlock(in_channels)
            )

    def forward(self, x):

        new_inputs = x
        for block in self.blocks:
            if self.use_checkpoint:
                x = self.residual_beta * checkpoint.checkpoint(block, new_inputs) + new_inputs
            else:
                new_inputs = self.residual_beta * block(new_inputs) + new_inputs

        out = self.residual_beta * new_inputs + x
        #x1 = x + self.residual_beta * self.basic_blk1(x)
        #x2 = x1 + self.residual_beta * self.basic_blk2(x1)
        #x3 = x2 + self.residual_beta * self.basic_blk3(x2)

        #out = x + x3 * self.residual_beta

        return out

class RRDBNet3D(nn.Module):

    def __init__(self, up_factor=1, in_channels=1, num_channels=64, num_blocks=23, upsample_method='nearest', use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.up_factor = up_factor

        self.conv1 = ConvBlock3D(in_channels, num_channels,  kernel_size=3, stride=1, padding=1, bias=True, use_act=False)

        self.basic_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.basic_blocks.append(
                RRDB3D(num_channels, use_checkpoint=use_checkpoint)
            )

        self.conv2 = ConvBlock3D(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True, use_act=False)

        if upsample_method is None:
            # Direct Feature Combination from paper: https://arxiv.org/pdf/2003.01217v1.pdf
            self.recon = nn.Conv3d(num_channels, 1, kernel_size=1, stride=1, padding=0)
        else:
            # if up_factor >= 2:
            #     #self.SR0 = UpsampleBlock3D(num_channels, num_channels, upsample_method)
            #     self.SR0 = SRBlock3D(num_channels, num_channels, k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2)
            # if up_factor >= 4:
            #     #self.SR1 = UpsampleBlock3D(num_channels, num_channels, upsample_method)
            #     self.SR1 = SRBlock3D(num_channels, num_channels, k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2)
            # if up_factor == 1:
            #     self.recon0 = ConvBlock3D(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True, use_act=True)
            #
            # self.recon1 = ConvBlock3D(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True, use_act=True)
            # self.recon2 = nn.Conv3d(num_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)

            recon_feats = num_channels
            if up_factor >= 2:
                self.SR0 = SRBlock3D(num_channels, num_channels // 2, k_size=6, pad=2, upsample_method=upsample_method,
                                     upscale_factor=2, use_checkpoint=False)
                recon_feats = num_channels // 2
            if up_factor >= 4:
                self.SR1 = SRBlock3D(num_channels // 2, num_channels // 2, k_size=6, pad=2, upsample_method=upsample_method,
                                     upscale_factor=2, use_checkpoint=False)
                recon_feats = num_channels // 2

            self.recon1 = ConvBlock3D(recon_feats, recon_feats, kernel_size=3, stride=1, padding=1, bias=True, use_act=True)
            self.recon2 = nn.Conv3d(recon_feats, in_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, input):

        # Shallow feature extraction
        x = self.conv1(input)

        # Sequence of RRDB blocks
        for basic_block in self.basic_blocks:
            x = basic_block(x)

        # Skip connection before upsampling blocks
        x = self.conv2(x) + input

        # Upsampling blocks
        if self.up_factor == 1:
            x = self.recon0(x)  # If upscaling = 1 pass through conv3D + PreLU instead
        if self.up_factor >= 2:
            x = self.SR0(x)
        if self.up_factor == 4:
            x = self.SR1(x)

        # Reconstruction layers
        x = self.recon1(x)
        out = self.recon2(x)

        return out

class LRconvBlock3D(nn.Module):
    def __init__(self, input_size, in_c, n, k_size, stride, padding, bias):
        super().__init__()

        self.conv0 = nn.Conv3d(in_c, n, kernel_size=k_size, stride=stride, padding=padding, bias=bias)
        #self.norm0 = nn.LayerNorm([DCSRN_config.BATCH_SIZE, n, 16, 16, 16])
        self.norm0 = nn.LayerNorm(input_size)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        out = self.act0(self.norm0(x))

        return out


class ESRGAN3D_discriminator(nn.Module):
    def __init__(self, patch_size=64, up_factor=1, in_channels=1, features=[64, 64, 128, 128, 256, 256, 512, 512], dense_features=[1024, 1], base_block_type="ConvBlock3D", use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        if base_block_type == "ConvBlock3D":
            dim = [int(math.ceil(patch_size * up_factor / 2 ** i)) for i in range(1, 5)]
            for idx, feature in enumerate(features):
                self.blocks.append(
                    ConvBlock3D(
                        in_channels,
                        feature,
                        kernel_size=3,
                        stride=1 + idx % 2,  # set stride to 2 for every other block
                        padding=1,
                        bias=True,
                    )
                )
                in_channels = feature  # Set input features to output of previous block
        elif base_block_type == "LRconvBlock3D":
            dim = [int(math.ceil(patch_size * up_factor / 2 ** i)) for i in range(0, 4)]
            dim_idx = 0
            in_c = 1
            out_c = features[0]
            for idx in range(len(features) - 1):
                stride = 1 + (idx % 2)
                self.blocks.append(
                    LRconvBlock3D(
                        dim[dim_idx],
                        in_c,
                        out_c,
                        k_size=3,
                        stride=stride,  # set stride to 2 for every other block
                        padding=1,
                        bias=True
                    )
                )
                in_c = features[idx]
                out_c = features[idx + 1]
                dim_idx = dim_idx + (stride % 2)  # Set input features to output of previous block

        # dim = [int(math.ceil(patch_size*up_factor / 2 ** i)) for i in range(1, 5)]
        ll_size = int(features[-1] * dim[-1] ** 3)

        # Classifier
        #self.avg_pool = nn.AdaptiveAvgPool3d((6, 6, 6))  # AvgPool3d ensures output of conv blocks is the same for different patch sizes.
        self.flatten = nn.Flatten()
        self.dense0 = nn.Linear(ll_size, dense_features[0])
        self.act = nn.LeakyReLU(0.2, inplace=True)  # Should perhaps be nn.LeakyReLU(0.2, inplace=True)
        self.dense1 = nn.Linear(dense_features[0], dense_features[1])

        self.act_sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Pass through conv blocks
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        # Classification head
        #x = self.avg_pool(x)
        x = self.dense0(self.flatten(x))
        x = self.act(x)
        out = self.dense1(x)

        # Final sigmoid activation (Remember to remove if BCEWithLogitsLoss() is used in training loop)
        # out = self.act_sigmoid(out)

        return out

def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    up_factor = 4
    print("Test ESRGAN3D")
    generator = RRDBNet3D(up_factor=up_factor, in_channels=1, num_channels=64, num_blocks=10, upsample_method='pixelshuffle3D', use_checkpoint=False).to(device)

    from models.mDCSRN_GAN import Discriminator as netD
    discriminator = netD(input_size=128,
                         in_c=1,
                         n_conv_vec=[64, 64, 128, 128, 256, 256, 512, 512],
                         n_dense=[1024, 1],
                         k_size=3,
                         use_checkpoint=False).to(device)

    #discriminator = ESRGAN3D_discriminator(patch_size=32, up_factor=2, in_channels=1, features=[64, 64, 128, 128, 256, 256, 512, 512], dense_features=[1024, 1], base_block_type="LRconvBlock3D", use_checkpoint=False).to(device)

    print("Number of parameters, G", numel(generator, only_trainable=True))
    print("Number of parameters, D", numel(discriminator, only_trainable=True))

    patch_size = 32
    x = torch.randn((1, 1, patch_size, patch_size, patch_size)).to(device)

    generator.train()

    start = time.time()
    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    gen_out = generator(x)
    stop = time.time()
    print("Time elapsed:", stop - start)
    #discriminator.train()

    x_hr = torch.randn((1, 1, patch_size * up_factor,
                        patch_size * up_factor,
                        patch_size * up_factor)).cuda()

    loss_func = nn.MSELoss()
    loss = loss_func(gen_out, x_hr)
    loss.backward()

    #disc_out = discriminator(gen_out)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    print(gen_out.shape)
    #print(disc_out.shape)

if __name__ == "__main__":
    test()