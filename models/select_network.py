import functools
import torch
from torch.nn import init

"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # mDCSRN Generator
    # ----------------------------------------
    if net_type == ("mDCSRN" or "mDCSRN-GAN"):  # mDCSRN
        from models.mDCSRN_GAN import MultiLevelDenseNet as net
        netG = net(up_factor=opt['up_factor'],
                   in_c=opt_net['in_channels'],
                   k_factor=opt_net['k_factor'],
                   k_size=opt_net['k_size'],
                   num_dense_blocks=opt_net['num_dense_blocks'],
                   num_dense_units=opt_net['num_dense_units'],
                   upsample_method=opt_net['upsample_method'],
                   use_checkpoint=opt_net['use_checkpoint'])

    # ----------------------------------------
    # RRDBNet3D / ESRGAN3D Generator
    # ----------------------------------------
    elif net_type == ("RRDBNet3D" or "ESRGAN3D"):
        from models.RRDBNet3D_official import RRDBNet as net
        netG = net(up_factor=opt['up_factor'],
                   in_nc=opt_net['in_channels'],
                   out_nc=1,
                   nf=opt_net['num_channels'],
                   nb=opt_net['num_blocks'],
                   gc=32)

    # ----------------------------------------
    # EDDSR
    # ----------------------------------------
    elif net_type == ("EDDSR"):
        from models.EDDSR import EDDSR_xs as net
        netG = net(up_factor=opt['up_factor'])


    # ----------------------------------------
    # MFER
    # ----------------------------------------
    elif net_type == ("MFER"):
        from models.MFER_official import MFER_xs as net
        netG = net(up_factor=opt['up_factor'])

    # ----------------------------------------
    # 3D Med SwinIR - SuperFormer
    # ----------------------------------------
    elif net_type == 'SuperFormer':
        from models.SuperFormer import SuperFormer as net

        netG = net(img_size=opt['datasets']['patch_size'],
                   patch_size=opt_net['patch_size'],
                   in_chans=opt['datasets']['n_channels'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   num_heads=opt_net['num_heads'],
                   window_size=opt_net['window_size'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   qkv_bias=True,
                   qk_scale=None,
                   drop_rate=0.,
                   attn_drop_rate=0.,
                   drop_path_rate=0.1,
                   norm_layer=torch.nn.LayerNorm,
                   ape=opt_net['ape'],
                   rpb=opt_net['rpb'],
                   patch_norm=True,
                   use_checkpoint=opt_net['use_checkpoint'],
                   upscale=opt['up_factor'],
                   img_range=1.,
                   upsampler=opt_net['upsampler'],
                   resi_connection='1conv',
                   output_type=opt_net['output_type'],
                   num_feat=opt_net['num_feat'],
                   init_type="default")

    # ----------------------------------------
    # MTVNet
    # ----------------------------------------

    elif net_type == 'MTVNet':

        from models.MTVNet_arch import MTVNet as net

        H = opt['datasets']['patch_size']
        W = opt['datasets']['patch_size']
        D = opt['datasets']['patch_size']
        input_size = (H, W, D)

        netG = net(input_size=input_size,
                  up_factor=opt['up_factor'],
                  num_levels=opt_net['num_levels'],
                  context_sizes=opt_net['context_sizes'],
                  num_blks=opt_net['num_blks'],
                  blk_layers=opt_net['blk_layers'],
                  in_chans=opt_net['in_channels'],
                  shallow_feats=opt_net['shallow_feats'],
                  pre_up_feats=opt_net['pre_up_feats'],
                  ct_embed_dims=opt_net['ct_embed_dims'],  # [512, 256, 128]
                  embed_dims=opt_net['embed_dims'],  # [512, 256, 128]
                  ct_size=opt_net['ct_size'],
                  ct_pool_method=opt_net['ct_pool_method'],
                  patch_sizes=opt_net['patch_sizes'],
                  num_heads=opt_net['num_heads'],
                  attn_window_sizes=opt_net['attn_window_sizes'],
                  enable_shift=opt_net['enable_shift'],
                  mlp_ratio=4.,
                  qkv_bias=True,
                  drop=0.,
                  attn_drop=0.,
                  drop_path=opt_net['drop_path'],
                  token_upsample_method=opt_net["token_upsample_method"],
                  upsample_method=opt_net["upsample_method"],
                  use_checkpoint=opt_net["use_checkpoint"],
                  layer_type=opt_net["layer_type"],  # fastervit_without_ct, swin, fastervit
                  enable_ape_ct=opt_net["enable_ape_ct"],
                  enable_ape_x=opt_net["enable_ape_x"],
                  enable_ct_rpb=opt_net["enable_ct_rpb"],
                  enable_conv_skip=opt_net["enable_conv_skip"],
                  patch_pe_method=opt_net["patch_pe_method"],)

    # ----------------------------------------
    # ArSSR
    # ----------------------------------------
    elif net_type == ("ArSSR"):
        from models.ArSSR import ArSSR as net
        netG = net(encoder_name=opt_net["encoder_name"],
                  feature_dim=opt_net["feature_dim"],
                  decoder_depth=opt_net["decoder_depth"],
                  decoder_width=opt_net["decoder_width"])

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # Here we search for layers called Conv3d, Conv2d or Linear.
        # We don't include ConvTranspose3d because we might want to use ICNR, which would otherwise be overwritten.
        if classname.find('Conv3d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        #elif classname.find('BatchNorm2d') != -1:
        elif classname.find('BatchNorm3d') != -1 or classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
