import glob
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import lr_scheduler
from torch.optim import Adam
from torchvision.utils import make_grid

import config
from models.select_network import define_G
from models.model_base import ModelBase

from loss_functions.loss_functions import VGGLoss, VGGLoss3D, GradientLoss3D, TotalVariationLoss3D, TextureLoss3D, \
    StructureLoss3D, LaplacianLoss3D, compute_generator_loss

from performance_metrics.performance_metrics import PSNR_3D, SSIM_3D, NRMSE_3D, compute_performance_metrics

from utils import utils_image
from utils import utils_3D_image

#from ..loss_functions import loss_functions
#from loss import GANLoss, PerceptualLoss
#from loss_ssim import SSIMLoss
import matplotlib.pyplot as plt


def coords_to_image(H, patch_size):
    B, N, C = H.shape
    return H.permute(0, 2, 1).contiguous().view(B, C, patch_size, patch_size, patch_size)


class ModelImplicit(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt):
        super(ModelImplicit, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.opt_test = self.opt['test']   # test option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

        print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

        # ------------------------------------
        # define early stopping parameters
        # ------------------------------------
        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def set_eval_mode(self):
        self.netG.eval()

    def set_train_mode(self):
        self.netG.train()

    def init_test(self, experiment_id):
        # Loads model based on the ID specified.
        # If there exists several logs using the same ID, will load latest one.
        self.load(experiment_id)  # load model
        self.netG.eval()  # set eval mode
        self.define_metrics()  # define metrics
        self.enable_automatic_mixed_precision()  # enable automatic mixed precision
        self.define_visual_eval()
        # self.log_dict = OrderedDict()          # log

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.define_wandb_run()                 # use wandb instead of OrderedDict()
        self.load()                             # load model
        self.netG.train()                       # set training mode,for BN
        self.define_loss()                      # define loss
        self.define_metrics()                   # define metrics
        self.define_optimizer()                 # define optimizer
        self.enable_automatic_mixed_precision() # enable automatic mixed precision
        self.load_optimizers()                  # load optimizer
        self.define_scheduler()                 # define scheduler
        self.define_visual_eval()
        #self.log_dict = OrderedDict()          # log

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self, experiment_id=None):
        # proposal for loading:
        # Change path to "logs"
        # navigate to appropriate directory using dataset -> wandb -> run ID -> latest

        load_path_G = self.opt['path']['pretrained_netG_experiment_id'] if experiment_id is None else experiment_id
        # load_path_E = True if force_load else self.opt['path']['pretrained_netE']

        # dataset = self.opt['datasets']['name']
        # ID = "*" + self.opt['experiment_id']
        # # Select latest modified log directory
        # wandb_path = os.path.join("logs", dataset, "wandb", ID, "files", "saved_models")

        if load_path_G is not None:
            #experiment_id_G = self.opt['path']['pretrained_netG_experiment_id']
            experiment_id_G = load_path_G
            opt_files = glob.glob(os.path.join(
                config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + experiment_id_G + "/files/saved_models/*G.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G file found, skipping loading of model...")
            else:
                print('Loading pretrained model for G [{:s}] ...'.format(G_file))
                self.load_network(G_file, self.netG, strict=self.opt_train['G_param_strict'])

    # ----------------------------------------
    # load optimizerG and optimizerD
    # ----------------------------------------
    def load_optimizers(self, experiment_id=None):

        load_path_optimizerG = self.opt['path']['pretrained_optimizerG_experiment_id'] if experiment_id is None else experiment_id

        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            experiment_id_opt_G = self.opt['path']['pretrained_optimizerG_experiment_id']
            opt_files = glob.glob(os.path.join(
                config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + experiment_id_opt_G + "/files/saved_optimizers/*optimizerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_opt_file = opt_files[0]
            except:
                print("An exception occurred: No G optimizer file found, skipping loading of optimizer...")
            else:
                print('Loading optimizer for G [{:s}] ...'.format(G_opt_file))
                self.load_optimizer(G_opt_file, self.G_optimizer)


    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        # WandB save directory
        self.model_save_dir = os.path.join(self.run.dir, "saved_models")
        self.opt_save_dir = os.path.join(self.run.dir, "saved_optimizers")

        self.save_network(self.model_save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.model_save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.opt_save_dir, self.G_optimizer, 'optimizerG', iter_label)


        #print("SAVING NETWORK PARAMETERS")
        #torch.save(generator.state_dict(), os.path.join(wandb.run.dir, "saved_models", "generator.h5"))
        filename_G = '{}_{}.h5'.format(iter_label, 'G')
        filename_optG = '{}_{}.h5'.format(iter_label, 'optimizerG')

        # Add saved models/optimizers to wandb artifacts
        print("TODO save model artifacts using wandb")
        #self.model_artifact_G.add_file(os.path.join(self.model_save_dir, filename_G))
        #self.model_artifact_G.add_file(os.path.join(self.opt_save_dir, filename_optG))
        #wandb.run.log_artifact(self.model_artifact_G)


    def define_wandb_run(self):

        ######### INITIALIZE WEIGHTS AND BIASES RUN #########

        self.run = wandb.init(
            mode=self.opt["wandb_mode"],
            # set the wandb project where this run will be logged
            entity=self.opt['wandb_entity'],
            project=self.opt['wandb_project'],
            name=self.opt['run_name'],
            id=self.opt['experiment_id'],
            notes=self.opt['note'],
            dir="logs/" + self.opt['datasets']['name'],

            # track hyperparameters and run metadata
            config={

                "epochs": self.opt['train']['epochs'],
                "G_learning_rate": self.opt['train']['G_optimizer_lr'],
                "batch_size": self.opt['datasets']['train']['dataloader_params']['dataloader_batch_size'],
                "patches_per_batch": self.opt['datasets']['train']['dataset_params']['patches_per_batch'],
                "dataset": self.opt['datasets']['name'],
                "up_factor": self.opt['up_factor'],
                "architecture": self.opt['model_architecture'],

            })

        ######### CREATE DIRECTORY FOR SAVED MODELS, OPTIMIZERS, ECT. #########
        os.mkdir(os.path.join(wandb.run.dir, "saved_models"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_optimizers"))

        # Create model artifacts for logging of files
        # Look at this link for how to construct an artifact with a more neat file structure:
        # https://docs.wandb.ai/guides/artifacts/construct-an-artifact
        self.wandb_config = wandb.config

        self.model_artifact_G = wandb.Artifact(
            "Generator", type=self.opt['netG']['net_type'],
            description=self.opt['netG']['description'],
            metadata=self.opt['netG'])


    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        self.loss_fn_dict = {
            "MSE": nn.MSELoss(),
            "L1": nn.L1Loss(),
            "BCE_Logistic": nn.BCEWithLogitsLoss(),
            "BCE": nn.BCELoss(),
            "VGG": VGGLoss(layer_idx=36, device=self.device),
            "VGG3D": VGGLoss3D(num_parts=2*self.opt['up_factor'], layer_idx=35, loss_func=nn.MSELoss(), device=self.device),
            "GRAD": GradientLoss3D(kernel='diff', order=1, loss_func=nn.L1Loss(), sigma=None),  # sigma = 0.8,
            "LAPLACE": LaplacianLoss3D(sigma=1.0, padding='valid', loss_func=nn.L1Loss()),
            "TV3D": TotalVariationLoss3D(mode="L2"),  # or mode = "sum_of_squares", "L2",
            "TEXTURE3D": TextureLoss3D(layer_idx=35),
            "STRUCTURE_TENSOR": StructureLoss3D(sigma=0.5, rho=0.5)
        }

        self.loss_val_dict = self.opt_train['G_loss_weights']

        # Define losses for G and D
        self.G_train_loss = 0.0
        self.G_valid_loss = 0.0

        # ------------------------------------
        # 1) G_loss
        # ------------------------------------



        # if self.opt_train['G_lossfn_weight'] > 0:
        #     G_lossfn_type = self.opt_train['G_lossfn_type']
        #     if G_lossfn_type == 'l1':
        #         self.G_lossfn = nn.L1Loss().to(self.device)
        #     elif G_lossfn_type == 'l2':
        #         self.G_lossfn = nn.MSELoss().to(self.device)
        #     elif G_lossfn_type == 'l2sum':
        #         self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        #     else:
        #         raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        #     self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        # else:
        #     print('Do not use pixel loss.')
        #     self.G_lossfn = None

        # ------------------------------------
        # 2) F_loss
        # ------------------------------------
        #if self.opt_train['F_lossfn_weight'] > 0:
        #    F_feature_layer = self.opt_train['F_feature_layer']
        #    F_weights = self.opt_train['F_weights']
        #    F_lossfn_type = self.opt_train['F_lossfn_type']
        #    F_use_input_norm = self.opt_train['F_use_input_norm']
        #    F_use_range_norm = self.opt_train['F_use_range_norm']
        #    if self.opt['dist']:
        #        self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm).to(self.device)
        #    else:
        #        self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm)
        #        self.F_lossfn.vgg = self.model_to_device(self.F_lossfn.vgg)
        #        self.F_lossfn.lossfn = self.F_lossfn.lossfn.to(self.device)
        #    self.F_lossfn_weight = self.opt_train['F_lossfn_weight']
        #else:
        #    print('Do not use feature loss.')
        #    self.F_lossfn = None

        # ------------------------------------
        # 3) D_loss
        # ------------------------------------

        #self.D_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        #self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

    # ----------------------------------------
    # define metrics
    # ----------------------------------------
    def define_metrics(self):
        #self.metric_fn_dict = {
        #    "psnr": performance_metrics.PSNR_3D(),
        #    "ssim": performance_metrics.SSIM_3D(),
        #    "nrmse": performance_metrics.NRMSE_3D(),
        #}

        # Define losses for G and D
        #self.psnr = 0.0
        #self.ssim = 0.0
        #self.nrmse = 0.0

        # TODO add functionality for 2D metrics (Here 3D images are assumed)
        self.metric_fn_dict = {}
        self.metric_val_dict = {}
        if "psnr" in self.opt_test['performance_metrics']:
            self.metric_val_dict["psnr"] = 0.0
            self.metric_fn_dict["psnr"] = PSNR_3D()
        if "ssim" in self.opt_test['performance_metrics']:
            self.metric_val_dict["ssim"] = 0.0
            self.metric_fn_dict["ssim"] = SSIM_3D()
        if "nrmse" in self.opt_test['performance_metrics']:
            self.metric_val_dict["nrmse"] = 0.0
            self.metric_fn_dict["nrmse"] = NRMSE_3D()



    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):

        # Testing gradient accumulation
        self.G_accum_count = 0
        self.num_accum_steps_G = self.opt_train['num_accum_steps_G']

        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0, betas=(0.9, 0.999))

    # ----------------------------------------
    # define gradient scaler for G and D
    # ----------------------------------------
    def define_gradscaler(self):
        self.gen_scaler = torch.cuda.amp.GradScaler()

    # ----------------------------------------
    # Set working precision for use with PyTorch AMP
    # ----------------------------------------
    def enable_automatic_mixed_precision(self):
        if self.opt_train['mixed_precision'] == "FP16":
            self.mixed_precision = torch.float16
            self.define_gradscaler()

        elif self.opt_train['mixed_precision'] == "FP32":
            self.mixed_precision = torch.float32
            self.define_gradscaler()

        elif self.opt_train['mixed_precision'] == "BF16":
            self.mixed_precision = torch.bfloat16
            self.define_gradscaler()

        else:
            self.mixed_precision = None

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    def define_visual_eval(self):
        if self.opt['datasets']['norm_type'] == 'znormalization':
            unnorm_image = True
            div_max_image = True
        else:
            unnorm_image = False
            div_max_image = False
        if self.opt['model_architecture'] == 'AugustNet':
            # TODO: Fix this for level_ratios other than 2
            patch_size_hr = int(self.opt['datasets']['patch_size']*((2/2**self.opt['netG']['num_levels'])*self.opt['up_factor']))
        else:
            patch_size_hr = self.opt['datasets']['patch_size_hr']
        self.comparison_tool = utils_3D_image.ImageComparisonTool(patch_size_hr=patch_size_hr,
                                                                  upscaling_methods=["tio_nearest", "tio_linear"],
                                                                  unnorm=unnorm_image,
                                                                  div_max=div_max_image,
                                                                  out_dtype=np.uint8)

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True, add_key=None):
        if add_key is not None:
            self.L = data['L'][add_key].to(self.device)
            self.H_xyz = data['H_xyz'][add_key].to(self.device)
            self.H = data['H'][add_key].to(self.device)
        else:
            self.L = data['L'].to(self.device)
            self.H_xyz = data['H_xyz'].to(self.device)
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        if self.opt_train['mixed_precision'] is not None:
            # Evaluate using AMP
            with torch.cuda.amp.autocast(dtype=self.mixed_precision):
                self.E = self.netG(self.L, self.H_xyz)  # self.L
        else:  # Standard precision
            self.E = self.netG(self.L, self.H_xyz)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters_amp(self, current_step, update=False):

        # ------------------------------------
        # optimize G
        # ------------------------------------

        with torch.cuda.amp.autocast(dtype=self.mixed_precision):
            # Forward G
            self.netG_forward()
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict,None, self.device)
            self.gen_loss = self.gen_loss / self.num_accum_steps_G  # Scale loss by number of accumulation steps

        self.G_train_loss += self.gen_loss  # Add generator training loss to total loss

        #self.G_optimizer.zero_grad()  # set parameter gradients to zero
        self.gen_scaler.scale(self.gen_loss).backward()  # backward-pass to compute gradients

        if ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update:  # Gradient acculumation
            # ------------------------------------
            # clip_grad on G
            # ------------------------------------
            G_clipgrad_max = self.opt_train['G_optimizer_clipgrad']
            if G_clipgrad_max > 0:
                # Unscales the gradients of optimizer's assigned params in-place if AMP is enabled
                self.gen_scaler.unscale_(self.G_optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                print("G gradient norm:", torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=G_clipgrad_max, norm_type=2).item())

            self.gen_scaler.step(self.G_optimizer)  # update weights
            self.gen_scaler.update()
            self.G_optimizer.zero_grad()  # set parameter gradients to zero

            # Reset gradient accumulation count
            self.G_accum_count = 0

        self.G_accum_count += 1


        # self.G_optimizer.zero_grad()
        #
        # # Forward G on fake image E with/without AMP
        # self.netG_forward()
        #
        # # Compute generator loss with/without AMP
        # if self.mixed_precision is not None:
        #     with torch.cuda.amp.autocast(dtype=self.mixed_precision):
        #         self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #         self.gen_scaler.scale(self.gen_loss).backward()
        # else:
        #     self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #     self.gen_loss.backward()  # Standard precision backward
        #
        # # Add generator training loss to total loss
        # self.G_train_loss += self.gen_loss
        #
        # # ------------------------------------
        # # clip_grad
        # # ------------------------------------
        # G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        # if G_optimizer_clipgrad > 0:
        #     # Unscales the gradients of optimizer's assigned params in-place if AMP is enabled
        #     if self.mixed_precision is not None:
        #         self.gen_scaler.unscale_(self.G_optimizer)
        #     # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        #     print("G gradient norm:", torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=G_optimizer_clipgrad, norm_type=2).item())
        #
        # # ------------------------------------
        # # update parameters for G
        # # ------------------------------------
        # if self.mixed_precision is not None:
        #     self.gen_scaler.step(self.G_optimizer)
        #     self.gen_scaler.update()
        # else:
        #     self.G_optimizer.step()
        #
        # # ------------------------------------
        # # TODO Regularizer as in SuperFormer
        # # ------------------------------------

    def optimize_parameters(self, current_step, update=False):

        # ------------------------------------
        # optimize G
        # ------------------------------------

        # Forward G
        self.netG_forward()
        self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict,None, self.device)
        self.gen_loss = self.gen_loss / self.num_accum_steps_G  # Scale loss by number of accumulation steps

        self.G_train_loss += self.gen_loss  # Add generator training loss to total loss
        self.gen_loss.backward()  # backward-pass to compute gradients

        if ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update:  # Gradient acculumation
            # ------------------------------------
            # clip_grad on G
            # ------------------------------------
            G_clipgrad_max = self.opt_train['G_optimizer_clipgrad']
            if G_clipgrad_max > 0:
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                print("G gradient norm:", torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=G_clipgrad_max, norm_type=2).item())

            self.G_optimizer.step()  # update weights
            self.G_optimizer.zero_grad()  # set parameter gradients to zero

            # Reset gradient accumulation count
            self.G_accum_count = 0

        self.G_accum_count += 1
        # # ------------------------------------
        # # TODO Regularizer as in SuperFormer
        # # ------------------------------------


    def record_train_log(self, current_step, idx_train):
        # ------------------------------------
        # record log
        # ------------------------------------

        # Record training losses using wandb
        self.run.log({"step": current_step,
                      "G_train_loss": self.G_train_loss.item() / idx_train})

        # Reset training losses
        self.G_train_loss = 0.0


        #self.log_dict['l_d_real'] = l_d_real.item()
        #self.log_dict['l_d_fake'] = l_d_fake.item()

        #  TODO update these four lines for use with wandb
        print("TODO update these four lines for use with wandb")
        #self.log_dict['D_real'] = torch.mean(self.prop_real.detach())
        #self.log_dict['D_fake'] = torch.mean(self.prop_fake.detach())

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def early_stopping(self, current_step, idx_train):
        validation_loss = self.G_valid_loss / idx_train  # calculate average validation loss

        if validation_loss.item() < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.patience_counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True

    def record_test_log(self, current_step, idx_test):
        # ------------------------------------
        # record log
        # ------------------------------------

        for key, value in self.metric_val_dict.items():
            # Get metric name for logging
            metric_name = "Average " + key
            # Record metric value using wandb
            self.run.log({metric_name: self.metric_val_dict[key] / idx_test})
            print(metric_name, self.metric_val_dict[key] / idx_test)
            # Reset performance metric
            self.metric_val_dict[key] = 0.0

        self.run.log({"G_valid_loss": self.G_valid_loss.item() / idx_test})

        # Reset validation losses
        self.G_valid_loss = 0.0

    # ----------------------------------------
    # test and inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        #with torch.no_grad():
        with torch.inference_mode():
            self.netG_forward()
        self.netG.train()

    def validation(self):

        # Forward G
        with torch.inference_mode():
            self.netG_forward()

        # Compute loss for G
        self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        # Compute performance metrics
        rescale_images = True if self.opt['datasets']['norm_type'] == "znormalization" else False

        # Reshape intensity predictions to images
        self.H = coords_to_image(self.H, patch_size=self.opt['datasets']['patch_size_hr'])
        self.E = coords_to_image(self.E, patch_size=self.opt['datasets']['patch_size_hr'])

        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def validation_amp(self):

        # Forward G
        with torch.cuda.amp.autocast(dtype=self.mixed_precision):
            self.netG_forward()

            # Compute loss for G
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        # Compute performance metrics
        rescale_images = True if self.opt['datasets']['norm_type'] == "znormalization" else False

        # Reshape intensity predictions to images
        self.H = coords_to_image(self.H, patch_size=self.opt['datasets']['patch_size_hr'])
        self.E = coords_to_image(self.E, patch_size=self.opt['datasets']['patch_size_hr'])

        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)

        # self.netG.eval()
        #
        # with torch.inference_mode():
        #
        #     # Forward G on fake image E with/without AMP
        #     self.netG_forward()
        #
        #     # Compute generator validation loss with/without AMP
        #     if self.mixed_precision is not None:
        #         with torch.cuda.amp.autocast(dtype=self.mixed_precision):
        #             self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #     else:
        #         self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #
        #     # Add generator validation loss to total loss
        #     self.G_valid_loss += self.gen_loss
        #
        #     # Compute performance metrics:
        #     #self.E_img = utils_image.tensor2ufloat(self.E) # returns floats clamped between 0 and 1
        #     #self.H_img = utils_image.tensor2ufloat(self.H) # returns floats clamped between 0 and 1
        #     compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict)
        #     #for key, value in self.metric_func_dict.items():
        #     #    self.metric_val_dict[key] += self.metric_func_dict[key](self.E_img, self.H_img, border=config.OPT['scale'])
        #
        #         #self.psnr += util.calculate_psnr(self.E_img, self.H_img, border=border)
        #         #self.nrmse += util.calculate_nrmse(self.H_img, self.E_img, border=border)
        #         #self.ssim += util.calculate_ssim_3d(self.H_img, self.E_img, border=border)
        #
        #     #save_img_path = os.path.join(img_dir, '{:s}_{:d}.nii.gz'.format(img_name, current_step))
        #     #output_nib = nib.Nifti1Image(E_img, np.eye(4))




    def full_reconstruction(self):
        pass
        # # Code for full sample reconstruction from SuperFormer
        # HR = self.H
        # #HR = test_data["H"]
        # H, W, D = HR.shape[2:]
        # patches = (HR.shape[2] // opt["datasets"]["test"]["train_size"]) * (
        #             HR.shape[3] // opt["datasets"]["test"]["train_size"]) * (
        #                       HR.shape[4] // opt["datasets"]["test"]["train_size"])
        # model.netG.eval()
        # output = torch.zeros_like(test_data['H'])
        # i = 0
        # for h in range(H // train_size):
        #     for w in range(W // train_size):
        #         for d in range(D // train_size):
        #             patch_L = test_data['L'][:, :, h * train_size:h * train_size + train_size,
        #                       w * train_size:w * train_size + train_size,
        #                       d * train_size:d * train_size + train_size]
        #             model.feed_data({'L': patch_L}, need_H=False)
        #             model.test()
        #             output[:, :, h * train_size:h * train_size + train_size,
        #             w * train_size:w * train_size + train_size,
        #             d * train_size:d * train_size + train_size] = model.E
        #             print(i)
        #             i += 1
        #
        # self.E_img = util.tensor2uint(output)
        # self.H_img = util.tensor2uint(HR)


    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H images
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    def log_comparison_image(self, img_dict, current_step):

        grid_image = self.comparison_tool.get_comparison_image(img_dict)
        figure_string = "SR comparison: %s, step %d, %dx upscaling" % (self.opt['model_architecture'], current_step, self.opt['up_factor'])

        if self.opt['run_type'] == "HOME PC":
            height, width = grid_image.shape[:2]
            plt.figure(figsize=(4 * width / 100, 4 * height / 100), dpi=100)
            plt.imshow(grid_image, vmin=0, vmax=255)
            plt.title(figure_string)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            #plt.figure()
            #plt.imshow(grid_image, vmin=0, vmax=255)
            #plt.title(figure_string)
            #plt.show()

        wandb.log({"Comparisons training": wandb.Image(grid_image, caption=figure_string, mode="RGB")})  # WandB assumes channel last

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg