import glob
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import lr_scheduler
from torch.optim import Adam

import config
from models.select_network import define_G
from models.model_base import ModelBase

from loss_functions.loss_functions import compute_generator_loss

from performance_metrics.performance_metrics import PSNR_3D, SSIM_3D, NRMSE_3D, compute_performance_metrics

from utils import utils_3D_image
import matplotlib.pyplot as plt

class ModelPlain(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
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

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self, experiment_id=None):
        # navigate to appropriate directory using dataset -> wandb -> run ID -> latest

        load_path_G = self.opt['path']['pretrained_netG_experiment_id'] if experiment_id is None else experiment_id

        if load_path_G is not None:
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

        filename_G = '{}_{}.h5'.format(iter_label, 'G')
        filename_optG = '{}_{}.h5'.format(iter_label, 'optimizerG')


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


    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        self.loss_fn_dict = {
            "MSE": nn.MSELoss(),
            "L1": nn.L1Loss(),
            "BCE_Logistic": nn.BCEWithLogitsLoss(),
            "BCE": nn.BCELoss(),
        }

        self.loss_val_dict = self.opt_train['G_loss_weights']

        # Define losses for G and D
        self.G_train_loss = 0.0
        self.G_valid_loss = 0.0

    # ----------------------------------------
    # define metrics
    # ----------------------------------------
    def define_metrics(self):

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

        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=self.opt_train['G_optimizer_wd'], betas=(0.9, 0.999))

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
        if self.opt['model_architecture'] == 'MTVNet':
            patch_size_hr = int(self.opt['netG']['context_sizes'][-1]*self.opt['up_factor'])
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
            if need_H:
                self.H = data['H'][add_key].to(self.device)
        elif self.opt['datasets']['dataset_type'] == 'MasterThesisDataset':
            self.L = data[1].to(self.device)
            if need_H:
                self.H = data[0].to(self.device)
        else:
            self.L = data['L'].to(self.device)
            if need_H:
                self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        if self.opt_train['mixed_precision'] is not None:
            # Evaluate using AMP
            with torch.cuda.amp.autocast(dtype=self.mixed_precision):
                self.E = self.netG(self.L)  # self.L
        else:  # Standard precision
            self.E = self.netG(self.L)

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

    def record_train_log(self, current_step, idx_train):
        # ------------------------------------
        # record log
        # ------------------------------------

        # Record training losses using wandb
        self.run.log({"step": current_step,
                      "G_train_loss": self.G_train_loss.item() / idx_train})

        # Reset training losses
        self.G_train_loss = 0.0

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def early_stopping(self, current_step, idx_train):
        validation_loss = self.G_valid_loss / idx_train  # calculate average validation loss

        if validation_loss.item() < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.patience_counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.patience_counter += 1
            if (self.patience_counter >= self.patience):
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
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)


    def validation_amp(self):

        with torch.cuda.amp.autocast(dtype=self.mixed_precision):
            # Forward G
            self.netG_forward()

            # Compute loss for G
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        # Compute performance metrics
        rescale_images = True if self.opt['datasets']['norm_type'] == "znormalization" else False
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)

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
