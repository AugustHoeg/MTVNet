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
from models.select_network import define_G, define_D
from models.model_base import ModelBase

from loss_functions.loss_functions_simple import VGGLoss, VGGLoss3D, GradientLoss3D, TotalVariationLoss3D, TextureLoss3D, \
    StructureLoss3D, LaplacianLoss3D, compute_generator_loss, compute_critic_loss, compute_gradient_penalty

from performance_metrics.performance_metrics import PSNR_3D, SSIM_3D, NRMSE_3D, compute_performance_metrics

from utils import utils_image
from utils import utils_3D_image

#import torchio.transforms as tiotransforms

#from ..loss_functions import loss_functions
#from loss import GANLoss, PerceptualLoss
#from loss_ssim import SSIMLoss


class ModelWGAN_GP(ModelBase):
    """ Wasserstein GAN model w. Gradient penalty """
    def __init__(self, opt):
        super(ModelWGAN_GP, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.opt_test = self.opt['test']   # test option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.is_train:
            self.netD = define_D(opt)
            self.netD = self.model_to_device(self.netD)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()

        # Lambda gradient penalty
        self.lambda_gp = self.opt['train']['lambda_gp']
        # Counter to track number of updates run on D vs G
        self.D_update_count = 0
        # Counter to track initial iterations on G
        self.G_init_count = 0

        print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))
        print("Number of trainable parameters, D", utils_3D_image.numel(self.netD, only_trainable=True))

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize testing/validation
    # ----------------------------------------
    def init_test(self, experiment_id):
        # Loads model based on the ID specified.
        # If there exists several logs using the same ID, will load latest one.
        self.load(experiment_id, load_D=False)  # load model
        self.netG.eval()  # set eval mode
        self.netD.eval()  # set eval mode
        self.define_metrics()  # define metrics
        self.enable_automatic_mixed_precision()  # enable automatic mixed precision
        self.define_visual_eval()
        # self.log_dict = OrderedDict()          # log


    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.define_wandb_run()                 # use wandb instead of OrderedDict()
        self.load(load_D=True)                             # load model
        self.netG.train()                       # set training mode,for BN
        self.netD.train()                       # set training mode,for BN
        self.define_loss()                      # define loss
        self.define_metrics()                   # define metrics
        self.define_optimizer()                 # define optimizer
        self.enable_automatic_mixed_precision() # enable automatic mixed precision
        self.load_optimizers(load_D=True)                  # load optimizer
        self.define_scheduler()                 # define scheduler
        self.define_visual_eval()
        #self.log_dict = OrderedDict()          # log

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self, experiment_id=None, load_D=False):

        # proposal for loading:
        # Change path to "logs"
        # navigate to appropriate directory using dataset -> wandb -> run ID -> latest

        load_path_G = self.opt['path']['pretrained_netG_experiment_id'] if experiment_id is None else experiment_id
        load_path_D = self.opt['path']['pretrained_netD_experiment_id'] if experiment_id is None else experiment_id
        #load_path_E = True if force_load else self.opt['path']['pretrained_netE']

        # dataset = self.opt['datasets']['name']
        # ID = "*" + self.opt['experiment_id']
        # # Select latest modified log directory
        # wandb_path = os.path.join("logs", dataset, "wandb", ID, "files", "saved_models")

        if load_path_G is not None:
            experiment_id_G = self.opt['path']['pretrained_netG_experiment_id']
            opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + experiment_id_G + "/files/saved_models/*G.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G file found, skipping loading of model...")
            else:
                print('Loading pretrained model for G [{:s}] ...'.format(G_file))
                self.load_network(G_file, self.netG, strict=self.opt_train['G_param_strict'])

        if load_D:
            if load_path_D is not None:
                experiment_id_D = self.opt['path']['pretrained_netD_experiment_id']
                opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + experiment_id_D + "/files/saved_models/*D.h5"))
                opt_files.sort(key=os.path.getmtime, reverse=True)
                try:
                    D_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
                except:
                    print("An exception occurred: No D file found, skipping loading of model...")
                else:
                    print('Loading pretrained model for D [{:s}] ...'.format(D_file))
                    self.load_network(D_file, self.netD, strict=self.opt_train['D_param_strict'])

        if False:
            if load_path_G is not None:
                # Get generator artifact of latest generator run
                artifact_G = self.run.use_artifact(load_path_G)
                #artifact = self.run.use_artifact("Discriminator:latest")
                # Get ID of run that logged the artifact so that the local file can be loaded
                ID = "*" + artifact_G.logged_by().id
                wandb_path = glob.glob(os.path.join("logs", config.DATASET, "wandb", ID, "files", "saved_models"))[-1]
                print('Loading model for G [{:s}] ...'.format(wandb_path))
                self.load_network(wandb_path, self.netG, strict=self.opt_train['G_param_strict'])

            if self.opt['is_train'] and load_path_D is not None:
                artifact_D = self.run.use_artifact(load_path_D)
                ID = "*" + artifact_D.logged_by().id
                wandb_path = glob.glob(os.path.join("logs", config.DATASET, "wandb", ID, "files", "saved_models"))[-1]
                print('Loading model for D [{:s}] ...'.format(wandb_path))
                self.load_network(wandb_path, self.netD, strict=self.opt_train['D_param_strict'])


        # if self.opt_train['E_decay'] > 0:
        #     if load_path_E is not None:
        #         print('Loading model for E [{:s}] ...'.format(load_path_E))
        #         self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'])
        #     else:
        #         print('Copying model for E')
        #         self.update_E(0)
        #     self.netE.eval()


        #print("Test load of model")
        # Get generator artifact of latest generator run
        #artifact = self.run.use_artifact("Generator:latest")
        # download the artifact files to a specific path
        #artifact_dir = artifact.download(root=os.path.join(self.run.dir, "saved_models"))
        # Use the get_path method to download only subset of files:
        #artifact.get_path("24_G.h5")


        #artifact = self.run.use_artifact(config.WANDB_ENTITY+ "/" + config.WANDB_PROJECT + "/" + "Generator:latest")
        #artifact = self.run.use_artifact("Generator:latest")
        #artifact = self.run.use_artifact("Discriminator:latest")
        #artifact.download()


        #artifact = self.run.use_artifact("my-entity/my-project/artifact:alias")
        #api = wandb.Api()
        #artifact = api.artifact("")
        #load_path_G =

        # original code here
        # if load_path_G is not None:
        #     print('Loading model for G [{:s}] ...'.format(load_path_G))
        #     self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'])
        # load_path_E = self.opt['path']['pretrained_netE']
        # if self.opt_train['E_decay'] > 0:
        #     if load_path_E is not None:
        #         print('Loading model for E [{:s}] ...'.format(load_path_E))
        #         self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'])
        #     else:
        #         print('Copying model for E')
        #         self.update_E(0)
        #     self.netE.eval()
        #
        # load_path_D = self.opt['path']['pretrained_netD']
        # if self.opt['is_train'] and load_path_D is not None:
        #     print('Loading model for D [{:s}] ...'.format(load_path_D))
        #     self.load_network(load_path_D, self.netD, strict=self.opt_train['D_param_strict'])

    # ----------------------------------------
    # load optimizerG and optimizerD
    # ----------------------------------------
    def load_optimizers(self, experiment_id=None, load_D=False):

        load_path_optimizerG = self.opt['path']['pretrained_optimizerG_experiment_id'] if experiment_id is None else experiment_id
        load_path_optimizerD = self.opt['path']['pretrained_optimizerD_experiment_id'] if experiment_id is None else experiment_id

        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            experiment_id_opt_G = self.opt['path']['pretrained_optimizerG_experiment_id']
            opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + experiment_id_opt_G + "/files/saved_optimizers/*optimizerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_opt_file = opt_files[0]
            except:
                print("An exception occurred: No G optimizer file found, skipping loading of optimizer...")
            else:
                print('Loading optimizer for G [{:s}] ...'.format(G_opt_file))
                self.load_optimizer(G_opt_file, self.G_optimizer)

        if load_D:
            if load_path_optimizerD is not None and self.opt_train['D_optimizer_reuse']:
                experiment_id_opt_D = self.opt['path']['pretrained_optimizerD_experiment_id']
                opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + experiment_id_opt_D + "/files/saved_optimizers/*optimizerD.h5"))
                opt_files.sort(key=os.path.getmtime, reverse=True)
                try:
                    D_opt_file = opt_files[0]
                except:
                    print("An exception occurred: No D optimizer file found, skipping loading of optimizer...")
                else:
                    print('Loading optimizer for D [{:s}] ...'.format(D_opt_file))
                    self.load_optimizer(D_opt_file, self.D_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        # WandB save directory
        self.model_save_dir = os.path.join(self.run.dir, "saved_models")
        self.opt_save_dir = os.path.join(self.run.dir, "saved_optimizers")

        self.save_network(self.model_save_dir, self.netG, 'G', iter_label)
        self.save_network(self.model_save_dir, self.netD, 'D', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.model_save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.opt_save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['D_optimizer_reuse']:
            self.save_optimizer(self.opt_save_dir, self.D_optimizer, 'optimizerD', iter_label)

        #print("SAVING NETWORK PARAMETERS")
        #torch.save(generator.state_dict(), os.path.join(wandb.run.dir, "saved_models", "generator.h5"))
        filename_G = '{}_{}.h5'.format(iter_label, 'G')
        filename_optG = '{}_{}.h5'.format(iter_label, 'optimizerG')
        filename_D = '{}_{}.h5'.format(iter_label, 'D')
        filename_optD = '{}_{}.h5'.format(iter_label, 'optimizerD')

        # Add saved models/optimizers to wandb artifacts
        print("TODO save model artifacts using wandb")
        #self.model_artifact_G.add_file(os.path.join(self.model_save_dir, filename_G))
        #self.model_artifact_G.add_file(os.path.join(self.opt_save_dir, filename_optG))
        #wandb.run.log_artifact(self.model_artifact_G)

        #self.model_artifact_D.add_file(os.path.join(self.model_save_dir, filename_D))
        #self.model_artifact_D.add_file(os.path.join(self.opt_save_dir, filename_optD))
        #wandb.run.log_artifact(self.model_artifact_D)


    def define_wandb_run(self):

        ######### INITIALIZE WEIGHTS AND BIASES RUN #########

        self.run = wandb.init(
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
                        "D_learning_rate": self.opt['train']['D_optimizer_lr'],
                        "batch_size":  self.opt['datasets']['train']['dataloader_params']['dataloader_batch_size'],
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

        self.model_artifact_D = wandb.Artifact(
            "Discriminator", type=self.opt['netD']['net_type'],
            description=self.opt['netD']['description'],
            metadata=self.opt['netD'])



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
        self.D_train_loss = 0.0
        self.D_valid_loss = 0.0

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

        self.G_init_iters = self.opt_train['G_init_iters'] if self.opt_train['G_init_iters'] else 0

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0

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
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        # KAIR uses G_optim_params and D_optim_params originally to account for parameters that do not require grad.
        self.G_optimizer = Adam(self.netG.parameters(), lr=self.opt_train['G_optimizer_lr'], weight_decay=0, betas=(0.9, 0.999))
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0, betas=(0.9, 0.999))

    # ----------------------------------------
    # define gradient scaler for G and D
    # ----------------------------------------
    def define_gradscaler(self):
        self.gen_scaler = torch.cuda.amp.GradScaler()
        self.dis_scaler = torch.cuda.amp.GradScaler()

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
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                        self.opt_train['D_scheduler_milestones'],
                                                        self.opt_train['D_scheduler_gamma']
                                                        ))

    def define_visual_eval(self):
        if self.opt['datasets']['norm_type'] == 'znormalization':
            unnorm_image = True
            div_max_image = True
        else:
            unnorm_image = False
            div_max_image = False
        self.comparison_tool = utils_3D_image.ImageComparisonTool(patch_size_hr=self.opt['datasets']['patch_size_hr'],
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

        # if self.opt['datasets']['dataset_type'] == 'MasterThesisDataset':
        #     if add_key is not None:
        #         self.L = data['L'][add_key].to(self.device)
        #         if need_H:
        #             self.H = data['H'][add_key].to(self.device)
        #     else:
        #         self.L = data[1].to(self.device)
        #         if need_H:
        #             self.H = data[0].to(self.device)
        #
        # else:
        #     if add_key is not None:
        #         self.L = data['L'][add_key].to(self.device)
        #         if need_H:
        #             self.H = data['H'][add_key].to(self.device)
        #     else:
        #         self.L = data['L'].to(self.device)
        #         if need_H:
        #             self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        if self.opt_train['mixed_precision'] is not None:
            # Evaluate using AMP
            with torch.cuda.amp.autocast(dtype=self.mixed_precision):
                self.E = self.netG(self.L)
        else:  # Standard precision
            self.E = self.netG(self.L)


    def netD_forward(self, input):
        if self.opt_train['mixed_precision'] is not None:
            # Evaluate using AMP
            with torch.cuda.amp.autocast(dtype=self.mixed_precision):
                return self.netD(input.to(self.device))
        else:  # Standard precision
            return self.netD(input.to(self.device))

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):

        if self.G_init_count >= self.G_init_iters:
            # ------------------------------------
            # optimize D for WGAN
            # ------------------------------------

            for p in self.netD.parameters():
                p.requires_grad = True

            with torch.cuda.amp.autocast(dtype=self.mixed_precision):
                # Forward G and forward D on fake image E
                self.netG_forward()
                self.prop_real = self.netD_forward(self.H)  # 1) real data
                self.prop_fake = self.netD_forward(self.E.detach())  # 2) fake data, detach to avoid backprop to G.

                # Compute mixed scores
                batch, channel, height, width, depth = self.H.shape
                alpha = torch.rand((batch, 1, 1, 1, 1)).repeat(1, channel, height, width, depth).to(self.device)
                interpolated_images = self.H * alpha + self.E * (1 - alpha)
                mixed_scores = self.netD_forward(interpolated_images)

                # Compute critic loss
                scaled_gradient_penalty = self.lambda_gp * compute_gradient_penalty(interpolated_images, mixed_scores)
                self.dis_loss = compute_critic_loss(self.prop_real, self.prop_fake, scaled_gradient_penalty)

            self.D_train_loss += self.dis_loss  # Add discriminator training loss to total loss

            self.D_optimizer.zero_grad()  # set parameter gradients to zero
            self.dis_scaler.scale(self.dis_loss).backward()  # backward-pass to compute gradients

            # ------------------------------------
            # clip_grad on D
            # ------------------------------------
            D_clipgrad_max = self.opt_train['D_optimizer_clipgrad']
            if D_clipgrad_max > 0:
                # Unscales the gradients of optimizer's assigned params in-place if AMP is enabled
                self.dis_scaler.unscale_(self.D_optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                print("D gradient norm:", torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=D_clipgrad_max, norm_type=2).item())

            self.dis_scaler.step(self.D_optimizer)  # update weights
            self.dis_scaler.update()

            # Update D count for WGAN
            self.D_update_count += 1


        # If D_update_count >= D_update_ratio or If G_init_iters >= G_init_count, forward G and update G
        if (self.D_update_count >= self.D_update_ratio) or (self.G_init_iters > self.G_init_count):

            # ------------------------------------
            # optimize G
            # ------------------------------------
            for p in self.netD.parameters():
                p.requires_grad = False

            with torch.cuda.amp.autocast(dtype=self.mixed_precision):
                # There are some issues with backprop when reusing the output from G from D update, so forward G again
                self.netG_forward()
                # Run forward if G_init_iters > G_init_count
                if (self.G_init_iters > self.G_init_count):
                    print("G initialization: %d / %d" % (self.G_init_count + 1, self.G_init_iters))
                    self.prop_fake = None  # Disable critic input during initialization of G
                else:
                    self.prop_fake = self.netD_forward(self.E)  # Compute prop_fake again, as the value was discarded by backward
                self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.prop_fake, model=self.opt['model'], device=self.device)

            self.G_train_loss += self.gen_loss  # Add generator training loss to total loss

            self.G_optimizer.zero_grad()  # set parameter gradients to zero
            self.gen_scaler.scale(self.gen_loss).backward()  # backward-pass to compute gradients

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

            # Update G init count
            self.G_init_count += 1

            # Reset D_update_count after G update
            self.D_update_count = 0



    def record_train_log(self, current_step, idx_train):
        # ------------------------------------
        # record log
        # ------------------------------------
        # if self.G_init_iters >= self.G_init_count:
        #     self.run.log({"step": current_step,
        #                   "G_train_loss": self.G_train_loss.item() / idx_train})
        #
        #     # Reset training losses
        #     self.G_train_loss = 0.0

        if (self.D_update_count == self.D_update_ratio) and current_step > self.D_init_iters:

            # Record training losses using wandb
            self.run.log({"step": current_step,
                          "D_train_loss": self.D_train_loss.item() / idx_train,
                          "G_train_loss": self.G_train_loss.item() / idx_train})

            # Reset training losses
            self.G_train_loss = 0.0
            self.D_train_loss = 0.0


        #self.log_dict['l_d_real'] = l_d_real.item()
        #self.log_dict['l_d_fake'] = l_d_fake.item()

        #  TODO update these four lines for use with wandb
        print("TODO update these four lines for use with wandb")
        #self.log_dict['D_real'] = torch.mean(self.prop_real.detach())
        #self.log_dict['D_fake'] = torch.mean(self.prop_fake.detach())

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])



    def record_test_log(self, current_step, idx_test):
        # ------------------------------------
        # record log
        # ------------------------------------
        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:

            for key, value in self.metric_val_dict.items():
                # Get metric name for logging
                metric_name = "Average " + key
                # Record metric value using wandb
                self.run.log({metric_name: self.metric_val_dict[key] / idx_test})
                print(metric_name, self.metric_val_dict[key] / idx_test)
                # Reset performance metric
                self.metric_val_dict[key] = 0.0

            self.run.log({"D_valid_loss": self.D_valid_loss.item() / idx_test,
                          "G_valid_loss": self.G_valid_loss.item() / idx_test})

            # Reset validation losses
            self.G_valid_loss = 0.0
            self.D_valid_loss = 0.0

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

        self.netG.eval()
        self.netD.eval()

        with torch.cuda.amp.autocast(dtype=self.mixed_precision):
            # Forward G and D
            self.netG_forward()
            self.prop_fake = self.netD_forward(self.E)
            self.prop_real = self.netD_forward(self.H)

            # Compute loss for G and D
            #self.dis_loss = compute_discriminator_loss(self.prop_real, self.prop_fake)
            self.dis_loss = -(torch.mean(self.prop_real) - torch.mean(self.prop_fake))
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.prop_fake, model=self.opt['model'], device=self.device)

        # Add discriminator validation loss to total loss
        self.D_valid_loss += self.dis_loss

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        # Compute performance metrics
        rescale_images = True if self.opt['datasets']['norm_type'] == "znormalization" else False
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)



        # with torch.inference_mode():
        #
        #     self.netG.eval()
        #     self.netD.eval()
        #
        #     # Forward G and forward D on fake image E
        #     self.netG_forward()
        #     self.prop_fake = self.netD_forward(self.E)
        #
        #     # Compute generator validation loss with/without AMP
        #     if self.mixed_precision is not None:
        #         with torch.cuda.amp.autocast(dtype=self.mixed_precision):
        #             self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.prop_fake, self.device)
        #     else:
        #         self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.prop_fake, self.device)
        #
        #     # Add generator validation loss to total loss
        #     self.G_valid_loss += self.gen_loss
        #
        #     # Compute discriminator validation loss with/without AMP
        #     self.prop_real = self.netD_forward(self.H)  # 1) real data
        #     #  TODO test clone and detach here
        #     self.prop_fake = self.netD_forward(self.E.detach().clone())  # 2) fake data, detach to avoid backprop to G.
        #
        #     if self.mixed_precision is not None:
        #         with torch.cuda.amp.autocast(dtype=self.mixed_precision):
        #             #  TODO separate dis_loss in real and fake backprop in order to avoid the error in distributed training
        #             self.dis_loss = compute_discriminator_loss(self.prop_real, self.prop_fake)
        #     else:
        #         self.dis_loss = compute_discriminator_loss(self.prop_real, self.prop_fake)
        #
        #     # Add discriminator validation loss to total loss
        #     self.D_valid_loss += self.dis_loss
        #
        #     # Compute performance metrics:
        #     #self.E_img = utils_image.tensor2ufloat(self.E) # returns floats clamped between 0 and 1
        #     #self.H_img = utils_image.tensor2ufloat(self.H) # returns floats clamped between 0 and 1
        #
        #     # In master thesis, SR and HR images are rescaled between 0.0 and 1.0 before computing performance metrics
        #     #rescale = tiotransforms.RescaleIntensity((0.0, 1.0))
        #
        #     rescale_images = True if self.opt['datasets']['norm_type'] == "znormalization" else False
        #     compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)
        #
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
        # out_dict = OrderedDict()
        # out_dict['L'] = self.L.detach()[0].float().cpu()
        # out_dict['E'] = self.E.detach()[0].float().cpu()
        # if need_H:
        #     out_dict['H'] = self.H.detach()[0].float().cpu()
        # return out_dict

        out_dict = OrderedDict()
        out_dict['L'] = self.L[0]
        out_dict['E'] = self.E[0]
        if need_H:
            out_dict['H'] = self.H[0]
        return out_dict

    def log_comparison_image(self, img_dict, current_step):

        grid_image = self.comparison_tool.get_comparison_image(img_dict)
        figure_string = "SR comparison: %s, step %d, %dx upscaling" % (self.opt['model_architecture'], current_step, self.opt['up_factor'])

        if self.opt['run_type'] == "HOME PC":
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(grid_image, vmin=0, vmax=255)
            plt.title(figure_string)
            plt.show()

        wandb.log({"Comparisons training": wandb.Image(grid_image, caption=figure_string, mode="RGB")})  # WandB assumes channel last

    """
    # ----------------------------------------
    # Information of netG, netD and netF
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)
        if self.is_train:
            msg = self.describe_network(self.netD)
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
        if self.is_train:
            msg += self.describe_network(self.netD)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg