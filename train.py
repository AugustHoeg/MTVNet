
import os
import time

import monai.transforms
import torch
from monai.data import SmartCacheDataset

import config
from utils.load_options import load_json, parse_arguments, save_json
from utils.utils_3D_image import crop_context, crop_center

from torch.profiler import profile, ProfilerActivity

import lightning as L

def train_lightning(opt, train_dataset, test_dataset, iterations):

    # Define datasets
    if opt['dist']:
        from data.dataset import DDP_CacheDataset
        train_dataset = DDP_CacheDataset(train_dataset)
        test_dataset = DDP_CacheDataset(test_dataset)

    dataloader_params_train = opt['datasets']['train']['dataloader_params']
    dataloader_params_test = opt['datasets']['test']['dataloader_params']

    train_loader = monai.data.DataLoader(train_dataset,
                                         batch_size=dataloader_params_train['dataloader_batch_size'],
                                         shuffle=dataloader_params_train['dataloader_shuffle'],
                                         num_workers=dataloader_params_train['num_load_workers'],
                                         persistent_workers=dataloader_params_train['persist_workers'],
                                         pin_memory=dataloader_params_train['pin_memory'])

    test_loader = monai.data.DataLoader(test_dataset,
                                        batch_size=dataloader_params_test['dataloader_batch_size'],
                                        shuffle=dataloader_params_test['dataloader_shuffle'],
                                        num_workers=dataloader_params_test['num_load_workers'],
                                        persistent_workers=dataloader_params_test['persist_workers'],
                                        pin_memory=dataloader_params_test['pin_memory'])

    # Define Lightning model

    # Set strategy
    strategy = "auto"
    if torch.cuda.device_count() > 1:
        strategy = 'ddp'

    # Trainer configuration
    trainer = L.Trainer(
        max_steps=iterations,
        precision="bf16-mixed",  # Mixed precision with bfloat16
        gradient_clip_val=opt['train']['G_optimizer_clipgrad'],  # Gradient clipping
        accumulate_grad_batches=opt['train']['num_accum_steps_G'],  # Gradient accumulation
        strategy=strategy,  # DDPStrategy(find_unused_parameters=False),  # DDP Strategy
        devices=1 if not torch.cuda.is_available() else torch.cuda.device_count(),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        benchmark=True,  # set torch.backends.cudnn.benchmark to True
        logger=False,
        use_distributed_sampler=False
    )

    # Train
    trainer.fit(model, train_loader, test_loader)



def train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True):
    """
    Train function for universal SR model.
    :param model:
    :param opt_dataset:
    :param iterations:
    :param train_loader:
    :param test_loader:
    :param print_status:
    :return:
    """

    current_step = 0

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 matrix multiplications on Ampere GPUs and later
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 operations on Ampere GPUs and later

    n_train_batches = len(train_loader)  # number of batches per epoch in the training dataset
    n_test_batches = len(test_loader)   # number of batches per epoch in the test dataset

    checkpoint_print = opt['train']['checkpoint_print']
    checkpoint_save = opt['train']['checkpoint_save']
    checkpoint_test = opt['train']['checkpoint_test']
    if checkpoint_print == 0: checkpoint_print = n_train_batches
    if checkpoint_save == 0: checkpoint_save = n_train_batches
    if checkpoint_test == 0: checkpoint_test = n_train_batches

    start_time = time.time()
    save_time = opt['save_time']

    while current_step < iterations:
        idx_train = 0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for batch_idx, train_batch in enumerate(train_loader):

            current_step += 1
            idx_train += 1

            # -------------------------------
            # 1) load batches of HR and LR images onto GPU and feed to model
            # -------------------------------
            #if opt['model_architecture'] == "MTVNet" and not opt['datasets']['enable_femur_padding']:
            #    train_batch['H'] = crop_context(train_batch['H'], L=model.opt['netG']['num_levels'], level_ratio=model.opt['netG']['level_ratio'])

            model.feed_data(train_batch)

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            update = True if (current_step % len(train_loader) == 0) else False  # force parameter update on last batch
            if opt['train']['mixed_precision'] is not None:
                model.optimize_parameters_amp(current_step, update=update)
            else:
                model.optimize_parameters(current_step, update=update)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            if model.update:
                model.update_learning_rate()  # removed current step here and moved line to after optimizer.step()

            # -------------------------------
            # 4) print training information
            # -------------------------------
            if current_step % checkpoint_print == 0 and opt['rank'] == 0:
                print("Iteration %d / %d" % (current_step, iterations))

            # -------------------------------
            # 5) record training log at the end of every epoch
            # -------------------------------
            if current_step % len(train_loader) == 0 and opt['rank'] == 0:
                model.record_train_log(current_step, idx_train)

            # -------------------------------
            # 6) save model
            # -------------------------------
            elapsed_time = (time.time() - start_time) / 3600  # Elapsed time in hours
            if ((current_step % checkpoint_save == 0) or (elapsed_time > save_time)) and opt['rank'] == 0:
                save_time = torch.inf  # disable time-based model saving
                print("SAVING NETWORK PARAMETERS AT STEP %d / %d" % (current_step, iterations))
                # logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 8) testing / validation
            # -------------------------------
            if current_step % checkpoint_test == 0 and opt['rank'] == 0:
                # Set model(s) to evaluation model
                model.set_eval_mode()

                torch.cuda.empty_cache()
                idx_test = 0
                with torch.inference_mode():
                    while idx_test < validation_iterations:
                        for batch_idx, test_batch in enumerate(test_loader):
                            idx_test += 1

                            if idx_test % 100 == 0 and opt['rank'] == 0:
                                print("Validation iteration %d / %d" % (idx_test, validation_iterations))

                            # -------------------------------
                            # 9) load batches of HR and LR images onto GPU and feed to model
                            # -------------------------------

                            #if opt['model_architecture'] == "MTVNet" and not opt['datasets']['enable_femur_padding']:
                            #    test_batch['H'] = crop_context(test_batch['H'], L=model.opt['netG']['num_levels'], level_ratio=model.opt['netG']['level_ratio'])

                            model.feed_data(test_batch)

                            # -------------------------------
                            # 10) Test model using inference mode
                            # -------------------------------
                            if opt['train']['mixed_precision'] is not None:
                                model.validation_amp()
                            else:
                                model.validation()

                    # -------------------------------
                    # 12) Record early stopping
                    # -------------------------------
                    model.early_stopping(current_step, idx_test)

                    # -------------------------------
                    # 11) calculate and record validation log
                    # -------------------------------
                    model.record_test_log(current_step, idx_test)

                # -------------------------------
                # 13) Save visual comparison
                # -------------------------------
                print("Saving comparison: test image")
                if opt['model_architecture'] == "MTVNet":
                    model.L = crop_center(model.L, center_size=model.opt['netG']['context_sizes'][-1])
                    #model.L = crop_context(model.L, L=model.opt['netG']['num_levels'], level_ratio=model.opt['netG']['level_ratio'])

                visuals = model.current_visuals()
                model.log_comparison_image(visuals, current_step)

                # Update test_loader with new samples if SmartCacheDataset
                if type(test_loader.dataset) == SmartCacheDataset:
                    test_dataset.update_cache()

                # Set model(s) to train model
                model.set_train_mode()

        # Update train_loader with new samples if SmartCacheDataset
        if type(train_loader.dataset) == SmartCacheDataset:
            train_dataset.update_cache()

        # -------------------------------
        # 7) Print maximum reserved GPU memory
        # -------------------------------

        if print_status and opt['rank'] == 0:
            print(f"Iteration: {current_step}/{iterations}")

            max_memory_reserved = torch.cuda.max_memory_reserved()
            print("Max memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, opt['total_gpu_mem']))

        # Trigger early stopping
        if model.early_stop:
            print("EARLY STOPPING")
            break

    # Shutdown SmartCacheDatasets
    if type(train_loader.dataset) == SmartCacheDataset:
        train_dataset.shutdown()
        test_dataset.shutdown()

    if opt['rank'] == 0:

        # Save final model
        if opt['save_model']:
            print("SAVING NETWORK PARAMETERS AFTER TRAINING")
            model.save(current_step)

        # Close WandB run
        model.run.finish()

    print("Training finished")

    return 0


if __name__ == "__main__":

    print("RUNNING TRAIN")

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))

    # Returns None if no arguments parsed, as when run in IDE
    args = parse_arguments()

    # Define experiment parameters
    options_file = args.options_file
    print("options_file", options_file)

    # Load default experiment option
    if options_file is None:
        if config.MODEL_ARCHITECTURE == "mDCSRN_GAN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN_GAN.json')
        elif config.MODEL_ARCHITECTURE == "mDCSRN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN.json')
        elif config.MODEL_ARCHITECTURE == "SuperFormer":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_SuperFormer.json')
        elif config.MODEL_ARCHITECTURE == "ESRGAN3D":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ESRGAN3D.json')
        elif config.MODEL_ARCHITECTURE == "RRDBNet3D":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RRDBNet3D.json')
        elif config.MODEL_ARCHITECTURE == "EDDSR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_EDDSR.json')
        elif config.MODEL_ARCHITECTURE == "MFER":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MFER.json')
        elif config.MODEL_ARCHITECTURE == "MTVNet":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MTVNet.json')
        elif config.MODEL_ARCHITECTURE == "ArSSR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ArSSR.json')
        elif config.MODEL_ARCHITECTURE == "RCAN":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RCAN.json')
        elif config.MODEL_ARCHITECTURE == "SwinIR":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_SwinIR.json')
        elif config.MODEL_ARCHITECTURE == "HAT":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_HAT.json')
        elif config.MODEL_ARCHITECTURE == "DRCT":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_DRCT.json')
        elif config.MODEL_ARCHITECTURE == "DUMMY":
            opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_DUMMY.json')
        else:
            raise NotImplementedError('Model architecture %s not implemented.' % config.MODEL_ARCHITECTURE)
    else:
        opt_path = os.path.join(config.ROOT_DIR, 'options', options_file)

    # Load options
    opt = load_json(opt_path)

    # Overwrite dataset in options file if specified
    if args.dataset is not None:
        opt['datasets']['name'] = args.dataset

    if args.cluster is not None:
        opt['cluster'] = args.cluster

    # Define universal SR model using the KAIR define_Model framework
    from models.select_model import define_Model
    model = define_Model(opt, mode='train')
    model.define_wandb_run()  # Define wandb run
    model.init_train()  # Initialize model for training

    # Save copy of options file in wandb directory
    save_json(opt, wandb_path=model.run.dir)

    from data.select_dataset import define_Dataset
    train_dataset, test_dataset, baseline_dataset = define_Dataset(opt, return_filepaths=False)  # optional to have baseline dataloader as final output

    dataloader_params_train = opt['datasets']['train']['dataloader_params']
    dataloader_params_test = opt['datasets']['test']['dataloader_params']

    train_loader = monai.data.DataLoader(train_dataset,
                                         batch_size=dataloader_params_train['dataloader_batch_size'],
                                         shuffle=dataloader_params_train['dataloader_shuffle'],
                                         num_workers=dataloader_params_train['num_load_workers'],
                                         persistent_workers=dataloader_params_train['persist_workers'],
                                         pin_memory=dataloader_params_train['pin_memory'])

    test_loader = monai.data.DataLoader(test_dataset,
                                        batch_size=dataloader_params_test['dataloader_batch_size'],
                                        shuffle=dataloader_params_test['dataloader_shuffle'],
                                        num_workers=dataloader_params_test['num_load_workers'],
                                        persistent_workers=dataloader_params_test['persist_workers'],
                                        pin_memory=dataloader_params_test['pin_memory'])

    # Train model
    if opt['datasets']['dataset_type'] == "MonaiSmartCacheDataset":
        train_dataset.start()
        test_dataset.start()

    iterations = opt['train']['iterations'] * opt['train']['num_accum_steps_G']
    validation_iterations = opt['train']['validation_iterations']

    time_start = time.time()

    from torch.profiler import profile, tensorboard_trace_handler

    if args.run_profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True,
                     record_shapes=False,
                     on_trace_ready=tensorboard_trace_handler("./profiles")) as prof:
            out_dict = train_model(model, opt, iterations, validation_iterations,  train_loader, test_loader, print_status=True)
    else:
        out_dict = train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True)

    time_end = time.time()
    print("Time taken to train: ", time_end - time_start)

    print("Done")

