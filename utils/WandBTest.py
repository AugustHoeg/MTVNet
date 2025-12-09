import os
import config
import torch
import torchvision.models
import wandb
import random

if __name__ == "__main__":

    wandb.init(
        # set the wandb project where this run will be logged
        entity="aulho",
        project="Xtreme-CT",
        name="Test Run",
        id="ID000000",
        notes="Note text here",
        dir="logs/" + config.DATASET,
        #dir="./",

        # track hyperparameters and run metadata
        config={

            "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "dataset": config.DATASET,
            "up_factor": config.UP_FACTOR,
            "architecture": "mDCSRN-GAN",

        }

    )

    #with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:

    config = wandb.config
    model_config = {"num_dense_blocks": 4,
                    "num_dense_units": 4,
                    "kernel_sizes": [3, 6],
                    "activation": "PReLU"}

    model_artifact = wandb.Artifact(
        "mDCSRN-GAN", type="model",
        description="Multi-Level Densely Connected Network",
        metadata=dict(model_config))

    test_model = torchvision.models.vgg19(weights="DEFAULT")
    # Look at this link for how to construct an artifact with a more neat file structure:
    # https://docs.wandb.ai/guides/artifacts/construct-an-artifact

    torch.save(test_model.state_dict(), os.path.join(wandb.run.dir,"model.h5"))
    model_artifact.add_file(os.path.join(wandb.run.dir,"model.h5"))
    #wandb.save("../saved_loss_dicts/IXI/files/model.pth")
    wandb.run.log_artifact(model_artifact)

    loss_weights_config = {
        "MSE": 0,  # 1e-2 for WGAN-GP
        "L1": 1.0,  # 1e-2 for WGAN-GP. L1 loss is used in mDCSRN-GAN paper as opposed to L2/MSE loss
        "BCE_Logistic": 1.0,
        "BCE": 1.0,
        "VGG": 0,  # 0.006
        "VGG3D": 0,  # 0.006,
        "GRAD": 0.1,  # 0.1
        "LAPLACE": 0,
        "TV3D": 0,  # 0.1
        "TEXTURE3D": 0,  # 0.5
        "ADV": 10 ** -3,  # 10**-3,  # is 0.1 in mDCSRN-GAN paper, but uses WGAN-GP instead of vanilla GAN
        "STRUCTURE_TENSOR": 0  # 10**-7
    }

    loss_artifact = wandb.Artifact(
        "Losses", type="dict",
        description="Model training/validation losses",
        metadata=dict(loss_weights_config))

    #loss_artifact.new_file()
    #wandb.run.log_artifact(loss_artifact)

    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

    print("Done")

    # [optional] finish the wandb run, necessary in notebooks
    #wandb.finish()