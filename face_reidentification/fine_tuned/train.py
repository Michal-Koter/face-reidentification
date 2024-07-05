import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from face_reidentification.fine_tuned import *
from face_reidentification.utils import dataset


def train_model(
    model: pl.LightningModule, name: str, datamodule: pl.LightningDataModule
) -> None:
    """
    Trains a PyTorch Lightning model using the specified data module and saves the trained model.

    This function sets up a TensorBoard logger and a model checkpoint callback for monitoring and saving the model
    during training. It then trains the model for a specified number of epochs.

    Args:
        - model (pl.LightningModule): The PyTorch Lightning model to be trained.
        - name (str): The name to be used for logging and saving the model.
        - datamodule (pl.LightningDataModule): The data module providing the training and validation data.

    """
    logger = TensorBoardLogger("../models/lightning_logs", name=name)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath="../models", filename=f"{name}_checkpoint"
    )

    trainer = pl.Trainer(
        logger=logger, max_epochs=25, callbacks=[checkpoint], log_every_n_steps=1
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    dm = dataset.FaceDatamodule()

    train_model(DenseFace(), name="densenet", datamodule=dm)
    train_model(GoogleFace(), name="googlenet", datamodule=dm)
    train_model(ResFace(), name="resnet", datamodule=dm)
    train_model(SqueezeFace(), name="squeezenet", datamodule=dm)
    train_model(VggFace(), name="vgg", datamodule=dm)
