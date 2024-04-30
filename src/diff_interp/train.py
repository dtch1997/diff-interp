import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from diff_interp.celeba_data_module import CelebADataModule
from diff_interp.vit_model import ViTModel

if __name__ == "__main__":

    model = ViTModel(
        learning_rate=0.05,
        freeze_backbone=True
    )
    datamodule = CelebADataModule(batch_size=256)

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices="auto",
        logger=WandbLogger(),
        callbacks=EarlyStopping('val_loss', patience=7),
    )

    trainer.fit(model = model, datamodule = datamodule)