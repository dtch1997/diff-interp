import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics import Accuracy
from transformers import AutoModelForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput


CHECKPOINT = "google/vit-base-patch16-224-in21k"
INPUT_SHAPE = (3, 224, 224)
# NOTE: Assume "attribute" classification
NUM_CLASSES = 40

def load_model(checkpoint, freeze_backbone=True):
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
    )
    # Freeze backbone
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    last_layer_input_size = model.classifier.in_features
    new_classification_head = torch.nn.Linear(last_layer_input_size, NUM_CLASSES)
    model.classifier = new_classification_head
    return model

class ViTModel(pl.LightningModule):
    def __init__(
        self, 
        learning_rate: float = 0.05, 
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        # load model
        self.model = load_model(CHECKPOINT, freeze_backbone=self.hparams.freeze_backbone)
        self.train_accuracy = Accuracy(task="multilabel", num_labels=NUM_CLASSES)
        self.val_accuracy = Accuracy(task="multilabel", num_labels=NUM_CLASSES)
        self.loss_fn = nn.BCEWithLogitsLoss()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: ImageClassifierOutput = self.model(x)
        return output.logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        acc = self.train_accuracy(logits, y)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        acc = self.val_accuracy(logits, y)
        self.log("val_accuracy", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            momentum=0.9
        )
        return optimizer