import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import Accuracy

class LightningResNet18(pl.LightningModule):
    def __init__(self, num_classes: int = 1000, learning_rate: float = 1e-3):
        super(LightningResNet18, self).__init__()
        self.save_hyperparameters()
        
        # Load a pre-trained ResNet18 model
        self.model = models.resnet18()
        
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Define a criterion for loss computation
        self.criterion = nn.CrossEntropyLoss()

        # Define a metric for accuracy computation
        self.accuracy = Accuracy(
            task='MULTICLASS',
            num_classes=num_classes,
        )

        
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Log training loss and accuracy
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(outputs, labels), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Log validation loss and accuracy
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(outputs, labels), prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer