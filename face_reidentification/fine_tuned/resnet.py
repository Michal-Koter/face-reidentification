import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from torch import nn, optim


class ResFace(pl.LightningModule):
    """
    A PyTorch Lightning Module for a ResNet-based face verification model.

    This model uses a pretrained ResNet-50 backbone for feature extraction, followed by a fully connected layer
    for classification. The model is designed for binary classification tasks such as face verification.

    Attributes:
        backbone (nn.Sequential): The ResNet-50 backbone excluding the final classification layer.
        fc (nn.Linear): The fully connected layer for classification.
        loss_function (nn.CrossEntropyLoss): The loss function used for training.
        train_acc (torchmetrics.Accuracy): Accuracy metric for training.
        val_acc (torchmetrics.Accuracy): Accuracy metric for validation.
        train_macro_f1 (torchmetrics.F1Score): Macro F1 score metric for training.
        val_macro_f1 (torchmetrics.F1Score): Macro F1 score metric for validation.
        confusion_matrix (torchmetrics.ConfusionMatrix): Confusion matrix metric for binary classification.

    Methods:
        forward(x, y):
            Performs a forward pass through the network with input image pairs.

        configure_optimizers():
            Configures the optimizer for training.

        training_step(train_batch, batch_idx):
            Defines a single training step.

        validation_step(val_batch, batch_idx):
            Defines a single validation step.
    """

    def __init__(self, num_classes: int = 2) -> None:
        """
        Initializes the ResFace model with the specified number of output classes.

        Args:
            - num_classes (int, optional): The number of output classes. Default is 2.

        """
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(4096, num_classes)

        self.loss_function = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="binary", num_classes=num_classes)

        self.train_macro_f1 = torchmetrics.F1Score(
            task="binary", average="macro", num_classes=num_classes
        )
        self.val_macro_f1 = torchmetrics.F1Score(
            task="binary", average="macro", num_classes=num_classes
        )

        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task="binary", num_classes=num_classes
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network with input image pairs.

        Args:
            - x (torch.Tensor): The first input image tensor.
            - y (torch.Tensor): The second input image tensor.

        Returns:
            - torch.Tensor: The output from the fully connected layers.

        """
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x).flatten(1)
            y = self.backbone(y).flatten(1)

        t = torch.cat((x, y), 1)

        t = self.fc(t)

        return t

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            - torch.optim.Optimizer: The optimizer for training.

        """
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, train_batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Defines a single training step.

        Args:
            - train_batch (tuple): A batch of training data, containing image pairs and labels.
            - batch_idx (int): The index of the current batch.

        Returns:
            - torch.Tensor: The computed loss for the batch.

        """
        img1, img2, labels = train_batch

        labels_hot = F.one_hot(labels, num_classes=2)

        outputs = self.forward(img1, img2)

        loss = self.loss_function(outputs, labels_hot.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        outputs = F.softmax(outputs, dim=1)

        self.train_acc(outputs, labels_hot)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)

        self.train_macro_f1(outputs, labels_hot)
        self.log("train_macro_f1", self.train_macro_f1, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Defines a single validation step.

        Args:
            - val_batch (tuple): A batch of validation data, containing image pairs and labels.
            - batch_idx (int): The index of the current batch.

        Returns:
            - torch.Tensor: The computed loss for the batch.

        """
        img1, img2, labels = val_batch

        labels_hot = F.one_hot(labels, num_classes=2)

        outputs = self.forward(img1, img2)
        loss = self.loss_function(outputs, labels_hot.float())

        self.log("val_loss", loss, on_step=True, on_epoch=True)

        outputs = F.softmax(outputs, dim=1)

        self.val_acc(outputs, labels_hot)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)

        self.val_macro_f1(outputs, labels_hot)
        self.log("val_macro_f1", self.val_macro_f1, on_epoch=True, on_step=False)

        return loss
