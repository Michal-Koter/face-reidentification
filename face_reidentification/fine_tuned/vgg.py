import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from opt_einsum.backends import torch
from torch import nn, optim


class VggFace(pl.LightningModule):
    """
    A PyTorch Lightning Module for a VGG-based face verification model.

    This model uses a pretrained VGG-16 backbone for feature extraction, followed by fully connected layers for
    classification. The model is designed for binary classification tasks such as face verification.

    Attributes:
        backbone (nn.Sequential): The VGG-16 backbone excluding the final classification layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The final fully connected layer for classification.
        dropout1 (nn.Dropout): Dropout layer applied after the first fully connected layer.
        dropout2 (nn.Dropout): Dropout layer applied after the second fully connected layer.
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
        """Initializes the VggFace model with the specified number of output classes.

        Args:
            - num_classes (int, optional): The number of output classes. Default is 2.

        """
        super().__init__()

        vgg = models.vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.children())[:-1])

        self.fc1 = nn.Linear(50176, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.dropout1 = nn.Dropout(0.5, inplace=False)
        self.dropout2 = nn.Dropout(0.5, inplace=False)

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
        """Performs a forward pass through the network with input image pairs.

        Args:
            - x (torch.Tensor): The first input image tensor.
            - y (torch.Tensor): The second input image tensor.

        Returns
        -------
            - torch.Tensor: The output from the fully connected layers.

        """
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x).flatten(1)
            y = self.backbone(y).flatten(1)

        t = torch.cat((x, y), 1)

        t = self.fc1(t)
        t = nn.ReLU()(t)
        t = self.dropout1(t)
        t = self.fc2(t)
        t = nn.ReLU()(t)
        t = self.dropout2(t)
        t = self.fc3(t)

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
