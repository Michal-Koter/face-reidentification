import torch
import torchvision.models as models
from torch import nn


class Encoder(nn.Module):
    """
    A neural network encoder for extracting feature vectors from images using various pretrained models.

    The `Encoder` class allows for the selection of different pretrained models (DenseNet, GoogleNet, ResNet,
    SqueezeNet, VGG) to be used as the backbone for feature extraction. The selected model is used to generate
    feature vectors from input images.

    Attributes:
        backbone (nn.Sequential): The selected pretrained model's layers, excluding the final classification layer.

    Methods:
        densenet():
            Initializes DenseNet-121 as the backbone model.

        googlenet():
            Initializes GoogleNet as the backbone model.

        resnet():
            Initializes ResNet-50 as the backbone model.

        squeezenet():
            Initializes SqueezeNet1.0 as the backbone model.

        vgg():
            Initializes VGG-16 as the backbone model.

        forward(x):
            Performs a forward pass through the backbone model, extracting and returning the feature vector.
    """

    def __init__(self, encoder_name: str = "densenet") -> None:
        """
        Initializes the Encoder with the specified model.

        Args:
            - encoder_name (str, optional): The name of the encoder model to use. Default is "densenet".

        Raises:
            ValueError: If the specified encoder_name is not supported.

        """
        super().__init__()
        # self.backbone = None

        match encoder_name:
            case "densenet":
                self.densenet()
            case "googlenet":
                self.googlenet()
            case "resnet":
                self.resnet()
            case "squeezenet":
                self.squeezenet()
            case "vgg":
                self.vgg()
            case _:
                raise ValueError(f"Encoder {encoder_name} not supported")

    def densenet(self) -> None:
        """Initializes DenseNet-121 as the backbone model."""
        densenet = models.densenet121(pretrained=True)
        self.backbone = nn.Sequential(*list(densenet.children())[:-1])

    def googlenet(self) -> None:
        """Initializes GoogleNet as the backbone model."""
        google = models.googlenet(pretrained=True)
        self.backbone = nn.Sequential(*list(google.children())[:-1])

    def resnet(self) -> None:
        """Initializes ResNet-50 as the backbone model."""
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def squeezenet(self) -> None:
        """Initializes SqueezeNet1.0 as the backbone model."""
        squeezenet = models.squeezenet1_0(pretrained=True)
        self.backbone = nn.Sequential(*list(squeezenet.children())[:-1])

    def vgg(self) -> None:
        """Initializes VGG-16 as the backbone model."""
        vgg = models.vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the backbone model, extracting and returning the feature vector.

        Args:
        -------
            - x (torch.Tensor): The input tensor representing the image.

        Returns:
            - torch.Tensor: The extracted feature vector.

        """
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x)

        x = x.flatten(1)
        return x
