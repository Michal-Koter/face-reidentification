import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class FaceDatamodule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling the Labeled Faces in the Wild (LFW) dataset for face verification tasks.

    This DataModule prepares the LFW dataset for training, validation, and testing by applying necessary transformations
    and creating data loaders for each stage.

    Attributes:
        batch_size (int): The size of the batches used for training and testing. Default is 32.
        train_dataset (torchvision.datasets.LFWPairs): The dataset for training.
        test_dataset (torchvision.datasets.LFWPairs): The dataset for testing.

    Methods:
        setup(stage=None):
            Sets up the datasets for training and testing with the required transformations.

        train_dataloader():
            Returns the DataLoader for the training dataset.

        val_dataloader():
            Returns the DataLoader for the validation dataset.

        test_dataloader():
            Returns the DataLoader for the testing dataset.
    """

    def __init__(self, batch_size: int = 32):
        """
        Initializes the FaceDatamodule with the given batch size.

        Args:
            - batch_size (int): The size of the batches used for training and testing. Default is 32.

        """
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size

    def setup(self, stage: str = None) -> None:
        """
        Set up the datasets for training and testing with the required transformations.

        This method applies a CenterCrop transformation followed by conversion to a tensor.
        It initializes the train_dataset and test_dataset attributes using the LFWPairs dataset from torchvision.

        Args:
            - stage (str): Either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``. Default is None.

        """
        transform = transforms.Compose(
            [transforms.CenterCrop(200), transforms.ToTensor()]
        )

        self.train_dataset = torchvision.datasets.LFWPairs(
            root="../data", split="train", transform=transform, download=True
        )
        self.test_dataset = torchvision.datasets.LFWPairs(
            root="../data", split="test", transform=transform, download=True
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            - DataLoader: DataLoader for the training dataset with shuffling enabled.

        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            - DataLoader: DataLoader for the validation dataset with shuffling disabled.

        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the testing dataset.

        Returns:
            - DataLoader: DataLoader for the testing dataset with shuffling disabled.

        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
