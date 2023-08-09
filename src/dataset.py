from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class CIFAR10Dataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class CIFAR10DataModule(LightningDataModule):

    def __init__(self):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None

    def setup(self, stage):
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2470, 0.2435, 0.2616]
        # transforms for images
        train_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=4, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8,
                                fill_value=means),
                ToTensorV2(),
            ]
        )

        test_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )

        # prepare transforms standard to MNIST
        self.train_dataset = CIFAR10Dataset(root='./data', train=True, download=True, transform=train_transforms)
        self.test_dataset = CIFAR10Dataset(root='./data', train=False, download=True, transform=test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=512)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=512)
