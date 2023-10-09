import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_training_dataloader(
    cifar_10_dir: str = "cifar-10-torch",
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for training data.

    Parameters
    ----------
    cifar_10_dir: str
        Path to CIFAR10 data root in torchvision format.
    batch_size: int
        Batch size for dataloader.
    num_workers: int
        Number of subprocesses for data loading.
    shuffle: bool
        Flag for shuffling training data.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader for training data.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    )
    cifar10_training = torchvision.datasets.CIFAR10(
        root=cifar_10_dir,
        train=True,
        download=True,
        transform=transform_train,
    )
    cifar10_training_loader = DataLoader(
        cifar10_training,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    return cifar10_training_loader


def get_test_dataloader(
    cifar_10_dir="cifar-10-torch",
    batch_size=16,
    num_workers=2,
    shuffle=True,
):
    """Create DataLoader for test data.

    Parameters
    ----------
    cifar_10_dir: str
        Path to CIFAR10 data root in torchvision format.
    batch_size: int
        Batch size for dataloader.
    num_workers: int
        Number of subprocesses for data loading.
    shuffle: bool
        Flag for shuffling training data.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader for test data.
    """
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root=cifar_10_dir,
        train=False,
        download=True,
        transform=transform_test,
    )
    cifar10_test_loader = DataLoader(
        cifar10_test,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    return cifar10_test_loader
