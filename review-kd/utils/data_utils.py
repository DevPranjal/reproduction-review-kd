import torch
import torchvision
from torch.utils import data
from torchvision import transforms


normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

def get_dataloaders(data):
    if data == "fashion_mnist":
        return get_fashion_mnist_dataloaders
    elif data == "cifar10":
        return get_cifar10_dataloaders


def get_fashion_mnist_dataloaders(batch_size, resize=224):
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    mnist_train_dataloader = data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=4)
    mnist_test_dataloader = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)

    return mnist_train_dataloader, mnist_test_dataloader, 10


def get_cifar10_dataloaders(batch_size):
    cifar10_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=train_transform, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=test_transform, download=True)

    cifar10_train_dataloader = data.DataLoader(
        cifar10_train, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    cifar10_test_dataloader = data.DataLoader(
        cifar10_test, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return cifar10_train_dataloader, cifar10_test_dataloader, 10
