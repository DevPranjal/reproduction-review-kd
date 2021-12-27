import torch
import torchvision
from torch.utils import data
from torchvision import transforms


def get_dataloaders(data):
    if data == "fashion_mnist":
        return get_fashion_mnist_dataloaders
    elif data == "cifar10":
        return get_cifar10_dataloaders


def get_fashion_mnist_dataloaders(batch_size, resize=224):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    mnist_train_dataloader = data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=4)
    mnist_test_dataloader = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)

    return mnist_train_dataloader, mnist_test_dataloader, 10


def get_cifar10_dataloaders(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True)

    mnist_train_dataloader = data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=4)
    mnist_test_dataloader = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)

    return mnist_train_dataloader, mnist_test_dataloader, 10
