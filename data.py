import torch
from torchvision import datasets, transforms


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


def get_dataloaders(dataset, batch_size):
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='data/', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            root='data/', train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root='data/', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            root='data/', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, pin_memory=True, num_workers=1)

    return train_loader, test_loader
