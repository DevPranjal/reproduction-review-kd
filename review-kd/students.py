from utils.resnet_blocks import ResNet
from torch import nn


def resnet8(num_classes=10):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', num_classes)


def resnet14(num_classes=10):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', num_classes)


def resnet20(num_classes=10):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', num_classes)

def custom_student(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=11, )
    )