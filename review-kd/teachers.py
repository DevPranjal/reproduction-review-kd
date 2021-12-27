from utils.resnet_blocks import ResNet


def resnet44(num_classes=10):
    return ResNet(44, [16, 16, 32, 64], 'basicblock', num_classes)


def resnet56(num_classes=10):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', num_classes)


def resnet110(num_classes=10):
    return ResNet(110, [16, 16, 32, 64], 'basicblock', num_classes)
