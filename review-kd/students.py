from utils.resnet_blocks import ResNet


def resnet8(num_classes=10):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', num_classes)


def resnet14(num_classes=10):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', num_classes)


def resnet20(num_classes=10):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', num_classes)
