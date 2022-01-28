from utils.resnets_for_cifar import ResNet


def resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def get_teacher(name):
    if name == 'resnet44':
        return resnet44()
    elif name == 'resnet56':
        return resnet56()
    elif name == 'resnet110':
        return resnet110()
