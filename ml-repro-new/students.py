from utils.resnets_for_cifar import ResNet


def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def get_student(name):
    if name == 'resnet8':
        return resnet8()
    elif name == 'resnet14':
        return resnet14()
    elif name == 'resnet20':
        return resnet20()
    elif name == 'resnet32':
        return resnet32()
