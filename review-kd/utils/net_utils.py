from students import *
from teachers import *


def get_net(model):
    if model == 'resnet8':
        return resnet14
    elif model == 'resnet14':
        return resnet14
    elif model == 'resnet20':
        return resnet20
    elif model == 'resnet44':
        return resnet44
    elif model == 'resnet56':
        return resnet56
    elif model == 'resnet110':
        return resnet110
