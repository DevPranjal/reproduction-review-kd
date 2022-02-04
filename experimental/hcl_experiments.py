import sys
sys.path.insert(0, '..')


from train import train
from params import params
from hcl_experiments import *
from abf_experiments import *
from framework import ABF, hcl, RLF_for_Resnet
import torch
import torch.nn.functional as F
from torch import nn


def hcl_level_1(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [h, 2, 1]
    level_weight = [1.0, 0.5, 0.25]
    total_weight = sum(level_weight)

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


def hcl_level_2(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [4, 1]
    level_weight = [1.0, 0.5]
    total_weight = sum(level_weight)

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


def hcl_level_3(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [h, h//2, h//4]
    level_weight = [1.0, 0.5, 0.25]
    total_weight = sum(level_weight)

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


def hcl_level_4(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [h, h-1, h-2, h-3]
    level_weight = [1.0, 0.5, 0.25, 0.125]
    total_weight = sum(level_weight)

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


def hcl_weight_1(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [h, 4, 2, 1]
    level_weight = [1.0, 1.0, 1.0, 1.0]
    total_weight = sum(level_weight)

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


def hcl_weight_2(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [h, 4, 2, 1]
    level_weight = [0.125, 0.25, 0.5, 1.0]
    total_weight = sum(level_weight)

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


def hcl_no_levels_l2(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [h]
    level_weight = [1]
    total_weight = sum(level_weight)

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


if __name__ == '__main__':
    # varying the levels of pooling
    train(params, hcl_level_1, ABF, RLF_for_Resnet, log_file_suffix='hcl_level_1')
    params["lr"] = 0.1; train(params, hcl_level_2, ABF, RLF_for_Resnet, log_file_suffix='hcl_level_2')
    params["lr"] = 0.1; train(params, hcl_level_3, ABF, RLF_for_Resnet, log_file_suffix='hcl_level_3')
    params["lr"] = 0.1; train(params, hcl_level_4, ABF, RLF_for_Resnet, log_file_suffix='hcl_level_4')

    # # varying the weights assigned to each level
    params["lr"] = 0.1; train(params, hcl_weight_1, ABF, RLF_for_Resnet, log_file_suffix='hcl_weight_1')
    params["lr"] = 0.1; train(params, hcl_weight_2, ABF, RLF_for_Resnet, log_file_suffix='hcl_weight_2')
