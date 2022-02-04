# framework.py consists of implementations of the proposals made by the authors
# We largely refer the original implementation of the paper by the authors,
# mainly refactoring their code

# Distilling Knowledge via Knowledge Review
# |----> Uses Residual Learning Framework
#        |----> Uses Hierarchical Context Loss
#        |----> Uses Attention Based Fusion Module

# Let us start from the bottom going upwards

# We implement the framework for general ResNet architectures only


import torch
import torch.nn.functional as F
from torch import nn


########## Hierarchical Context Loss ##########

def hcl(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    # the levels of hcl loss here are predefined, according to
    # the authors' implementation
    # ablation studies have been performed and these levels and their
    # weights have been changed (refer experimental/hcl_experiments.py)
    levels = [h, 4, 2, 1]
    level_weight = [1.0, 0.5, 0.25, 0.125]
    total_weight = sum(level_weight)
    # keeping the corresponding weights same, we change the levels to:
    # [h, 2, 1], [4, 1], [h, h//2, h//4], [h, h-1, h-2, h-3]
    # keeping the levels same, we change the weights to:
    # [1.0, 1.0, 1.0, 1.0], [0.125, 0.25, 0.5, 1.0]

    for lvl, lvl_weight in zip(levels, level_weight):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

    return loss / total_weight


########## Attention Based Fusion Module ##########

# Paper states that the output from the ABF module (single output as
# presented in the ABF flow diagram, fig. 3(a)) is the one of the inputs to
# the next ABF module.

# But the authors' code implementation provides two different outputs, one that
# proceeds to the next ABF module (`residual_output`) and one that
# is the output of the ABF module and which is involved in the loss
# function (`abf_output`)
# The `residual_output` differs from the `abf_output` in terms of the number
# of channels. The `residual_output` has `mid_channels` while the `abf_output`
# has `out_channels`

# In this implementation, we have taken the latter approach

# The second approach can be found in experimental/abf_experiments.py

class ABF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ABF, self).__init__()

        self.mid_channel = 64

        self.conv_to_mid_channel = nn.Sequential(
            nn.Conv2d(in_channel, self.mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channel),
        ).to(torch.device('cuda:0'))
        nn.init.kaiming_uniform_(self.conv_to_mid_channel[0].weight, a=1)

        self.conv_to_out_channel = nn.Sequential(
            nn.Conv2d(self.mid_channel, out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ).to(torch.device('cuda:0'))
        nn.init.kaiming_uniform_(self.conv_to_out_channel[0].weight, a=1)

        self.conv_to_att_maps = nn.Sequential(
            nn.Conv2d(self.mid_channel * 2, 2, kernel_size=1),
            nn.Sigmoid(),
        ).to(torch.device('cuda:0'))
        nn.init.kaiming_uniform_(self.conv_to_att_maps[0].weight, a=1)

    def forward(self, student_feature, prev_abf_output, teacher_shape):
        n, c, h, w = student_feature.shape
        student_feature = self.conv_to_mid_channel(student_feature)

        if prev_abf_output is None:
            residual_output = student_feature
        else:
            prev_abf_output = F.interpolate(prev_abf_output, size=(
                teacher_shape, teacher_shape), mode='nearest')

            concat_features = torch.cat(
                [student_feature, prev_abf_output], dim=1)
            attention_maps = self.conv_to_att_maps(concat_features)
            attention_map1 = attention_maps[:, 0].view(n, 1, h, w)
            attention_map2 = attention_maps[:, 1].view(n, 1, h, w)

            residual_output = student_feature * attention_map1 \
                + prev_abf_output * attention_map2

        # the output of the abf is obtained after the residual
        # output is convolved to have `out_channels` channels
        abf_output = self.conv_to_out_channel(residual_output)

        return abf_output, residual_output


########## Residual Learning Framework ##########

class RLF_for_Resnet(nn.Module):
    def __init__(self, student, abf_to_use):
        super(RLF_for_Resnet, self).__init__()

        self.student = student

        in_channels = [16, 32, 64, 64]
        out_channels = [16, 32, 64, 64]

        self.shapes = [1, 8, 16, 32, 32]

        ABFs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            ABFs.append(abf_to_use(in_channel, out_channels[idx]))

        self.ABFs = ABFs[::-1]
        self.to('cuda')

    def forward(self, x):
        student_features = self.student(x, is_feat=True)

        student_preds = student_features[1]
        student_features = student_features[0][::-1]

        results = []

        abf_output, residual_output = self.ABFs[0](
            student_features[0], None, self.shapes[0])

        results.append(abf_output)

        for features, abf, shape in zip(student_features[1:], self.ABFs[1:], self.shapes[1:]):
            # here we use a recursive technique to obtain all the ABF
            # outputs and store them in a list
            abf_output, residual_output = abf(features, residual_output, shape)
            results.insert(0, abf_output)

        return results, student_preds
