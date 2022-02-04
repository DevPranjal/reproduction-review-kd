import sys
sys.path.insert(0, '..')


from train import train
from params import params
from hcl_experiments import *
from abf_experiments import *
from framework import ABF, hcl
import torch
import torch.nn.functional as F
from torch import nn


class ABF_without_mid_channels(nn.Module):
    def __init__(self, in_channel, out_channel, pabf_channel):
        super(ABF_without_mid_channels, self).__init__()

        self.conv_to_out_channel_sf = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ).to(torch.device('cuda:0'))
        nn.init.kaiming_uniform_(self.conv_to_out_channel_sf[0].weight, a=1)

        self.conv_to_out_channel_pabf = nn.Sequential(
            nn.Conv2d(pabf_channel, out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ).to(torch.device('cuda:0'))
        nn.init.kaiming_uniform_(self.conv_to_out_channel_pabf[0].weight, a=1)

        self.conv_to_att_maps = nn.Sequential(
            nn.Conv2d(out_channel * 2, 2, kernel_size=1),
            nn.Sigmoid(),
        ).to(torch.device('cuda:0'))
        nn.init.kaiming_uniform_(self.conv_to_att_maps[0].weight, a=1)

    def forward(self, student_feature, prev_abf_output, teacher_shape):
        n, c, h, w = student_feature.shape
        student_feature = self.conv_to_out_channel_sf(student_feature)

        if prev_abf_output is None:
            residual_output = student_feature
        else:
            print(prev_abf_output.shape)
            prev_abf_output = self.conv_to_out_channel_pabf(prev_abf_output)
            prev_abf_output = F.interpolate(prev_abf_output, size=(
                teacher_shape, teacher_shape), mode='nearest')

            concat_features = torch.cat(
                [student_feature, prev_abf_output], dim=1)
            attention_maps = self.conv_to_att_maps(concat_features)
            attention_map1 = attention_maps[:, 0].view(n, 1, h, w)
            attention_map2 = attention_maps[:, 1].view(n, 1, h, w)

            residual_output = student_feature * attention_map1 \
                + prev_abf_output * attention_map2

        # here we just equate both the outputs instead of having
        # a single output to have the same training code for both
        # the implementations
        abf_output = residual_output

        return abf_output, residual_output


class ABF_without_attention_maps(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ABF_without_attention_maps, self).__init__()

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

            residual_output = student_feature + prev_abf_output

        # the output of the abf is obtained after the residual
        # output is convolved to have `out_channels` channels
        abf_output = self.conv_to_out_channel(residual_output)

        return abf_output, residual_output


class RLF_for_Resnet_with_ABF_without_mid_channels(nn.Module):
    def __init__(self, student, abf_to_use):
        super(RLF_for_Resnet_with_ABF_without_mid_channels, self).__init__()

        self.student = student

        in_channels = [16, 32, 64, 64]
        out_channels = [16, 32, 64, 64]
        pabf_channels = [1, 16, 32, 64]

        self.shapes = [1, 8, 16, 32, 32]

        ABFs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            ABFs.append(abf_to_use(in_channel, out_channels[idx], pabf_channels[idx]))

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


if __name__ == '__main__':
    train(params, hcl, ABF_without_attention_maps,
        log_file_suffix='abf_without_attention_maps')
    params['lr'] = 0.1
    train(params, hcl, ABF_without_mid_channels, RLF_for_Resnet_with_ABF_without_mid_channels,
          log_file_suffix='abf_without_mid_channels')
