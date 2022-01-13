import torch
from torch import nn
import torch.nn.functional as F


# def abf(features1, features2, device):
#     features2 = upsample_and_match_channels(features2, features1, device)

#     features_concat = torch.cat((features1, features2), dim=1)
#     n, c, h, w = features_concat.shape

#     att_maps = nn.Conv2d(c, 2, kernel_size=1).to(device)(features_concat)
#     att_map1, att_map2 = att_maps[:, 0, :, :], att_maps[:, 1, :, :]
#     att_map1, att_map2 = att_map1.reshape(
#         (n, 1, h, w)), att_map2.reshape((n, 1, h, w))

#     return features1 * att_map1 + features2 * att_map2


class ABF(nn.Module):
    def __init__(self, in_channel, out_channel, teacher_shape, device):
        super(ABF, self).__init__()

        self.teacher_shape = teacher_shape
        self.mid_channel = 64

        self.conv_to_mid_channel = nn.Sequential(
            nn.Conv2d(in_channel, self.mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channel),
        ).to(device)
        nn.init.kaiming_uniform_(self.conv_to_mid_channel[0].weight, a=1)

        self.conv_to_out_channel = nn.Sequential(
            nn.Conv2d(self.mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ).to(device)
        nn.init.kaiming_uniform_(self.conv_to_out_channel[0].weight, a=1)

        self.conv_to_att_maps = nn.Sequential(
            nn.Conv2d(self.mid_channel * 2, 2, kernel_size=1),
            nn.Sigmoid(),
        ).to(device)
        nn.init.kaiming_uniform_(self.conv_to_att_maps[0].weight, a=1)

    def forward(self, student_feature, prev_abf_output):
        n, c, h, w = student_feature.shape
        student_feature = self.conv_to_mid_channel(student_feature)

        if prev_abf_output is None:
            residual_output = student_feature
        else:
            prev_abf_output = F.interpolate(prev_abf_output, size=(self.teacher_shape, self.teacher_shape), mode='nearest')
            
            concat_features = torch.cat([student_feature, prev_abf_output], dim=1)
            attention_maps = self.conv_to_att_maps(concat_features)
            attention_map1 = attention_maps[:, 0].view(n, 1, h, w)
            attention_map2 = attention_maps[:, 1].view(n, 1, h, w)

            residual_output = student_feature * attention_map1 + prev_abf_output * attention_map2

        abf_output = self.conv_to_out_channel(residual_output)

        return abf_output, residual_output


def build_abfs_for_resnet(device):
    abf1 = ABF(64, 64, 1, device)
    abf2 = ABF(64, 64, 8, device)
    abf3 = ABF(32, 32, 16, device)
    abf4 = ABF(16, 16, 32, device)
    abf5 = ABF(16, 16, 32, device)

    return [abf1, abf2, abf3, abf4, abf5]


# def upsample_and_match_channels(features2, features1, device):
#     """Upsamples features2 to match features1"""
#     features2 = nn.Conv2d(
#         features2.shape[1],
#         features1.shape[1],
#         kernel_size=1
#     ).to(device)(features2)

#     return F.interpolate(features2, (features1.shape[2], features1.shape[3]))


# if __name__ == "__main__":
#     features1 = torch.randint(10, [2, 3, 2, 2], dtype=torch.float32)
#     features2 = torch.randint(10, [2, 6, 4, 4], dtype=torch.float32)
#     print(abf(features1, features2))
