import torch
from torch import nn
import torch.nn.functional as F


def abf(features1, features2, device):
    features2 = upsample_and_match_channels(features2, features1, device)

    features_concat = torch.cat((features1, features2), dim=1)
    n, c, h, w = features_concat.shape

    att_maps = nn.Conv2d(c, 2, kernel_size=1).to(device)(features_concat)
    att_map1, att_map2 = att_maps[:, 0, :, :], att_maps[:, 1, :, :]
    att_map1, att_map2 = att_map1.reshape(
        (n, 1, h, w)), att_map2.reshape((n, 1, h, w))

    return features1 * att_map1 + features2 * att_map2


def upsample_and_match_channels(features2, features1, device):
    """Upsamples features2 to match features1"""
    features2 = nn.Conv2d(
        features2.shape[1],
        features1.shape[1],
        kernel_size=1
    ).to(device)(features2)

    return F.interpolate(features2, (features1.shape[2], features1.shape[3]))


if __name__ == "__main__":
    features1 = torch.randint(10, [2, 3, 2, 2], dtype=torch.float32)
    features2 = torch.randint(10, [2, 6, 4, 4], dtype=torch.float32)
    print(abf(features1, features2))
