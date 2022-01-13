import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def hcl(student_features, teacher_features):
    loss = 0.0
    n, c, h, w = student_features.shape

    levels = [h, 4, 2, 1]
    lvl_weight = 1.0
    total_weight = 0.0

    for i, lvl in enumerate(levels):
        if lvl > h:
            continue

        lvl_sf = F.adaptive_avg_pool2d(student_features, (lvl, lvl))
        lvl_tf = F.adaptive_avg_pool2d(teacher_features, (lvl, lvl))

        # print(f'LOSS LEVEL {i} SHAPE: {lvl_sf.shape}')

        lvl_loss = F.mse_loss(lvl_sf, lvl_tf) * lvl_weight
        loss += lvl_loss

        total_weight += lvl_weight
        lvl_weight = lvl_weight / 2.0

    return loss / total_weight


def calculate_accuracy(net, test_iter, device):
    num_test_examples = 0
    num_correct_preds = 0

    net.eval()
    with torch.no_grad():
        with tqdm(total=len(test_iter)) as t:
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                preds = net(X)
                preds = preds.argmax(axis=1)
                comp = preds.type(y.dtype) == y
                num_correct_preds += float(comp.type(y.dtype).sum())
                num_test_examples += y.numel()
                accuracy = num_correct_preds / num_test_examples

                t.set_postfix(accuracy=f'{accuracy:.3f}')
                t.update()

    return accuracy


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


if __name__ == "__main__":
    features1 = torch.ones([2, 2, 2, 2], dtype=torch.float32)
    features2 = torch.ones([2, 2, 2, 2], dtype=torch.float32) * 2
    print(hcl(features1, features2))
