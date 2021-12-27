import torch
from torch import nn
import json
import argparse
from tqdm import tqdm
from metrics import hcl, calculate_accuracy, RunningAverage
from utils.data_utils import get_dataloaders
from utils.net_utils import get_net
from utils.save_load_utils import is_pretrained_present, load_model, store_model

from transforms import abf

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='params.json',
                    help='json file storing params in the default file format')

args = parser.parse_args()

params = json.load(open(args.param_file))
data = params["data"]
net_type = "student_review_kd"

device = params["device"]
device = torch.device(
    'cuda:0') if device == 'cuda' else torch.device('cpu')
num_epochs = params[net_type]["num_epochs"]

kd_loss_weight = params[net_type]["kd_loss_weight"]


def train(student, teacher, train_iter, loss, optimizer):
    if is_pretrained_present(params, net_type):
        print('using pretrained')
        student = load_model(params, net_type)
        return student

    average_loss = RunningAverage()

    print("\nstarting training")

    student.train()
    teacher.eval()

    for _ in range(num_epochs):
        with tqdm(total=len(train_iter)) as t:
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)

                student_features, student_preds = student(X, with_features=True)
                with torch.no_grad():
                    teacher_features, teacher_preds = teacher(
                        X, with_features=True)

                ce_loss = nn.CrossEntropyLoss()(student_preds, y)

                student_features = student_features[::-1]
                teacher_features = teacher_features[::-1]

                total_kd_loss = 0

                prev_abf_output = student_features[0]

                for sf, tf in zip(student_features[1:], teacher_features[1:]):
                    # print("SF:\n", sf.shape)
                    # print("PREV ABF OUTPUT:\n", prev_abf_output.shape)
                    abf_output = abf(sf, prev_abf_output, device)
                    total_kd_loss += loss(abf_output, tf)
                    prev_abf_output = abf_output
                
                total_loss = ce_loss + total_kd_loss * kd_loss_weight
                # print(total_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                average_loss.update(total_loss.item())
                t.set_postfix(loss=f'{average_loss():.3f}')
                t.update()

    store_model(student, params, net_type)

    return student


def test(net, test_iter):
    accuracy = calculate_accuracy(net, test_iter, device)
    print(f"test accuracy = {accuracy}")


if __name__ == "__main__":
    # defining dataloaders
    train_iter, test_iter, num_classes = get_dataloaders(
        data)(params[net_type]["batch_size"])

    # get network based on model
    teacher = load_model(params, "teacher")
    print("testing teacher")
    test(teacher, test_iter)
    student = load_model(params, "student")
    print("testing student")
    test(student, test_iter)

    student_kd = get_net(params[net_type]["model"])(num_classes)

    print(f"\ntransferring model to device ({device})")
    teacher.to(device)
    student.to(device)
    student_kd.to(device)

    # define loss
    loss = hcl

    # define optimizer
    optimizer = torch.optim.SGD(student_kd.parameters(), lr=params[net_type]["lr"])

    # train
    student_kd = train(student_kd, teacher, train_iter, loss, optimizer)

    # test
    test(student_kd, test_iter)
