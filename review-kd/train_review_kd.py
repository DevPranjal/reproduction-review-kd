import torch
from torch import nn
import json
import argparse
from tqdm import tqdm

from metrics import hcl, calculate_accuracy, RunningAverage

from utils.data_utils import get_dataloaders
from utils.net_utils import get_net
from utils.save_load_utils import is_pretrained_present, load_model, store_model
from utils.plot_utils import Animator

from transforms import build_abfs_for_resnet

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='params.json',
                    help='json file storing params in the default file format')

args = parser.parse_args()

params = json.load(open(args.param_file))
data = params["data"]
net_type = "student_review_kd"

device = params["device"]
device = torch.device('cuda:0') if device == 'cuda' else torch.device('cpu')
num_epochs = params[net_type]["num_epochs"]

kd_loss_weight = params[net_type]["kd_loss_weight"]


def train(student, teacher, abfs, train_iter, loss_base, loss_kd, optimizer):
    if is_pretrained_present(params, net_type):
        print('using pretrained')
        student = load_model(params, net_type)
        print('\ntesting student review kd')
        test(student, test_iter)
        return student

    print("\nstarting training")

    student.train()
    teacher.eval()

    average_loss = RunningAverage()

    animator = Animator(xlabel='iteration', xlim=[1, num_epochs * len(train_iter)],
                        legend=['total norm'])

    for e in range(num_epochs):
        print(f'epoch: {e}')
        with tqdm(total=len(train_iter)) as t:
            for i, (X, y) in enumerate(train_iter):
                X, y = X.to(device), y.to(device)

                student_features, student_preds = student(X, with_features=True)
                with torch.no_grad():
                    teacher_features, teacher_preds = teacher(X, with_features=True)

                ce_loss = loss_base(student_preds, y)

                student_features = student_features[::-1]
                teacher_features = teacher_features[::-1]

                # for i, sf in enumerate(student_features):
                #     print(f'sf {i}: {sf.shape}')
                # for i, tf in enumerate(teacher_features):
                #     print(f'tf {i}: {tf.shape}')

                total_kd_loss = 0
                total_loss = 0

                residual_output = None

                for i, (sf, tf) in enumerate(zip(student_features, teacher_features)):
                    abf_output, residual_output = abfs[i](sf, residual_output)
                    total_kd_loss += loss_kd(abf_output, tf)

                    # print(f'RESIDUAL OUTPUT {i} SHAPE: {residual_output.shape}')
                    # print(f'STUDENT FEATURE {i} SHAPE: {sf.shape}')
                    # print(f'ABF OUTPUT {i} SHAPE: {abf_output.shape}')
                    # print(f'TEACHER FEATURE {i} SHAPE: {tf.shape}')

                # total_loss += ce_loss
                total_loss += total_kd_loss * kd_loss_weight

                # total_norms = []

                total_norm = 0
                parameters = [p for p in student.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                # total_norms.append(total_norm)
                # animator.add(i, total_norm)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # del residual_output
                # del abf_output

                average_loss.update(total_loss.item())
                t.set_postfix(loss=f'{average_loss():.3f}', total_norm=f'{total_norm:.3f}')
                t.update()

        print('testing:')
        test(student, test_iter)

        print(total_norms)

    store_model(student, params, net_type)

    return student


def test(net, test_iter):
    accuracy = calculate_accuracy(net, test_iter, device)
    print(f"test accuracy = {accuracy}")


if __name__ == "__main__":
    # defining dataloaders
    train_iter, test_iter, num_classes = get_dataloaders(data)(params[net_type]["batch_size"])

    # get network based on model and test
    print()
    teacher = load_model(params, "teacher")
    print("testing teacher")
    test(teacher, test_iter)

    print()
    student = load_model(params, "student")
    print("testing base student")
    test(student, test_iter)

    student_kd = get_net(params[net_type]["model"])(num_classes)

    print(f"\ntransferring model to device ({device})")
    teacher.to(device)
    student.to(device)
    student_kd.to(device)

    # define loss
    loss_base = nn.CrossEntropyLoss()
    loss_kd = hcl

    # define optimizer
    optimizer = torch.optim.SGD(
        student_kd.parameters(),
        lr=params[net_type]["lr"],
        # momentum=0.9,
        # nesterov=True,
        # weight_decay=5e-3
    )

    # define transforms
    abfs = build_abfs_for_resnet(device)

    # train
    student_kd = train(student_kd, teacher, abfs, train_iter, loss_base, loss_kd, optimizer)

    # test
    test(student_kd, test_iter)
