import torch
from torch import nn
import json
import argparse
from tqdm import tqdm
from metrics import hcl, calculate_accuracy, RunningAverage
from utils.data_utils import get_dataloaders
from utils.net_utils import get_net
from utils.save_load_utils import is_pretrained_present, load_model, store_model

from transforms import build_abfs_for_resnet

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


def train(student, teacher, abfs, train_iter, loss, optimizer):
    if is_pretrained_present(params, net_type):
        print('using pretrained')
        student = load_model(params, net_type)
        return student


    print("\nstarting training")

    student.train()
    teacher.eval()

    for _ in range(num_epochs):
        running_loss = 0
        num_steps = 0

        with tqdm(total=len(train_iter)) as t:
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)

                student_features, student_preds = student(X, with_features=True)
                with torch.no_grad():
                    teacher_features, teacher_preds = teacher(X, with_features=True)

                ce_loss = nn.CrossEntropyLoss()(student_preds, y)

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
                    # print(f'RESIDUAL OUTPUT {i} SHAPE: {residual_output.shape}')
                    # print(f'STUDENT FEATURE {i} SHAPE: {sf.shape}')
                    # print(f'ABF OUTPUT {i} SHAPE: {abf_output.shape}')
                    # print(f'TEACHER FEATURE {i} SHAPE: {tf.shape}')
                    total_kd_loss += loss(abf_output, tf)
                
                total_loss += ce_loss
                # total_loss += total_kd_loss * kd_loss_weight

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                del residual_output
                del abf_output

                running_loss += total_loss
                num_steps += 1
                t.set_postfix(loss=f'{running_loss / num_steps:.3f}')
                t.update()

        test(student, test_iter)

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
    optimizer = torch.optim.SGD(
        student_kd.parameters(),
        lr=params[net_type]["lr"],
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-3
    )

    # define transforms
    abfs = build_abfs_for_resnet(device)

    # train
    student_kd = train(student_kd, teacher, abfs, train_iter, loss, optimizer)

    # test
    test(student_kd, test_iter)
