import torch
from torch import nn
import json
import argparse
from tqdm import tqdm
from metrics import calculate_accuracy, RunningAverage
from utils.data_utils import get_dataloaders
from utils.net_utils import get_net
from utils.save_load_utils import is_pretrained_present, load_model, store_model

parser = argparse.ArgumentParser()
parser.add_argument('--type', help='student/teacher')
parser.add_argument('--param_file', default='params.json',
                    help='json file storing params in the default file format')

args = parser.parse_args()

params = json.load(open(args.param_file))
data = params["data"]
net_type = args.type

device = params["device"]
device = torch.device(
    'cuda:0') if device == 'cuda' else torch.device('cpu')
num_epochs = params[net_type]["num_epochs"]


def train(net, train_iter, loss, optimizer):
    if is_pretrained_present(params, net_type):
        print('using pretrained')
        net = load_model(params, net_type)
        return net

    average_loss = RunningAverage()

    print("\nstarting training")

    for _ in range(num_epochs):
        net.train()
        with tqdm(total=len(train_iter)) as t:
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)
                preds = net(X)
                l = loss(preds, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                average_loss.update(l.item())
                t.set_postfix(loss=f'{average_loss():.3f}')
                t.update()

        # test
        print('testing:')
        test(net, test_iter)

    store_model(net, params, net_type)

    return net


def test(net, test_iter):
    accuracy = calculate_accuracy(net, test_iter, device)
    print(f"test accuracy = {accuracy}")


if __name__ == "__main__":
    # defining dataloaders
    train_iter, test_iter, num_classes = get_dataloaders(data)(params[net_type]["batch_size"])

    # get network based on model
    net = get_net(params[net_type]["model"])(num_classes)
    net.to(device)

    # define loss
    loss = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=params[net_type]["lr"])

    # train
    net = train(net, train_iter, loss, optimizer)
