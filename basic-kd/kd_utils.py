import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")


def get_teacher():
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10))

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    return net


def get_student():
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 1024), nn.ReLU(),
                        nn.Linear(1024, 1024), nn.ReLU(),
                        nn.Linear(1024, 256), nn.ReLU(),
                        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    return net


def calculate_accuracy(net, test_iter, device):
    num_test_examples = 0
    num_correct_preds = 0

    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            preds = net(X)
            preds = preds.argmax(axis=1)
            comp = preds.type(y.dtype) == y
            num_correct_preds += float(comp.type(y.dtype).sum())
            num_test_examples += y.numel()

    return num_correct_preds / num_test_examples


def train(net, train_iter, test_iter, loss, optimizer, params, is_student=False):
    if is_student:
        lr = params["STUDENT_LR"]
        num_epochs = params["STUDENT_NUM_EPOCHS"]
        save_as = load_as = f"student-ne{num_epochs}-lr{lr}.pt"
        force_retrain = params["STUDENT_FORCE_RETRAIN"]
    else:
        lr = params["TEACHER_LR"]
        num_epochs = params["TEACHER_NUM_EPOCHS"]
        save_as = load_as = f"teacher-ne{num_epochs}-lr{lr}.pt"
        force_retrain = params["TEACHER_FORCE_RETRAIN"]
    accuracies_file = params["RESULTS"]
    device = params["DEVICE"]

    if not force_retrain:
        # if model is stored as .pt file, load it and display accuracy on test set
        from pathlib import Path
        net_weights = Path(load_as)
        if net_weights.is_file():
            print("loading pre-existing weights")
            net = torch.load(net_weights).to(device)
            print("loaded weights successfully")
            accuracy = parse_pretrained_accuracies(open(accuracies_file))[save_as]
            print(f"test accuracy = {accuracy}")
            return net

    net.to(device)
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            preds = net(X)
            l = loss(preds, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        accuracy = calculate_accuracy(net, test_iter, device)
        print(f"test accuracy at epoch {epoch + 1} = {accuracy}")

    torch.save(net, save_as)
    with open(accuracies_file, "a") as file:
        file.write(f"\n{save_as} {accuracy}")

    return net


def get_dataloaders(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    mnist_train_dataloader = data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=4)
    mnist_test_dataloader = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)

    return mnist_train_dataloader, mnist_test_dataloader


def parse_pretrained_accuracies(file):
    accuracies = {model.split(" ")[0]: float(model.split(" ")[1]) for model in file.read().split("\n")}
    return accuracies
