from kd_utils import get_student, get_teacher, get_dataloaders, train, calculate_accuracy, parse_pretrained_accuracies
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")


params = {
    "TEACHER_BATCH_SIZE": 128,
    "TEACHER_LR": 0.1,
    "TEACHER_NUM_EPOCHS": 10,
    "TEACHER_FORCE_RETRAIN": False,

    "STUDENT_BATCH_SIZE": 128,
    "STUDENT_LR": 0.3,
    "STUDENT_NUM_EPOCHS": 20,
    "STUDENT_FORCE_RETRAIN": False,

    "DISTILL_BATCH_SIZE": 128,
    "DISTILL_LR": 0.3,
    "DISTILL_NUM_EPOCHS": 20,
    "DISTILL_FORCE_RETRAIN": False,

    "DEVICE": torch.device("cuda:0"),

    "LAMBDA": 0.7,
    "T": 10,

    "RESULTS": "kd_pretrained_accuracies.txt"
}


def KD_loss(outputs, labels, teacher_outputs, params):
    lmbda = params["LAMBDA"]
    T = params["T"]
    return nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                          F.softmax(teacher_outputs/T, dim=1)) * (lmbda * T * T) + \
        F.cross_entropy(outputs, labels) * (1. - lmbda)


def KD_train(student, teacher, train_iter, test_iter, loss, optimizer, params):
    lr = params["DISTILL_LR"]
    device = params["DEVICE"]
    num_epochs = params["DISTILL_NUM_EPOCHS"]
    lmbda = params["LAMBDA"]
    T = params["T"]
    save_as = load_as = f"distilled-ne{num_epochs}-lr{lr}-lmbda{lmbda}-t{T}.pt"
    accuracies_file = params["RESULTS"]
    force_retrain = params["DISTILL_FORCE_RETRAIN"]

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

    student.to(device)
    teacher.to(device)

    for epoch in range(num_epochs):
        student.train()
        teacher.eval()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            student_preds = student(X)
            with torch.no_grad():
                teacher_preds = teacher(transforms.Resize(224)(X))
            l = loss(student_preds, y, teacher_preds, params)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        student.eval()
        accuracy = calculate_accuracy(student, test_iter, device)
        print(f"test accuracy at epoch {epoch + 1} = {accuracy}")

    torch.save(student, save_as)
    with open(accuracies_file, "a") as file:
        file.write(f"\n{save_as} {accuracy}")

    return student


if __name__ == "__main__":

    teacher = get_teacher()
    print("\nTraining teacher network:")
    train_iter, test_iter = get_dataloaders(
        batch_size=params["TEACHER_BATCH_SIZE"], resize=224)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(teacher.parameters(), lr=params["TEACHER_LR"])
    teacher = train(teacher, train_iter, test_iter, loss,
                    trainer, params, is_student=False)

    student = get_student()
    print("\nTraining student network (baseline):")
    train_iter, test_iter = get_dataloaders(
        batch_size=params["STUDENT_BATCH_SIZE"])
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(student.parameters(), lr=params["STUDENT_LR"])
    student = train(student, train_iter, test_iter, loss,
                    trainer, params, is_student=True)

    student_kd = get_student()
    print("\nTraining student network (distillation):")
    train_iter, test_iter = get_dataloaders(
        batch_size=params["DISTILL_BATCH_SIZE"])
    loss = KD_loss
    trainer = torch.optim.SGD(student_kd.parameters(), lr=params["DISTILL_LR"])
    student_distilled = KD_train(
        student_kd, teacher, train_iter, test_iter, loss, trainer, params)
