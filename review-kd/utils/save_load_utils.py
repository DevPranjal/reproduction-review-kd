import json
import torch


def generate_name(params, net_type):
    for nt, param in params.items():
        if nt == net_type:
            name = ""
            for value in param.values():
                name += f"{value}-"
            name += ".pt"
    return name


def store_model(net, params, net_type):
    name = generate_name(params, net_type)
    location = f"/home/pranjal/Projects/ml-repro-2021/review-kd/pretrained/{net_type}/{name}"
    print(f"saving model to {location}")
    torch.save(net, f"{location}")


def generate_names(params):
    names = []
    for nt in ["teacher", "student", "student_review_kd"]:
        names.append(generate_name(params, nt))
    return names


def is_pretrained_present(params, net_type):
    name = generate_name(params, net_type)
    import os
    os.chdir(f"/home/pranjal/Projects/ml-repro-2021/review-kd/pretrained/{net_type}")
    if name in os.listdir():
        return True
    return False


def load_model(params, net_type):
    name = generate_name(params, net_type)
    location = f"/home/pranjal/Projects/ml-repro-2021/review-kd/pretrained/{net_type}/{name}"
    print(f"loading {net_type} model from {location}")
    net = torch.load(f"{location}")
    return net


if __name__ == "__main__":
    params = json.load(open("../params.json"))
    print(is_pretrained_present(params, "student"))
