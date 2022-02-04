import torch


def test(net, test_loader):
    net.eval()
    correct_preds = 0.
    total_images = 0.
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            features, preds = net(images)

        preds = torch.max(preds.data, 1)[1]
        total_images += labels.size(0)
        correct_preds += (preds == labels).sum().item()

    test_acc = correct_preds / total_images
    net.train()
    return test_acc
