import torch
import torchvision
import torchvision.transforms as transforms

import os
from pathlib import Path
import argparse
import sys

import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from wideresnet import WideResNet
from tiny_imagenet_dataloader import TinyImageNetDataset

parser = argparse.ArgumentParser(description='LISAD CIFAR10 Training')
parser.add_argument('--data-root-path', default='../datasets/', type=Path, metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--checkpoint-dir', default='models/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--network', default='ResNet18', type=str, help='Network architecture (ResNet18 and others)')
parser.add_argument('--dataset-name', default='CIFAR10', type=str, help='Dataset Name')
parser.add_argument('--gpu-device', default='cuda', type=str, help='The GPU Device: cuda, and integers')


test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

args = parser.parse_args()
args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv))
print(' '.join(sys.argv), file=stats_file)

epochs = args.epochs
batch_size = args.batch_size
data_root_path = args.data_root_path

if args.gpu_device == 'cuda':
    gpu = 'cuda'
elif args.gpu_device == 'cpu':
    print('Device not allowed')
    assert False
else: 
    gpu = int(args.gpu_device)

match args.dataset_name:
    case 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.workers)

        num_classes = 10
    case 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR100(root='../datasets', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.workers)

        num_classes = 100

    case 'TinyImageNet':
        data_root_path = os.path.join(data_root_path, 'tiny-imagenet-200')
        trainset_path = os.path.join(data_root_path, 'train')
        testset_path = os.path.join(data_root_path, 'val')
        train_csv_path = os.path.join(trainset_path, 'train_labels.csv')
        test_csv_path = os.path.join(testset_path, 'val_labels.csv')

        trainset = TinyImageNetDataset(csv_file=train_csv_path, root_dir=trainset_path, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        testset = TinyImageNetDataset(csv_file=test_csv_path, root_dir=testset_path, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.workers)

        num_classes = 200

match args.network:
    case 'ResNet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    case 'ResNet50':
        model = torchvision.models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    case 'ResNet101':
        model = torchvision.models.resnet101(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    case 'WideResNet28_10':
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)

    case 'VGG19':
        model = torchvision.models.vgg19(pretrained=False)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    case 'MobileNetv2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    case 'EfficientNetB2':
        model = torchvision.models.efficientnet_b2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    case _:
        print('Not Supported Network')
        assert False

model.to(gpu)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True) # EfficientNetB2

best_test_accuracy = 0
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

    train_total = 0
    train_correct = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda(gpu, non_blocking=True)
        labels = labels.cuda(gpu, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()
    
    train_accuracy = 100.0 * train_correct / train_total
    print('Train Accuracy', train_accuracy)
    print('Train Accuracy', train_accuracy, file=stats_file)

    test_total = 0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()


    test_accuracy = 100.0 * test_correct / test_total
    print('Test Accuracy', test_accuracy)
    print('Test Accuracy', test_accuracy, file=stats_file)

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        model_path = os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
        torch.save(model.state_dict(), model_path)
        print('Model saved on ', model_path, ' with test accuracy of ', best_test_accuracy,  ' %')
        print('Model saved on ', model_path, ' with test accuracy of ', best_test_accuracy,  ' %', file=stats_file)

print('Finished Training')
