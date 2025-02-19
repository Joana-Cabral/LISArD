
from pathlib import Path
import argparse
import json
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import torchattacks
from tqdm import tqdm
from networks.resnet import ResNet18, ResNet50, ResNet101
from tinyimagenet_loader import TinyImageNetDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def adjust_layer_name(state_dict):
    new_state_dict =  {}

    for name, tensor in state_dict.items():
        if 'backbone' in name:
            name = name[9:]

        if 'shortcut' in name:
            name = name.replace('shortcut', 'downsample')

        new_state_dict[name] = tensor

    return new_state_dict


parser = argparse.ArgumentParser(description='Train using Adversarial Training after Barlow Twins')
parser.add_argument('pretrained', type=Path, metavar='FILE', help='path to pretrained model')
parser.add_argument('num_classes', type=int, help='Number of classes in the Dataset')
parser.add_argument('--dataset-name', default='CIFAR10', type=str, help='Name of the Dataset')
parser.add_argument('--data-root-path', default='datasets/', type=Path, metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--checkpoint-dir', default='exps_CIFAR10/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--network', default='ResNet18', type=str, metavar='MLP', help='Network architecture (ResNet18 and others)')
parser.add_argument('--gpu-device', default='cuda', type=str, help='The GPU Device: cuda, and integers')

args = parser.parse_args()
args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv))
print(' '.join(sys.argv), file=stats_file)

num_classes = args.num_classes
pretrained_path = args.pretrained

if args.gpu_device == 'cuda':
    gpu = 'cuda'
elif args.gpu_device == 'cpu':
    print('Device not allowed')
    assert False
else: 
    gpu = int(args.gpu_device)

data_root_path = args.data_root_path
batch_size = args.batch_size
epochs = args.epochs

train_transform = transforms.Compose(
[
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose(
[
    transforms.ToTensor(),
])

torch.backends.cudnn.benchmark = True # Find the optimal algorithm leading to faster runtime

match args.dataset_name:
    case 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=data_root_path, train=True, download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root=data_root_path, train=False, download=False, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.workers)

    case 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=data_root_path, train=True, download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(root=data_root_path, train=False, download=False, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.workers)

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

    case _:
        print('No supported dataset')
        assert False

match args.network:
    case 'ResNet18':
        model = ResNet18(num_classes)
    case 'ResNet50':
        model = ResNet50(num_classes)
    case 'ResNet101':
        model = ResNet101(num_classes)
    case _:
        print('No supported network')
        assert False

state_dict = torch.load(pretrained_path, map_location='cpu')
new_state_dict = adjust_layer_name(state_dict)
model.load_state_dict(new_state_dict, strict=False)
model.to(gpu)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

PGD_atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)

best_test_accuracy = 0
best_adv_test_accuracy = 0

for epoch in tqdm(range(0, epochs)):

    train_total = 0
    train_correct = 0
    test_total = 0
    test_correct = 0
    PGD_train_total = 0
    PGD_train_correct = 0
    PGD_test_total = 0
    PGD_test_correct = 0
    model.train()

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        imgs = inputs.to(gpu)
        lbls = labels.to(gpu)

        # PGD
        PGD_imgs = PGD_atk(imgs, lbls).to(gpu)

        optimizer.zero_grad()

        PGD_outputs = model(PGD_imgs)
        loss = criterion(PGD_outputs, lbls)

        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        train_total += lbls.size(0)
        train_correct += (predicted == lbls).sum().item()

        _, predicted = torch.max(PGD_outputs, 1)
        PGD_train_total += lbls.size(0)
        PGD_train_correct += (predicted == lbls).sum().item()

        loss.backward()
        optimizer.step()

    train_accuracy = 100.0 * train_correct / train_total
    print('Train Clean Accuracy', train_accuracy) 

    adv_train_accuracy = 100.0 * PGD_train_correct / PGD_train_total
    print('Train PGD Accuracy', adv_train_accuracy)

    model.eval()

    for i, data in enumerate(tqdm(testloader), 0):
        inputs, labels = data

        imgs = inputs.to(gpu)
        lbls = labels.to(gpu)
        outputs = model(imgs)

        _, predicted = torch.max(outputs, 1)
        test_total += lbls.size(0)
        test_correct += (predicted == lbls).sum().item()
        
        # PGD
        PGD_imgs = PGD_atk(imgs, lbls).to(gpu)
        PGD_outputs = model(PGD_imgs)

        _, predicted = torch.max(PGD_outputs, 1)
        PGD_test_total += lbls.size(0)
        PGD_test_correct += (predicted == lbls).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print('Test Clean Accuracy', test_accuracy)
    
    adv_test_accuracy = 100.0 * PGD_test_correct / PGD_test_total
    print('Test PGD Accuracy', adv_test_accuracy)

    scheduler.step(adv_test_accuracy)

    stats = dict(epoch=epoch, loss=loss.item(), test_accuracy=test_accuracy, PGD_test_accuracy=adv_test_accuracy)
    print(json.dumps(stats))
    print(json.dumps(stats), file=stats_file)

    if test_accuracy > best_test_accuracy:
        # save checkpoint
        state = model.state_dict()
        torch.save(state, os.path.join(args.checkpoint_dir, 'checkpoint_best.pth'))
        best_test_accuracy = test_accuracy

    if adv_test_accuracy > best_adv_test_accuracy:
        state = model.state_dict()
        torch.save(state, os.path.join(args.checkpoint_dir, 'checkpoint_best_adv.pth'))
        best_adv_test_accuracy = adv_test_accuracy

