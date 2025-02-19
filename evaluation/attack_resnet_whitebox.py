import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import torch.nn as nn
import torchattacks
from tqdm import tqdm
from pathlib import Path

from orig_atk_dataset_loader import TinyImageNetDataloader
from networks.mobilenetv2 import MobileNetV2 
from networks.wideresnet import WideResNet
from networks.vgg import VGG
from networks.efficientnet import EfficientNetB2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def adjust_layer_name(model, state_dict):
    new_state_dict =  {}

    if 'ResNet' in model:

        for name, tensor in state_dict.items():
            if 'backbone' in name:
                name = name[9:]

            if 'shortcut' in name:
                name = name.replace('shortcut', 'downsample')

            if 'linear' in name:
                name = name.replace('linear', 'fc')

            new_state_dict[name] = tensor

    else:

        for name, tensor in state_dict.items():
            if 'backbone' in name:
                name = name[9:]

            new_state_dict[name] = tensor

    return new_state_dict

parser = argparse.ArgumentParser(description='Attack ResNets using the different datasets')
parser.add_argument('dataset', type=str, help='Dataset Name')
parser.add_argument('pretrained', type=str, metavar='FILE', help='path to pretrained model')
parser.add_argument('num_classes', type=int, help='path to pretrained model')
parser.add_argument('--data-root-path', default='../datasets/', type=Path, metavar='DIR', help='path to dataset')
parser.add_argument('--network', default='ResNet18', type=str, metavar='N', help='Model Type')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--gpu-device', default='cuda', type=str, help='The GPU Device: cuda, and integers')

args = parser.parse_args()
num_classes = args.num_classes
pretrained_path = args.pretrained
batch_size = args.batch_size
data_root_path = args.data_root_path

if args.gpu_device == 'cuda':
    gpu = 'cuda'
elif args.gpu_device == 'cpu':
    print('Device not allowed')
    assert False
else: 
    gpu = int(args.gpu_device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

match args.network:
    case 'ResNet18':
        model = torchvision.models.resnet18()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    case 'ResNet50':
        model = torchvision.models.resnet50()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    case 'ResNet101':
        model = torchvision.models.resnet101()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    case 'WideResNet28_10':
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
    
    case 'MobileNetv2':
        model = MobileNetV2(10)

    case 'VGG19':
        model = VGG('VGG19')

    case 'EfficientNetB2':
        model = EfficientNetB2(10)

    case _:
        print('Not Supported Network')
        assert False


match args.dataset:
    case 'CIFAR10':
        testset = torchvision.datasets.CIFAR10(root=data_root_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    case 'CIFAR100':
        testset = torchvision.datasets.CIFAR100(root=data_root_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    case 'TinyImageNet':
        data_root_path = os.path.join(data_root_path, 'tiny-imagenet-200')
        testset_path = os.path.join(data_root_path, 'val')
        test_csv_path = os.path.join(testset_path, 'labels_test.csv')

        test_set = TinyImageNetDataloader(csv_file=test_csv_path, root_dir=testset_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

state_dict = torch.load(pretrained_path, map_location='cpu')
new_state_dict = adjust_layer_name(args.network, state_dict)
model.load_state_dict(new_state_dict, strict=False)
model.to(gpu)

PGD_atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
FGSM_atk = torchattacks.FGSM(model, eps=8/255)
AA_atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=num_classes, seed=None, verbose=False)

test_total = 0
test_correct = 0
PGD_total = 0
PGD_correct = 0
FGSM_total = 0
FGSM_correct = 0
AA_total = 0
AA_correct = 0
model.eval()

for i, data in enumerate(tqdm(test_loader), 0):
    inputs, labels = data

    imgs = inputs.cuda(gpu, non_blocking=True)
    lbls = labels.cuda(gpu, non_blocking=True)
    outputs = model(imgs)

    _, predicted = torch.max(outputs, 1)
    test_total += lbls.size(0)
    test_correct += (predicted == lbls).sum().item()

test_accuracy = 100.0 * test_correct / test_total
print('Clean Accuracy', test_accuracy)


for i, data in enumerate(tqdm(test_loader), 0):
    inputs, labels = data

    imgs = inputs.cuda(gpu, non_blocking=True)
    lbls = labels.cuda(gpu, non_blocking=True)

    # FGSM
    FGSM_imgs = FGSM_atk(imgs, lbls).cuda(gpu, non_blocking=True)
    FGSM_outputs = model(FGSM_imgs)

    _, predicted = torch.max(FGSM_outputs, 1)
    FGSM_total += lbls.size(0)
    FGSM_correct += (predicted == lbls).sum().item()

FGSM_accuracy = 100.0 * FGSM_correct / FGSM_total
print('FGSM Accuracy', FGSM_accuracy)

for i, data in enumerate(tqdm(test_loader), 0):
    inputs, labels = data

    imgs = inputs.cuda(gpu, non_blocking=True)
    lbls = labels.cuda(gpu, non_blocking=True)

    # PGD
    PGD_imgs = PGD_atk(imgs, lbls).cuda(gpu, non_blocking=True)
    PGD_outputs = model(PGD_imgs)

    _, predicted = torch.max(PGD_outputs, 1)
    PGD_total += lbls.size(0)
    PGD_correct += (predicted == lbls).sum().item()

PGD_accuracy = 100.0 * PGD_correct / PGD_total
print('PGD Accuracy', PGD_accuracy)

for i, data in enumerate(tqdm(test_loader), 0):
    inputs, labels = data

    imgs = inputs.cuda(gpu, non_blocking=True)
    lbls = labels.cuda(gpu, non_blocking=True)

    # AA
    AA_imgs = AA_atk(imgs, lbls).cuda(gpu, non_blocking=True)
    AA_outputs = model(AA_imgs)
    
    _, predicted = torch.max(AA_outputs, 1)
    AA_total += lbls.size(0)
    AA_correct += (predicted == lbls).sum().item()

AA_accuracy = 100.0 * AA_correct / AA_total
print('AA Accuracy', AA_accuracy)
