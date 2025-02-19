import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import torch.nn as nn

from orig_atk_dataset_loader import ImageNameDataset
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
parser.add_argument('dataset', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('pretrained', type=str, metavar='FILE', help='path to pretrained model')
parser.add_argument('num_classes', type=int, help='path to pretrained model')
parser.add_argument('--network', default='ResNet18', type=str, metavar='N', help='Model Type')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--gpu-device', default='cuda', type=str, help='The GPU Device: cuda, and integers')

args = parser.parse_args()
num_classes = args.num_classes
pretrained_path = args.pretrained
dataset_path = args.dataset

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

csv_path = os.path.join(dataset_path, 'labels_test.csv')

test_set = ImageNameDataset(csv_file=csv_path, root_dir=dataset_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=8)

state_dict = torch.load(pretrained_path, map_location='cpu')
new_state_dict = adjust_layer_name(args.network, state_dict)
model.load_state_dict(new_state_dict, strict=False)
model.to(gpu)

test_total = 0
test_correct = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels, _ = data

        inputs = inputs.cuda(gpu, non_blocking=True)
        labels = labels.cuda(gpu, non_blocking=True)      

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
    

test_accuracy = 100.0 * test_correct / test_total
print('Test Accuracy', test_accuracy)

