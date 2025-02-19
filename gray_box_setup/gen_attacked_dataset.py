
import argparse
from tqdm import tqdm
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchattacks

import pandas as pd

from tiny_imagenet_dataloader import TinyImageNetDataset
from wideresnet import WideResNet

parser = argparse.ArgumentParser(description='Generate Attacked Datasets')
parser.add_argument('load', type=str, help='Path to the model you want to load')
parser.add_argument('--data-root-path', default='../datasets/', type=Path, metavar='DIR', help='path to dataset')
parser.add_argument('--save_folder', default='datasets/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset Name: CIFAR10 (default), CIFAR100')
parser.add_argument('--attack', default='AutoAttack', type=str, help='Attack Name: AutoAttack (default), PGD')
parser.add_argument('--network', default='ResNet50', type=str, help='Model to Attack: ResNet50 (default), ResNet18, VGG19')
parser.add_argument('--set', default='test', type=str, help='Train, Test (default) or Both Sets')
parser.add_argument('--gpu-device', default='cuda', type=str, help='The GPU Device: cuda, and integers')


if __name__ == '__main__':
    args   = parser.parse_args()
    save_folder = os.path.join(args.save_folder, '{}_{}_{}'.format(args.dataset, args.attack, args.network))
    os.makedirs(save_folder, exist_ok=True)
    test_save_folder = os.path.join(save_folder, 'test')
    os.makedirs(test_save_folder, exist_ok=True)
    train_save_folder = os.path.join(save_folder, 'train')
    os.makedirs(train_save_folder, exist_ok=True)

    if args.gpu_device == 'cuda':
        gpu = 'cuda'
    elif args.gpu_device == 'cpu':
        print('Device not allowed')
        assert False
    else: 
        gpu = int(args.gpu_device)

    batch_size = 128

    transform = transforms.Compose([transforms.ToTensor()])
    transform_to_image = transforms.ToPILImage()

    data_root_path = args.data_root_path
    match args.dataset:
        case 'CIFAR100':
            train_dataset = torchvision.datasets.CIFAR100(root=data_root_path, train=True, download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

            test_dataset = torchvision.datasets.CIFAR100(root=data_root_path, train=False, download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            n_classes = 100

        case 'TinyImageNet':
            dataset_path = os.path.join(data_root_path, 'tiny-imagenet-200')
            trainset = TinyImageNetDataset(csv_file=os.path.join(dataset_path, 'train/train_labels.csv'), root_dir=os.path.join(dataset_path, 'train'), transform=transform)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

            testset = TinyImageNetDataset(csv_file=os.path.join(dataset_path, 'val/val_labels.csv'), root_dir=os.path.join(dataset_path, 'val'), transform=transform)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
            n_classes = 200

        case 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(root=data_root_path, train=True, download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

            test_dataset = torchvision.datasets.CIFAR10(root=data_root_path, train=False, download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8) 
            n_classes = 10

    match args.network:
        case 'ResNet18':
            model = torchvision.models.resnet18()
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, n_classes)
        case 'ResNet50':    
            model = torchvision.models.resnet50()
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, n_classes)
        case 'ResNet101':
            model = torchvision.models.resnet101()
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, n_classes)
        case 'VGG19':
            model = torchvision.models.vgg19()
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, n_classes)
        case 'EfficientNetB2':
            model = torchvision.models.efficientnet_b2()
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, n_classes)
        case 'MobileNetv2':
            model = torchvision.models.mobilenet_v2()
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, n_classes)
        case 'WideResNet28_10':
            model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        case _:
            print('Not Supported Network')
            assert False

    if args.load != "":
        print(args.load)
        model.load_state_dict(torch.load(args.load, map_location='cpu'))

    model.to(gpu)

    if args.attack == 'PGD':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
    elif args.attack == 'FGSM':
        atk = torchattacks.FGSM(model, eps=8/255)
    else:   
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=n_classes, seed=None, verbose=False)
    
    model.eval()

    if args.set in ['train', 'both']:
        csv_save_path = os.path.join(train_save_folder, 'labels_train.csv')
        labels_dict = {
            'name': [],
            'label': []
        }
        img_count = 0

        for step, (images, labels) in enumerate(tqdm(train_loader)):
            adv_images = atk(images, labels)

            for adv, lbl in zip(adv_images, labels):
                adv_img = transform_to_image(adv)
                img_name = '{}.png'.format(img_count)
                # img_name = '{}.JPEG'.format(img_count) # Tiny ImageNet
                save_path = os.path.join(train_save_folder, img_name)
                adv_img.save(save_path)

                labels_dict['name'].append(img_name)
                labels_dict['label'].append(lbl.item())

                img_count += 1

        df = pd.DataFrame(labels_dict)
        df.to_csv(csv_save_path, index=False)

    if args.set in ['test', 'both']:
        csv_save_path = os.path.join(test_save_folder, 'labels_test.csv')
        labels_dict = {
            'name': [],
            'label': []
        }
        img_count = 0

        for step, (images, labels) in enumerate(tqdm(test_loader)):
            adv_images = atk(images, labels)

            for adv, lbl in zip(adv_images, labels):
                adv_img = transform_to_image(adv)
                img_name = '{}.png'.format(img_count)
                # img_name = '{}.JPEG'.format(img_count) # Tiny ImageNet
                save_path = os.path.join(test_save_folder, img_name)
                adv_img.save(save_path)

                labels_dict['name'].append(img_name)
                labels_dict['label'].append(lbl.item())

                img_count += 1


        df = pd.DataFrame(labels_dict)
        df.to_csv(csv_save_path, index=False)
