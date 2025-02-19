
import argparse
from tqdm import tqdm
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

import pandas as pd

parser = argparse.ArgumentParser(description='Generate Attacked Datasets')
parser.add_argument('--save_folder', default='datasets/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='Dataset Name: CIFAR10 (default), CIFAR100')
parser.add_argument('--set', default='test', type=str, help='Train, Test (default) or Both Sets')
parser.add_argument('--gpu-device', default='cuda', type=str, help='The GPU Device: cuda, and integers')

if __name__ == '__main__':
    args   = parser.parse_args()
    save_folder = os.path.join(args.save_folder, '{}_w_images'.format(args.dataset))
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

    batch_size = 1
    transform = transforms.Compose([transforms.ToTensor()])
    transform_to_image = transforms.ToPILImage()


    if args.dataset == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root='../datasets', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        test_dataset = torchvision.datasets.CIFAR100(root='../datasets', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        n_classes = 100

    else:
        train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        test_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8) 

        n_classes = 10


    if args.set in ['train', 'both']:
        csv_save_path = os.path.join(train_save_folder, 'labels_train.csv')
        labels_dict = {
            'name': [],
            'label': []
        }
        img_count = 0

        for step, (images, labels) in enumerate(tqdm(train_loader)):
            images = images[0, :, :, :]
            img = transform_to_image(images)
            img_name = '{}.png'.format(img_count)
            save_path = os.path.join(train_save_folder, img_name)
            img.save(save_path)

            labels_dict['name'].append(img_name)
            labels_dict['label'].append(labels.item())

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
            images = images[0, :, :, :]
            img = transform_to_image(images)
            img_name = '{}.png'.format(img_count)
            save_path = os.path.join(test_save_folder, img_name)
            img.save(save_path)

            labels_dict['name'].append(img_name)
            labels_dict['label'].append(labels.item())

            img_count += 1


        df = pd.DataFrame(labels_dict)
        df.to_csv(csv_save_path, index=False)
