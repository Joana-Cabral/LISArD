
from pathlib import Path
import argparse
import json
import math
import os
import sys

from torch import nn, optim
import torch
import torchvision.transforms as transforms

from tqdm import tqdm

from networks.resnet import ResNet18
from tinyimagenet_loader import TinyImageNetDataset

parser = argparse.ArgumentParser(description='LISAD Tiny ImageNet Training')
parser.add_argument('--data-root-path', default='../datasets/tiny-imagenet-200', type=Path, metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=1024, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.02, type=float, metavar='LR', help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.00048, type=float, metavar='LR', help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L', help='weight on off-diagonal terms')
parser.add_argument('--projector', default='512-512-512', type=str, metavar='MLP', help='projector MLP')
parser.add_argument('--checkpoint-dir', default='exps_TinyImageNet/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--network', default='ResNet18', type=str, metavar='MLP', help='Network architecture (ResNet18 and others)')
parser.add_argument('--gpu-device', default='cuda', type=str, help='The GPU Device: cuda, and integers')


def add_random_gaussian_noise(x, var, device):
    gauss_x = x + (var**0.5)*torch.randn(x.shape[0], 3, 64, 64).cuda(device)
    return gauss_x



def main():
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    if args.gpu_device == 'cuda':
        gpu = 'cuda'
    elif args.gpu_device == 'cpu':
        print('Device not allowed')
        return
    else: 
        gpu = int(args.gpu_device)

    data_root_path = args.data_root_path
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    torch.backends.cudnn.benchmark = True # Find the optimal algorithm leading to faster runtime

    model = BarlowTwins(args).cuda(gpu)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint_best.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint_best.pth', map_location='cpu')
        epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])
        print("Checkpoint resumed at", epoch, " epoch")
        print("Checkpoint resumed at", epoch, " epoch", file=stats_file)

    trainset_path = os.path.join(data_root_path, 'train')
    testset_path = os.path.join(data_root_path, 'val')
    train_csv_path = os.path.join(trainset_path, 'train_labels.csv')
    test_csv_path = os.path.join(testset_path, 'val_labels.csv')

    trainset = TinyImageNetDataset(csv_file=train_csv_path, root_dir=trainset_path, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = TinyImageNetDataset(csv_file=test_csv_path, root_dir=testset_path, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    scaler = torch.cuda.amp.GradScaler()
    best_test_accuracy = 0
    for epoch in tqdm(range(0, args.epochs)):
        test_total = 0
        test_correct = 0

        model.train()
        for step, (y1, lbl) in enumerate(trainloader, start=epoch * len(trainloader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            lbl = lbl.cuda(gpu, non_blocking=True)

            y2 = add_random_gaussian_noise(y1, 8/255, gpu)
            y2 = y2.cuda(gpu, non_blocking=True)

            adjust_learning_rate(args, optimizer, trainloader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                alpha, loss, _ = model.forward(y1, y2, lbl, epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        for imgs, lbls in testloader:
            imgs = imgs.cuda(gpu, non_blocking=True)
            lbls = lbls.cuda(gpu, non_blocking=True)

            _, _, logits = model.forward(imgs, imgs, lbls, epoch)
            _, predicted = torch.max(logits, 1)
            test_total += lbls.size(0)
            test_correct += (predicted == lbls).sum().item()
        
        test_accuracy = 100.0 * test_correct / test_total
                    
        stats = dict(epoch=epoch, lr_weights=optimizer.param_groups[0]['lr'], lr_biases=optimizer.param_groups[1]['lr'], loss=loss.item(), alpha=alpha, test_accuracy=test_accuracy)
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

        if test_accuracy > best_test_accuracy:
            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, args.checkpoint_dir / 'checkpoint_best.pth')
            best_test_accuracy = test_accuracy
            print("The model has been saved on", args.checkpoint_dir / 'checkpoint_best.pth', " with a test accuracy of ", best_test_accuracy, " %")
            print("The model has been saved on", args.checkpoint_dir / 'checkpoint_best.pth', " with a test accuracy of ", best_test_accuracy, " %", file=stats_file)

    torch.save({'model_state_dict': model.state_dict()}, args.checkpoint_dir / 'checkpoint_final.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Custom architectures are used in order to manipulate their forward function
        match args.network:
            case 'ResNet18':
                self.backbone = ResNet18(200)
                sizes = [512] + list(map(int, args.projector.split('-')))
            case _:
                print('That network does not match')
                assert False

        self.criterion = nn.CrossEntropyLoss()
        self.alpha_0 = 0.5
        self.temp = 9000

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2, labels, epochs):
        out1, logits1 = self.backbone(y1)
        out2, logits2 = self.backbone(y2)

        z1 = out1
        z2 = out2

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss_BT = on_diag + self.args.lambd * off_diag

        loss_class1 = self.criterion(logits1, labels)
        loss_class2 = self.criterion(logits2, labels)

        alpha = self.alpha_0 + (1 / 400) * (epochs - 1)

        loss = alpha * (loss_class1 + loss_class2) + (1 - alpha) * (loss_BT/self.temp)

        return alpha, loss, logits1


if __name__ == '__main__':
    main()
