
#!/bin/bash   


# MODEL='../exps_CIFAR10/Resnet18/checkpoint_best.pth'
MODEL='../gray_box_setup/models/resnet18_CIFAR10/checkpoint_best.pth'
TYPE='ResNet18'
NUMCLASS=10
DEVICE=0

echo "Evaluating on CIFAR10 white-box settings"
python3 attack_resnet_whitebox.py 'CIFAR10' $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# MODEL='../exps_CIFAR100/Resnet18/checkpoint_best.pth'
MODEL='../gray_box_setup/models/resnet18_CIFAR100/checkpoint_best.pth'
TYPE='ResNet18'
NUMCLASS=100
DEVICE=0

echo "Evaluating on CIFAR100 white-box settings"
python3 attack_resnet_whitebox.py 'CIFAR100' $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# MODEL='../exps_TinyImageNet/Resnet18/checkpoint_best.pth'
MODEL='../gray_box_setup/models/resnet18_TinyImageNet/checkpoint_best.pth'
TYPE='ResNet18'
NUMCLASS=200
DEVICE=0

echo "Evaluating on TinyImageNet white-box settings"
python3 attack_resnet_whitebox.py 'TinyImageNet' $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE