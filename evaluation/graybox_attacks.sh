
#!/bin/bash   

# MODEL='../exps_CIFAR10/best_gray_box_wo_projector/checkpoint_best.pth'
MODEL='../train_resnets/models/resnet18_CIFAR10/checkpoint_best.pth'
TYPE='ResNet18'
DEVICE=0

########### CIFAR10 ###########
NUMCLASS=10
# Evaluate on normal CIFAR10
echo "Evaluating on CIFAR10 clean dataset"
python3 attack_resnet.py ../datasets/CIFAR10_w_images/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on FGSM CIFAR10
echo "Evaluating on CIFAR10 with FGSM attack dataset"
python3 attack_resnet.py ../datasets/CIFAR10/CIFAR10_FGSM_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on PGD CIFAR10
echo "Evaluating on CIFAR10 with PGD attack dataset"
python3 attack_resnet.py ../datasets/CIFAR10/CIFAR10_PGD_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on AutoAttack CIFAR10
echo "Evaluating on CIFAR10 with AutoAttack attack dataset"
python3 attack_resnet.py ../datasets/CIFAR10/CIFAR10_AutoAttack_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# MODEL='../exps_CIFAR100/ResNet18/checkpoint_best.pth'
MODEL='../train_resnets/models/resnet18_CIFAR100/checkpoint_best.pth'
TYPE='ResNet18'
DEVICE=0

########### CIFAR100 ###########
NUMCLASS=100
# Evaluate on normal CIFAR100
echo "Evaluating on CIFAR100 clean dataset"
python3 attack_resnet.py ../datasets/CIFAR100_w_images/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on FGSM CIFAR100
echo "Evaluating on CIFAR100 with FGSM attack dataset"
python3 attack_resnet.py ../datasets/CIFAR100/CIFAR100_FGSM_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on PGD CIFAR100
echo "Evaluating on CIFAR100 with PGD attack dataset"
python3 attack_resnet.py ../datasets/CIFAR100/CIFAR100_PGD_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on AutoAttack CIFAR100s
echo "Evaluating on CIFAR100 with AutoAttack attack dataset"
python3 attack_resnet.py ../datasets/CIFAR100/CIFAR100_AutoAttack_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# MODEL='../exps_TinyImageNet/ResNet18/checkpoint_best.pth'
MODEL='../train_resnets/models/resnet18_TinyImageNet/checkpoint_best.pth'
TYPE='ResNet18'
DEVICE=0

########### Tiny ImageNet ###########
NUMCLASS=200
# Evaluate on normal TinyImageNet
echo "Evaluating on TinyImageNet clean dataset"
python3 attack_resnet.py ../datasets/tiny-imagenet-200/val/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on FGSM TinyImageNet
echo "Evaluating on TinyImageNet with FGSM attack dataset"
python3 attack_resnet.py ../datasets/TinyImageNet/TinyImageNet_FGSM_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on PGD TinyImageNet
echo "Evaluating on TinyImageNet with PGD attack dataset"
python3 attack_resnet.py ../datasets/TinyImageNet/TinyImageNet_PGD_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE

# Evaluate on AutoAttack TinyImageNet
echo "Evaluating on TinyImageNet with AutoAttack attack dataset"
python3 attack_resnet.py ../datasets/TinyImageNet/TinyImageNet_AutoAttack_$TYPE/test/ $MODEL $NUMCLASS --network $TYPE --gpu-device $DEVICE