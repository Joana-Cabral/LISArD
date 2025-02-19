MODEL='models/resnet18_TinyImageNet/checkpoint_best.pth'
NET='ResNet18'
DATASET='TinyImageNet'
DEVICE=0

echo "#### Auto Attack Test ####"
# AutoAttack
python3 gen_attacked_dataset.py $MODEL --network $NET --set test --dataset $DATASET --gpu-device $DEVICE
echo "#### Auto Attack Train ####"
# AutoAttack
python3 gen_attacked_dataset.py $MODEL --network $NET --set train --dataset $DATASET --gpu-device $DEVICE

echo "#### PGD Test ####"
# PGD
python3 gen_attacked_dataset.py $MODEL --network $NET --attack PGD --set test --dataset $DATASET --gpu-device $DEVICE
echo "#### PGD Train ####"
# PGD
python3 gen_attacked_dataset.py $MODEL --network $NET --attack PGD --set train --dataset $DATASET --gpu-device $DEVICE

echo "#### FGSM Test ####"
# FGSM
python3 gen_attacked_dataset.py $MODEL --network $NET --attack FGSM --set test --dataset $DATASET --gpu-device $DEVICE
echo "#### FGSM Train ####"
# FGSM
python3 gen_attacked_dataset.py $MODEL --network $NET --attack FGSM --set train --dataset $DATASET --gpu-device $DEVICE
