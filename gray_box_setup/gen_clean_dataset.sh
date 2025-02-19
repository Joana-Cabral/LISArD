DATASET='CIFAR100'
DEVICE=0

echo "#### Test ####"
# Test
python3 gen_clean_dataset.py --set test --dataset $DATASET --gpu-device $DEVICE
echo "#### Train ####"
# Train
python3 gen_clean_dataset.py --set train --dataset $DATASET --gpu-device $DEVICE
