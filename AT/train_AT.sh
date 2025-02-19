

MODEL='../exps_CIFAR10/ResNet50_best_gray_box/checkpoint_best.pth'
python3 main_AT.py $MODEL 10 --dataset-name 'CIFAR10' --data-root-path '../../datasets' --network ResNet50 --gpu-device '0'
