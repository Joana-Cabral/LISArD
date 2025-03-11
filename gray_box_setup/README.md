# LISArD Gray-box Setup

The `gen_clean_dataset.sh` saves the images from the PyTorch datasets CIFAR-10 and CIFAR-100.

### Train the Target Network

The `train_networks.py` trains the baseline networks used to generate the attacked datasets.

To train the network, execute the command:
```bash
python3 train_networks.py --data-root-path $DATA_ROOT_PATH --network $NET --dataset-name $DATASET --gpu-device '0'
```
- `DATA_ROOT_PATH` is the root folder containing the datasets;
- `NET` is the architecture type;
- `DATASET` is the name of the dataset (CIFAR10, CIFAR100, or TinyImageNet).

To train in Tiny ImageNet, the following structure is expected in Tiny ImageNet folder:
```bash
|-- tiny-imagenet-200
|   |-- train
|   |   |-- 0.JPEG
|   |   |-- 1.JPEG
|   |   |-- ...
|   |   |-- train_labels.csv
|   |-- val
|   |   |-- val_0.JPEG
|   |   |-- val_1.JPEG
|   |   |-- ...
|   |   |-- val_labels.csv
|
```

### Generate Attacked Datasets

The `gen_multiple_attacks.sh` generates the attack datasets needed to perform the gray-box evaluation.

1. Previously train a model according to Train the Target Network;
2. To generate the attacked datasets, execute the command:
```bash
python3 gen_attacked_dataset.py $MODEL --network $NET --set $SET --dataset $DATASET --gpu-device '0'
```
- `MODEL` is the path to the previously trained model
- `NET` is the architecture type;
- `SET` is the set (train or test);
- `DATASET` is the name of the dataset (CIFAR10, CIFAR100, or TinyImageNet).
