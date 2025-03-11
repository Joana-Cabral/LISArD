# LISArD Adversarial Training

1. Previously train a model according to the main folder of this GitHub;
2. To train the model, execute the command:
```bash
python3 main_AT.py $PATH_TO_MODEL $NUM_CLASS --dataset-name $DATASET --data-root-path $DATA_ROOT_PATH --network $NETWORK --gpu-device '0'
```
- `$PATH_TO_MODEL` is the path to the previously trained model;
- `$NUM_CLASS` is the number of classes in the dataset;
- `$DATASET` is the name of the dataset (CIFAR10, CIFAR100, or TinyImageNet);
- `$DATA_ROOT_PATH` is the root folder containing the datasets;
- `$NETWORK` is the architecture type.

Alternatively, you can run the script `train_AT.sh` which contains an example of training the model in CIFAR10 using a ResNet50.

### Tiny ImageNet

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
|
```

## Cite
