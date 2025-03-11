# Gray-box and White-box Evaluation

The script `graybox_attacks.sh` has multiple commands to evaluate the model in graybox settings according to the FGSM, PGD, and AutoAttack attacks, and considering CIFAR-10, CIFAR-100, and Tiny ImageNet datasets.

1. Previously train a model using LISArD methodology;
2. In `graybox_attacks.sh`, change the `MODEL` to the path of the previously trained model;
3. In `graybox_attacks.sh`, change the `TYPE` to correspond to the `MODEL` architecture.

The script `whitebox_attacks.sh` has multiple commands to evaluate the model in whitebox settings for FGSM, PGD, and AutoAttack, and considering CIFAR-10, CIFAR-100, and Tiny ImageNet datasets.

1. Previously train a model using LISArD methodology;
2. In `whitebox_attacks.sh`, change the `MODEL` to the path of the previously trained model;
3. In `whitebox_attacks.sh`, change the `TYPE` to correspond to the `MODEL` architecture.

### Cite

