# LISArD: Learning Image Similarity to Defend Against Gray-box Adversarial Attacks

*Official Pytorch implementation of the [LISArD: Learning Image Similarity to Defend Against Gray-box Adversarial Attacks](https://www.arxiv.org/abs/2502.20562)*

Learning Image Similarity Adversa*r*ial Defense (LISArD) relates the similarity between clean and perturbed images by calculating the cross-correlation matrix between the embeddings of these images and using the loss to approximate this matrix to the identity while teaching the model to classify objects correctly. The goal of this approach is to reduce the effect of perturbations, motivating the model to recognize the clean and perturbed images as similar.

![main_image](images/main_image.png)
*Types of approaches commonly used to defend against adversarial attacks. The Teacher Model refers to a previously trained model, usually bigger than the Student Model, that aids the latter by providing soft labels. The DDPM refers to a Denoising Diffusion Probabilistic Model (a generative model) that uses noise and denoise to produce a "purified" image.*

### Gray-box Settings on CIFAR10

Comparison of different training methods on **gray-box settings** on CIFAR-10. S, I, and L refer to ResNet trained from scratch, with ImageNet pretraining, and LISArD, respectively.

| Model              | Clean     | FGSM      | PGD       | AA        |
|:-------------------|:---------:|:---------:|:---------:|:---------:|
| ResNet<sub>S</sub> | 87.88     | 53.53     | 43.34     | 46.56     |
| ResNet<sub>I</sub> | **94.43** | 38.21     | 3.25      | 7.13      |
| ResNet<sub>L</sub> | 87.22     | **83.14** | **83.54** | **84.19** |
