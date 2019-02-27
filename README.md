# Image-Classification-Demos
Playground for experimenting deeplearning for image classification

playground for the challenge Humpback whale identification hosted by Kaggle

Scripts do not run per se, because I did not provide pretrained models (size above 100Ms, github complained when trying to push) nor did I provide the dataset which is also hosted on Kaggle.
All examples are run in PyTorch
Some Jupyter notebooks are also provided when used for data transformation / data manipulation (statistics) in the original dataset.

Experiments used only original images trained either end-to-end or fine-tuning last linear layer of the models VGG/Inception-v3, plus an implementation of siamese network that behaves okay only if trained on fashion-MNIST, rather that humpback dataset.

For the models run on vgg/Inception, best accuracy is 70% on test dataset (small variation if using vgg / inception) and no learning taking place if using siamese net.

*En echange*, siamese net achieves 40% accuracy on fashionMNIST, trained with only 100 images (evenly distributed across all 10 classes) and tested for the rest 59900
