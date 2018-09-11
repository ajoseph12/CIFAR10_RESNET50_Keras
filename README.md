RESNET50 with Keras 
====================

Introduction
-------------

First and foremost, this repository has been inspired by Andrew Ng's Convolutional Neural Networks course: https://www.coursera.org/learn/convolutional-neural-networks . The python script here was written in primarily using Keras and was used to train on the "CIFAR-10" dataset : https://www.cs.toronto.edu/~kriz/cifar.html. The script too has been motivated from one of Andrew Ng's assignments. The pre-processing part, however, was redone to deconstruct the "CIFAR-10" dataset. 

In the repository, one would also find a saved model with a test accuracy of about 74%. The parametric tuning used to achieve this result was:
- Epochs : 100
- Batch size : 64

The model can be downloaded from this link : https://drive.google.com/open?id=1TglOyjOa8W8qQQzctO5GmfCelvwc4H2r
Be sure to save the model into a folder called 'data' within the cloned repo.


The script with the above tuning was run on a machine with 16 GB ram, 2GB V graphics and 1TB storage.  

Getting started
----------------
Running the script is quite straightforward and can be run directly from the terminal. The script performs three main tasks: training, comparing or testing, predicting. For each of these tasks, the last part of the script must be modified - mode, path. 

The accuracy I achieved using the RESNET50 network was quite low - adding dropout could possibly help. So feel free to  pull the repo, tweak the model and try climbing higher on the accuracy ladder. Have fun :)

