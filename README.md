# Reproducing "Learning to See in the Dark" with Reduced Dataset Size and Epochs

This is a reproduction of the model presented in the paper "Learning to See in the Dark" by Chen et al. (2018) with a smaller dataset and fewer epochs. The aim of this reproduction is to investigate the performance of different CNN architectures, specifically SegNet and DAnet, and compare them with the U-Net architecture used in the original paper. The source code has been rewritten in Python 3.8 from the original code, which was in Python 2.7.

## Steps

The steps to train and test the models are the same as the ones in the original paper, which can be found in the repository https://github.com/cchen156/Learning-to-See-in-the-Dark. The difference is that we have reduced the dataset size and the number of epochs for each of the three models used in this reproduction.

To train the desired model, run one of the following commands in the command line:
bash
python train_Sony_Unet.py
python train_Sony_segnet.py
python train_Sony_danet.py


After training, to test the model, run one of the following commands:

bash
python test_Sony_Unet.py
python test_Sony_segnet.py
python test_Sony_danet.py


If the user has a better GPU, they can change the dataset size and the number of epochs in the train files to train the model on the whole dataset with the same number of epochs as the original paper and thus get better results than we did.

## Conclusion

The results of this reproduction show that the U-Net architecture performs better than the SegNet and DaNet architectures on a reduced dataset size and fewer epochs. One reason for this is that we did not change the hyperparameters of the U-Net architecture while using the same hyperparameters for the other architectures. The results also show that the DAnet architecture has a much faster training time and smaller model size than the other two architectures. This makes the DaNet architecture a promising replacement for the U-net architecture if the hyperparameters are tuned. 