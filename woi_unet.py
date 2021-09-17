#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ashutosh

Main script to execute the UNET model.
"""
from woi_model import unet_model
from util_funcs import plot_loss, plot_iou, plot_random_samples, plot_selected_samples
from util_funcs import callbacks_util, save_model_explicit
from keras import backend as K
from datetime import datetime
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow
import cv2
import numpy
import sys
import rasterio
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # set Tensorflow logging for only errors and above level
# tensorflow.get_logger().setLevel("WARNING")  # in case above filtering doesn't work
numpy.random.seed(seed=42)  # set the seed of RNG for reproducibility and debugging

# Tensorflow flag to set whether to use CPU or GPU for training.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # activate this line to use CPU
physical_devices = tensorflow.config.list_physical_devices("GPU")  # activate this...
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)  # ...and this line to use GPU


# Constants to be used throughout. Put it in separate constants.py file
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
NUM_OBSERVATIONS = 2913  # 72 images, 2913 with 2 datasets

#####     load data     #####
data_path = "./data/kaggle_npz/images_labels_{n1}x{n2}.npz".format(n1=IMG_HEIGHT, n2=IMG_WIDTH)
data = numpy.load(data_path, mmap_mode="r")
images = data["images"]
images = images.astype(numpy.float32)/255.0
labels = data["labels"]

print("Dataset dimensions (observations, height, width, channels):", images.shape)

# splitting data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.30)

print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)

#####     build model     #####
unet = unet_model(n_classes=2, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS)
# print(unet.summary())

#####     train model     #####
# some objects initializations
epochs = 15
batch_size = 16
optimization_function = tensorflow.keras.optimizers.Nadam(learning_rate=1e-3)
loss_function = ["binary_crossentropy"]
metrics_list = [tensorflow.keras.metrics.MeanIoU(num_classes=2)]

# Keras callbacks
all_callbacks = callbacks_util(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, csv_log=True,
                               ely_stop=True, lr_schd=False, mdl_ckpt=True, rlrplat=True)

# compile model
unet.compile(optimizer=optimization_function, loss=loss_function, metrics=metrics_list)

# fit the model
history = unet.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size,
                   epochs=epochs, callbacks=all_callbacks, shuffle=True)

# saving the model explicitly, when not using Checkpoint save. Change flag to True
save_model_explicit(unet, img_height=IMG_WIDTH, img_width=IMG_WIDTH, save_flag=False)

print("\n\n")
print(history.history)

#####     results     #####
plot_loss(history)
plot_iou(history)
plot_random_samples(X_test, y_test, unet, n_examples=10)
plot_selected_samples(unet, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)


print("\n\n")
print("Done")
print("\n\n")

##########          To do          ##########
# Clean up
# add code for confusion matrix for binary case
# how to log ML experiments and runs
#    using CSV file for now
#    look into MLflow and Tensorboard and other options
# add code for saving the model
#    gotta save model and history separately?
#        https://stackoverflow.com/questions/47843265/how-can-i-get-a-keras-models-history-after-loading-it-from-a-file-in-python
#        https://stackoverflow.com/questions/65352369/tf-saving-model-with-its-history-history-to-use-it-in-the-following-training-s
# try out different optimizers (currently Nadam)
# generate docstring
# try out more metrics
# problem with current UNET model as it takes image height and width as Inputs
# try out resizing with nearest neighbour instead of bilinear default
# accessing the augmented data from disk
# divide 5000x5000 into multiple smaller images my 3 sentinel images
# edge case of area calculation of partial waterbodies
# try this code wit grayscale images instead of RGB
# # # # # Below is github upload
