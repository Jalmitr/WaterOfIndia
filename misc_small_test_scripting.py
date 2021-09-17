#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ashutosh

A miscellaneous python file for testing and trying snipppets of code. Kind of a scratch pad.
"""
from util_funcs import selected_samples_area, plot_selected_samples, plot_random_samples
from sklearn.model_selection import train_test_split
import tensorflow
import numpy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # set Tensorflow logging for only errors and above level
# tensorflow.get_logger().setLevel("WARNING")  # in case above filtering doesn't work
numpy.random.seed(seed=42)  # set the seed of RNG for reproducibility and debugging

# Tensorflow flag to set whether to use CPU or GPU for training.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # activate this line to use CPU
physical_devices = tensorflow.config.list_physical_devices("GPU")  # activate this...

arr = numpy.arange(start=1, stop=61, step=1, dtype=int)
arr1 = numpy.random.randint(0, 255, (10, 10, 3), numpy.uint8)

# load model
model_path = "./woi_model/unet_128x128/"
# unet_model = tensorflow.saved_model.load(model_path)
unet_model = tensorflow.keras.models.load_model(model_path)

# call selecte
# area_result = selected_samples_area(unet_model, img_height=128, img_width=128)

# call plot selected
# plot_selected_samples(unet_model, img_height=128, img_width=128)

# multiply to get area

# area_result[:, 3:] = 100*area_result[:, 3:]

# print(area_result)
# print(area_result[:, 4])

#####     load data     #####
IMG_WIDTH = 128
IMG_HEIGHT = 128
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

plot_random_samples(X_test, y_test, unet_model, n_examples=10)
plot_selected_samples(unet_model, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
