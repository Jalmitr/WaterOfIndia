#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ashutosh

A one-line description or name.

A longer description that spans multiple lines with 72 character limit per line.
Explain the purpose of the file and provide a short list of the key
classes/functions it contains.
This is the docstring shown when some does 'import foo;foo?' in IPython, so it
should be reasonably useful and informative.
"""

# Imports
import os
import numpy
import cv2
import tensorflow
from tensorflow.keras.utils import to_categorical


IMG_CHANNELS = 3
NUM_OBSERVATIONS = 2913  # 72 images, 2913 with 2 datasets
dataset_small = "../data/dataset_small_kaggle/"
dataset_large = "../data/dataset_large_kaggle/"
water = numpy.array([226, 169, 41])
dimensions_list = [16, 32, 64, 128, 256]

for onedim in dimensions_list:

    IMG_WIDTH = onedim
    IMG_HEIGHT = onedim
    images1 = []
    images = numpy.zeros((NUM_OBSERVATIONS, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=numpy.uint8)
    labels = numpy.zeros((NUM_OBSERVATIONS, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=numpy.uint8)

    # read satellite images and masks from disk drive
    idx1 = 0
    for path, subs, files in os.walk(dataset_small):
        dirname = path.split(os.path.sep)[-1]
        if dirname == "images":
            for idx1, one_img_name in enumerate(files):
                img_full_path = path + "/" + one_img_name
                oneimage = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
                oneimage = cv2.resize(oneimage, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                images[idx1] = oneimage
                # images1.append(numpy.array(oneimage))
            idx1 = 0
        elif dirname == "masks":
            for idx1, one_mask_name in enumerate(files):
                mask_full_path = path + "/" + one_mask_name
                onemask = cv2.imread(mask_full_path, cv2.IMREAD_COLOR)
                onemask = cv2.cvtColor(onemask, cv2.COLOR_BGR2RGB)
                onemask = cv2.resize(onemask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                labels[idx1, numpy.all(onemask == water, axis=-1)] = 1

    idx1 = idx1 + 1

    images2 = []

    # read satellite images and masks from disk drive
    for path, subs, files in os.walk(dataset_large):
        dirname = path.split(os.path.sep)[-1]
        if dirname == "images":
            for idx2, one_img_name in enumerate(files):
                img_full_path = path + "/" + one_img_name
                oneimage = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
                oneimage = cv2.resize(oneimage, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                images[idx1 + idx2] = oneimage
                # images2.append(numpy.array(oneimage))
            idx2 = 0
        elif dirname == "masks":
            for idx2, one_mask_name in enumerate(files):
                mask_full_path = path + "/" + one_mask_name
                onemask = cv2.imread(mask_full_path, cv2.IMREAD_COLOR)
                onemask = cv2.cvtColor(onemask, cv2.COLOR_BGR2RGB)
                onemask = cv2.resize(onemask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                labels[idx1 + idx2, numpy.all(onemask > 128, axis=-1)] = 1

    labels = labels[:, :, :, 0]  # all target labels
    labels = to_categorical(labels, num_classes=2, dtype=numpy.uint8)
    # labels = labels.astype(numpy.uint8)

    numpy.savez_compressed(
        "../data/kaggle_npz/images_labels_{d1}x{d2}.npz".format(d1=int(IMG_HEIGHT), d2=int(IMG_WIDTH)), images=images, labels=labels)

    print("\n\nDone with", onedim)
