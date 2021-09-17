#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ashutosh

A helper utility to plot results of the CNN.

Plotting results include:
    Loss plots
    IoU plots
    Predicted masks for randomly selected samples
    Predicted masks for specific images passed as arguments
"""

import os
import numpy
import matplotlib.pyplot as plt
import rasterio
import tensorflow
from util_funcs import calculate_area, calculate_centroid
plt.rcParams.update({"font.size": 8})


def plot_loss(history):

    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    xaxis_epochs = numpy.arange(1, len(train_loss) + 1)
    plt.figure(dpi=120)
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(xaxis_epochs, train_loss, 'y', label='Training loss')
    plt.plot(xaxis_epochs, valid_loss, 'r', label='Validation loss')
    plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.savefig("./plots_results/loss_vs_epochs.png", dpi=1200, facecolor="w", edgecolor="w",
                orientation="portrait", format="png", transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None)
    plt.close()

    print("Loss plot saved in directory ./plot_results")

    return 0


def plot_iou(history):
    train_iou = history.history['mean_io_u']
    valid_iou = history.history['val_mean_io_u']
    xaxis_epochs = numpy.arange(1, len(train_iou) + 1)
    plt.figure(dpi=120)
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('Intersection over Union [IoU]')
    plt.plot(xaxis_epochs, train_iou, 'y', label='Training IoU')
    plt.plot(xaxis_epochs, valid_iou, 'r', label='Validation IoU')
    plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.savefig("./plots_results/iou_vs_epochs.png", dpi=1200, facecolor="w", edgecolor="w",
                orientation="portrait", format="png", transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None)
    plt.close()

    print("IoU plot saved in directory ./plot_results")

    return 0


def plot_random_samples(X_test, y_test, unet, n_examples=10):

    random_indices = numpy.random.choice(X_test.shape[0], n_examples, replace=False)
    random_X_test = X_test[random_indices]
    random_y_test = y_test[random_indices]

    for imgidx, one_image, one_label in zip(range(n_examples), random_X_test, random_y_test):
        one_image = numpy.expand_dims(one_image, axis=0)
        one_label = numpy.argmax(one_label, axis=2)
        one_label = 255*one_label
        one_label = numpy.expand_dims(one_label, axis=(0, 3))
        prediction = unet.predict(one_image)[0]
        prediction = 255*numpy.argmax(prediction, axis=2)
        prediction = numpy.expand_dims(prediction, axis=2)
        _, org_area = calculate_area(one_label)
        _, prd_area = calculate_area(prediction)
        org_cent = calculate_centroid(one_label)
        prd_cent = calculate_centroid(prediction)

        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        axs[0].set_title('Original image')
        axs[0].imshow(one_image[0])
        axs[1].set_title('Original mask {a1} PEU'.format(a1=org_area[1]))
        axs[1].imshow(one_label[0])
        axs[1].plot(org_cent[0], org_cent[1], marker="o", markersize=8, markeredgecolor="black", markerfacecolor="red")
        axs[1].text(org_cent[0], org_cent[1], "({x}, {y})".format(x=org_cent[0], y=org_cent[1]),
                    fontsize=10, color="red")
        axs[2].set_title('Predicted mask {a1} PEU'.format(a1=prd_area[1]))
        axs[2].imshow(prediction)
        axs[2].plot(prd_cent[0], prd_cent[1], marker="o", markersize=8, markeredgecolor="black", markerfacecolor="red")
        axs[2].text(prd_cent[0], prd_cent[1], "({x}, {y})".format(x=prd_cent[0], y=prd_cent[1]),
                    fontsize=10, color="red")

        plt.savefig("./plots_results/predicted_original_masks/masks_predictions_{n1}.png".format(n1=imgidx), dpi=1200,
                    facecolor="w", edgecolor="w", orientation="portrait", format="png", transparent=False,
                    bbox_inches="tight", pad_inches=0.1, metadata=None)
        plt.close()

    print("Random samples predicted masks saved in directory ./plots_results/predicted_original_masks/")

    return 0


def plot_selected_samples(unet, img_height, img_width):

    satellite_tiff = "./data/sentinel_test/"
    idx1 = 0

    for path, subs, files in os.walk(satellite_tiff):
        for onefile in files:
            dataset = rasterio.open(path + onefile)
            image = dataset.read()
            image = numpy.moveaxis(image, 0, -1)
            image = image.astype(numpy.float32)/255.0
            image = tensorflow.image.resize(image, [img_height, img_width], method="lanczos5")
            image = image[None, :, :, :]
            prediction = unet.predict(image)[0]
            prediction = 255*numpy.argmax(prediction, axis=2)
            prediction = numpy.expand_dims(prediction, axis=2)
            _, prd_area = calculate_area(prediction)
            prd_cent = calculate_centroid(prediction)

            fig, axs = plt.subplots(1, 2, constrained_layout=True)
            axs[0].set_title('Original image')
            axs[0].imshow(image[0])
            axs[1].set_title('Predicted mask {a1} PEU'.format(a1=prd_area[1]))
            axs[1].imshow(prediction)
            axs[1].plot(prd_cent[0], prd_cent[1], marker="o", markersize=8,
                        markeredgecolor="black", markerfacecolor="red")
            axs[1].text(prd_cent[0], prd_cent[1], "({x}, {y})".format(x=prd_cent[0], y=prd_cent[1]),
                        fontsize=10, color="red")

            plt.savefig("./plots_results/predicted_original_masks/sentinel_predictions_{n1}.png".format(n1=idx1),
                        dpi=1200, facecolor="w", edgecolor="w", orientation="portrait", format="png", transparent=False,
                        bbox_inches="tight", pad_inches=0.1, metadata=None)
            plt.close()

            idx1 += 1

    print("Selected sample image masks saved in directory ./plots_results/predicted_original_masks/")

    return 0

# , horizontalalignment='center', verticalalignment='center'
