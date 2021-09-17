#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ashutosh

A utility to calculate area in pixel-equivalent-unit, i.e. PEU.

"""
import os
import cv2
import numpy
import tensorflow
import rasterio


def calculate_area(binary_image):

    label_peu, area_peu = numpy.unique(binary_image, return_counts=True)

    return label_peu, area_peu


def calculate_centroid(binary_image):

    # for multiple bodies, needs contour detection and for loop
    # Ref: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    if binary_image.ndim == 4:
        binary_image = numpy.squeeze(binary_image, axis=0)
        binary_image = numpy.uint8(binary_image)
    else:
        binary_image = numpy.uint8(binary_image)
    image_moments = cv2.moments(binary_image)
    x_coord = int(image_moments["m10"]/image_moments["m00"])
    y_coord = int(image_moments["m01"]/image_moments["m00"])
    centroid = numpy.array([x_coord, y_coord])

    return centroid


def selected_samples_area(unet, img_height, img_width):

    satellite_tiff = "./data/sentinel_test/"
    # print(os.walk(satellite_tiff))
    # path, subs, files = os.walk(satellite_tiff)

    for path, subs, files in os.walk(satellite_tiff):
        area_result = numpy.zeros((len(files), 5), dtype=object)
        for idx1, onefile in enumerate(files):
            area_result[idx1, 0] = onefile
            # if onefile.endswith(".tif"):
            dataset = rasterio.open(path + onefile)
            image = dataset.read()
            image = numpy.moveaxis(image, 0, -1)
            # elif onefile.endswith("jpg"):
            #     image = tensorflow.io.read_file(path + onefile)
            #     image = tensorflow.io.decode_image(image, channels=3)
            #     image = image.numpy()
            image = image.astype(numpy.float32)/255.0
            image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
            # image = tensorflow.image.resize(image, [img_height, img_width], method="lanczos5")
            image = image[None, :, :, :]
            prediction = unet.predict(image)[0]
            prediction = numpy.argmax(prediction, axis=2)
            # prediction = numpy.expand_dims(prediction, axis=2)

            label_peu, area_peu = calculate_area(prediction)
            area_result[idx1, 1:3] = label_peu
            area_result[idx1, 3:] = area_peu

    return area_result
