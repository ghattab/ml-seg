
# coding: utf-8

import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d, max_pool_2d, upscore_layer
import tflearn.layers.core
import cv2
import numpy
import matplotlib.pyplot as plt


img = cv2.imread('dd/smallblue_kolonie4_16_4_14t115.tif')

patches = img.reshape([1] + list(img.shape))

input_shape = [None, patches.shape[1], patches.shape[2], patches.shape[3]]

# img = cv2.resize(img, (img.shape[0] / 2, img.shape[1] / 2))
# img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
# print img.shape
NUM_CHANNELS = 3
NUM_LABELS = 10

img_prep = tflearn.ImagePreprocessing()
# Zero Center (With mean computed over the whole dataset)
img_prep.add_featurewise_zero_center(124.165396724)

tflearn.init_graph()

# Building 'Network In Network'
network = input_data(shape=input_shape, data_preprocessing=img_prep)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 160, 1, activation='relu')
network = conv_2d(network, 96, 1, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = conv_2d(network, NUM_LABELS, 1, activation='relu', restore=False)
# network = upscore_layer(network, NUM_LABELS, [1, input_shape[1] / 2, input_shape[2] / 2], restore=False)
network = upscore_layer(network, NUM_LABELS, [1, input_shape[1], input_shape[2]], restore=False)


# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('bignin.tflearn')
heatmap = model.predict(patches)[0]
idxmap = numpy.zeros(img.shape[0:2], numpy.uint8)

for i in range(NUM_LABELS):
    heatmaps = numpy.zeros(img.shape[0:2], numpy.uint8)
    plt.imsave('heatmaps/ninheatmap' + str(i) + ".png", heatmaps[:, :, i])
idxmap = numpy.argmax(heatmap, 2)
plt.imsave('heatmaps/ninidxmap.png', idxmap)
