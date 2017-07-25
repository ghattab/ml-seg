import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d, max_pool_2d, upscore_layer
from tflearn.layers.estimator import regression
import tflearn.layers.core
import cv2
import numpy
import matplotlib.pyplot as plt
import collections

img = cv2.imread('dd/small2015_09_15_colony17t19.tif')

xs = [0, img.shape[0] / 4, img.shape[0] / 2, 3 * img.shape[0] / 4]
ys = [0, img.shape[1] / 4, img.shape[1] / 2, 3 * img.shape[1] / 4]

patches = []
for x in xs:
    for y in ys:
        patches.append(img[x:x + img.shape[0] / 4, y:y + img.shape[1] / 4, :])

patches = numpy.array(patches, float)

input_shape = [None, patches.shape[1], patches.shape[2], patches.shape[3]]

NUM_CHANNELS = 3
NUM_LABELS = 10

# remove mean

img_prep = tflearn.ImagePreprocessing()
# Zero Center (With mean computed over the whole dataset)
img_prep.add_featurewise_zero_center(124.165396724)


tflearn.init_graph()

network = input_data(shape=input_shape)
network = conv_2d(network, 100, 5, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 250, 5, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 500, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 500, 1, activation='relu', restore=False)
# network = conv_2d(network, 500, 1, activation='relu', restore=False)
network = conv_2d(network, NUM_LABELS, 1, activation='relu', restore=False)
network = upscore_layer(network, NUM_LABELS, [1, input_shape[1] / 4, input_shape[2] / 4], restore=False)
network = upscore_layer(network, NUM_LABELS, [1, input_shape[1] / 2, input_shape[2] / 2], restore=False)
network = upscore_layer(network, NUM_LABELS, [1, input_shape[1], input_shape[2]], restore=False)
network = regression(network, optimizer='adam', loss='softmax_categorical_crossentropy', learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load("bigconv.tflearn")
heatmap = []
for i in patches:
    heatmap.append(numpy.array(model.predict(i.reshape((1, i.shape[0], i.shape[1], i.shape[2])))[0]))
heatmap = numpy.array(heatmap)
idxmap = numpy.zeros(img.shape[0:2], numpy.uint8)

for i in range(NUM_LABELS):
    heatmaps = numpy.zeros(img.shape[0:2], numpy.uint8)
    c = 0
    for x in xs:
        for y in ys:
            heatmaps[x:x + img.shape[0] / 4, y:y + img.shape[1] / 4] = heatmap[c, :, :, i]
            c += 1
    plt.imsave('heatmaps/convheatmap' + str(i) + ".png", heatmaps)
c = 0
for x in xs:
    for y in ys:
        idxmap[x:x + img.shape[0] / 4, y:y + img.shape[1] / 4] = numpy.argmax(heatmap[c], 2)
        c += 1
plt.imsave('heatmaps/convidxmap.png', idxmap)
print collections.Counter(idxmap.flatten())
