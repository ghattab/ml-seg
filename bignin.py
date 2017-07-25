
# coding: utf-8

# In[ ]:

import tflearn
from tflearn.layers.core import input_data, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
import tflearn.layers.core
import h5py

# In[ ]:
IMAGE_SIZE = 32
NUM_CHANNELS = 3


h5f = h5py.File('data.h5', 'r')
X = h5f['X']
Y = h5f['Y']

NUM_LABELS = h5f['Y'].shape[1]


# Real-time image preprocessing
img_prep = tflearn.ImagePreprocessing()
# Zero Center (With mean computed over the whole dataset)
img_prep.add_featurewise_zero_center()

# img_aug = tflearn.ImageAugmentation()
# img_aug.add_random_flip_leftright()
# img_aug.add_random_90degrees_rotation()
# img_aug.add_random_blur(3.0)


# Classification
tflearn.init_graph()

# Building 'Network In Network'
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 160, 1, activation='relu')
network = conv_2d(network, 96, 1, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 1, activation='relu')
network = conv_2d(network, NUM_LABELS, 1, activation='relu')
network = avg_pool_2d(network, 8)
network = flatten(network)
network = regression(network, optimizer='adam', loss='softmax_categorical_crossentropy', learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=30, shuffle=True, batch_size=128, validation_set=0.15, snapshot_epoch=True, snapshot_step=None, show_metric=False, run_id='bignin')
model.save('bignin.tflearn')
