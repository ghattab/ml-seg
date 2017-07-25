import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.layers.core
import h5py


# In[ ]:
IMAGE_SIZE = 32
NUM_CHANNELS = 3


h5f = h5py.File('data.h5', 'r')
X = h5f['X']
Y = h5f['Y']

NUM_LABELS = h5f['Y'][0].shape[0]
print NUM_LABELS


# Real-time image preprocessing
img_prep = tflearn.ImagePreprocessing()
# Zero Center (With mean computed over the whole dataset)
img_prep.add_featurewise_zero_center()


tflearn.init_graph()

network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep)
network = conv_2d(network, 100, 5, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 250, 5, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 500, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = fully_connected(network, 1500, activation='relu')
network = fully_connected(network, NUM_LABELS)
network = regression(network, optimizer='adam', loss='softmax_categorical_crossentropy', learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=30, shuffle=True, batch_size=64, validation_batch_size=32, validation_set=0.15, snapshot_epoch=True, snapshot_step=None, show_metric=False, run_id='bigconv')
model.save('bigconv.tflearn')
