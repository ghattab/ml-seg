dataset_file = '../res.txt'

from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_file, image_shape=(32, 32), mode='file', output_path='../data.h5', categorical_labels=True, normalize=True)
