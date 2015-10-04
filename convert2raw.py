import os.path
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('right-whale')

from collections import defaultdict
import cPickle as pkl 

import numpy as np 
import pandas as pd 
import skimage.io 

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def load_from_jpeg():
    """
    Load the train data_loader from `.jpeg`files.
    """
    labels = pd.read_csv(FILE_PATH + '/data/train.csv').values
    images = np.empty((labels.shape[0], 256, 256, 3), dtype='uint8')
    for k in range(labels.shape[0]):
    	img = skimage.io.imread(FILE_PATH + '/data/imgs/%s.jpeg' % labels[k, 0])
    	images[k, :, :, :] = img
    	logger.debug("Loaded image %s.jpeg" % labels[k, 0])
    logger.info("Loaded %s samples." % labels.shape[0])
    return images, np.array(labels[:, 1], dtype=np.int32)

def convert_to_raw():
	X, y = load_from_jpeg()
	X.tofile(FILE_PATH + '/data/images.raw')
	y.tofile(FILE_PATH + '/data/labels.raw')

def load_from_raw():
	"""
	Load data_loader using memory mapping for large arrays.
	"""
    labels = pd.read_csv(FILE_PATH + '/data/train.csv').values
    n = labels.shape[0]
	X = np.memmap(FILE_PATH + '/data/images.raw',
		dtype=np.uint8, shape=(n, 256, 256, 3))
	y = np.memmap(FILE_PATH + '/data/labels.raw',
		dtype=np.int32, shape=(n,))
	logger.info("Loaded %s samples." % y.shape[0])
	return X, y

def create_train_test(X, y, test_size):
	from sklearn.cross_validation import StratifiedShuffleSplit
	sss = StratifiedShuffleSplit(y,1,test_size=test_size,random_state=0)
	train_index, test_index = iter(sss).next()

	X_train = X[train_index]
	X_train.tofile(FILE_PATH + '/data/X_train.raw')
	y_train = y[train_index]
	y_train.tofile(FILE_PATH + '/data/y_train.raw')
	logger.info("Dumped %s training examples" % y_train.shape[0])

	X_test = X[test_index]
	X_test.tofile(FILE_PATH + '/data/X_test.raw')
	y_test = y[test_index]
	y_test.tofile(FILE_PATH + '/data/y_test.raw')
	logger.info("Dumped %s testing examples" % y_test.shape[0])

if __name__ == "__main__":

	# Create train and test raw files
	convert_to_raw()
	X, y = load_from_raw()
	create_train_test(X, y, test_size=.2)