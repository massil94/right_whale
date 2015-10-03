import numpy as np
from abc import ABCMeta, abstractmethod

class Loader(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def get_chunks(self, data):
		pass

class SimpleChunkLoader(Loader):

	def __init__(self, chunk_size=2048):
		self.chunk_size = chunk_size
		self.y_train = np.memmap(FILE_PATH + '/data/y_train.raw',
			dtype=np.int32, mode='r')
		self.X_train = np.memmap(FILE_PATH + '/data/X_train.raw',
			dtype=np.uint8, mode='r', shape=(self.y_train.shape[0],256,256,3))
		self.y_test = np.memmap(FILE_PATH + '/data/y_test.raw',
			dtype=np.int32, mode='r')
		self.X_test = np.memmap(FILE_PATH + '/data/X_test.raw',
			dtype=np.uint8, mode='r', shape=(self.y_test.shape[0],256,256,3))
		
	def get_random_chunks(self, data='train'):
		chunk_size = self.chunk_size

		if data == 'train':
			X = self.X_train
			y = self.y_train
			num_chunks = self.X_train.shape[0] // chunk_size
		elif data == 'test':
			X = self.X_test
			y = self.y_test
			num_chunks = self.X_test.shape[0] // chunk_size
		else:
			raise Exception("`data`should be in {'train', 'test'}")

		for _ in range(num_chunks):
			indices = np.random.randint(y.shape[0], size=chunk_size)
			xc = X[indices]
			yc = y[indices]
			yield xc, yc

	def get_contiguous_chunks(self, data='train'):    
        chunk_size = self.chunk_size

        if data == 'train':
            X = self.X_train
            y = self.y_train
        elif data == 'test':
            X = self.X_test
            y = self.y_test
        else:
            raise Exception("`data` should be in {'train', 'test'}")
        
        num_chunks = int(np.ceil(y.shape[0] // chunk_size))
        for i in range(num_chunks):
            # TODO: This part is not correct. The last batch contains previously
            # seen example.
            if i == num_chunks:
                xc = X[chunk_size * i:]
                yc = y[chunk_size * i:]
                yield xc, yc
            else:
                xc = X[chunk_size * i: chunk_size * (i + 1)]
                yc = y[chunk_size * i: chunk_size * (i + 1)]
                yield xc, yc

    def get_chunks(self, data):
    	return self.get_contiguous_chunks(data)