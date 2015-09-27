import time
import numpy as np
import lasagne
import theano
import theano.tensor as T
from chunk import SimpleChunkLoader
from image import ImageConventionConverter

# the following model is useless
def build_model():
	input_layer = lasagne.layers.InputLayer((None, 256, 256, 3))
	hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=100)
	output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=2, 
		nonlinearity=T.nnet.softmax)
	return output_layer

def create_iter_func(dataset, output_layer, batch_size=128, 
	learning_rate=0.01, momentum=0.9):

	batch_index = T.iscalar('batch_index')
	X_batch = lasagne.utils.shared_empty(dim=4, dtype=np.float32)
	y_batch = lasagne.utils.shared_empty(dim=1, dtype=np.int32)
	batch_slice = slice(batch_index * batch_size, 
		(batch_index + 1) * batch_size)
	objective = lasagne.objectives.Objective(
		output_layer, loss_function=lasagne.objectives.categorical_crossentropy)

	loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)
	pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
	accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

	all_params = lasagne.layers.get_all_params(output_layer)
	updates = lasagne.updates.adagrad(loss_train, all_params, learning_rate)

	iter_train = theano.function(
		[batch_index], loss_train, 
		updates=updates, 
		givens={
		X_batch: dataset['X_train'][batch_slice],
		y_batch: dataset['y_train'][batch_slice],
		},
	)

	iter_valid = theano.function(
		[batch_index], [loss_eval, accuracy], 
		givens={
		X_batch: dataset['X_train'][batch_slice],
		y_batch: dataset['y_train'][batch_slice],
		},
	)

	return {'train': iter_train, 'valid': iter_valid,}

def test(l_out, data_loader, train_or_test='train', 
	chunk_size=2048, batch_size=128):
	pass

def train():
	pass