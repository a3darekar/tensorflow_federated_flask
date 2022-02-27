import collections
import os

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.python.keras import models
from keras import backend as K
from setup import INPUT_SHAPE, CLASS_COUNT

ModelVariables = collections.namedtuple(
	'ModelVariables', 'dense_kernel_0 dense_bias_0 dense_1_kernel_0 dense_1_bias_0 dense_2_kernel_0 dense_2_bias_0 '
					  'num_examples loss_sum accuracy_sum '
)

input_spec = collections.OrderedDict(
	x=tf.TensorSpec(dtype=tf.float32, shape=INPUT_SHAPE),
	y=tf.TensorSpec(shape=[None, CLASS_COUNT], dtype=tf.float32))


def create_variables():
	return ModelVariables(
		dense_kernel_0=tf.Variable(
			lambda: tf.zeros(shape=(28, 512), dtype=tf.float32),
			name='dense/kernel:0',
			trainable=True),
		dense_bias_0=tf.Variable(
			lambda: tf.zeros(shape=(512,), dtype=tf.float32),
			name='dense/bias:0',
			trainable=True),
		dense_1_kernel_0=tf.Variable(
			lambda: tf.zeros(shape=(14336, 512), dtype=tf.float32),
			name='dense_1/kernel:0',
			trainable=True),
		dense_1_bias_0=tf.Variable(
			lambda: tf.zeros(shape=(512,), dtype=tf.float32),
			name='dense_1/bias:0',
			trainable=True),
		dense_2_kernel_0=tf.Variable(
			lambda: tf.zeros(shape=(512, 10), dtype=tf.float32),
			name='dense_2/kernel:0',
			trainable=True),
		dense_2_bias_0=tf.Variable(
			lambda: tf.zeros(shape=(10,), dtype=tf.float32),
			name='dense_2/bias:0',
			trainable=True
		),
		num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
		loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
		accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False)
	)


def forward_pass(batch, variables):
	y = predict_on_batch(variables, batch['x'])
	predictions = tf.cast(tf.argmax(y, 1), tf.int32)
	labels = tf.reshape(batch['y'], [-1])

	loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(labels, 10) * tf.math.log(y), axis=[1]))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

	num_examples = tf.cast(tf.size(batch['y']), tf.float32)
	variables.num_examples.assign_add(num_examples)
	variables.loss_sum.assign_add(loss * num_examples)
	variables.accuracy_sum.assign_add(accuracy * num_examples)
	return loss, predictions


def predict_on_batch(x, variables):
	return tf.nn.softmax(tf.matmul(x, variables.weights) + variables.bias)


@tff.federated_computation
def aggregate_metrics(metrics):
	return collections.OrderedDict(
		num_examples=tff.federated_sum(metrics.num_examples),
		loss=tff.federated_mean(metrics.loss, metrics.num_examples),
		accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples)
	)


def get_local_metrics(variables):
	return collections.OrderedDict(
		num_examples=variables.num_examples,
		loss=variables.loss_sum / variables.num_examples,
		accuracy=variables.accuracy_sum / variables.num_examples
	)


def fetch_model(model_file=None):
	model = models.load_model(model_file)
	import numpy as np
	# readWeights
	weights = np.array()
	if model_file and os.path.isfile(model_file):
		print("Loading model with saved weights")
		weights = np.load(model_file, allow_pickle=True)
		model.set_weights(weights)
	return model


def set_model_weights(model: models.Model, weight_list):
	for i, symbolic_weights in enumerate(model.weights):
		weight_values = weight_list[i]
		K.set_value(symbolic_weights, weight_values)


def create_keras_model():
	return tf.keras.Sequential([
		tf.keras.layers.Input(shape=tuple(INPUT_SHAPE)),
		tf.keras.layers.Dense(512),
		tf.keras.layers.Activation('relu'),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512),
		tf.keras.layers.Activation('relu'),
		tf.keras.layers.Dense(10, activation="softmax"),
	])


def fetch_callable_model():
	return tff.learning.from_keras_model(
		keras_model=create_keras_model(),
		input_spec=input_spec,
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)


def fetch_optimizer(learning_rate=None):
	return tf.keras.optimizers.SGD(learning_rate)


def get_rid_of_the_models():
	return None
