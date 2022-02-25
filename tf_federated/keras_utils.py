import collections
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.python.keras import models
from keras import backend as K
from setup import INPUT_SHAPE, CLASS_COUNT

ModelVariables = collections.namedtuple(
	'ModelVariables', 'weights bias num_examples loss_sum accuracy_sum'
)


def create_variables():
	return ModelVariables(
		weights=tf.Variable(
			lambda: tf.zeros(dtype=tf.float32, shape=INPUT_SHAPE),
			name='weights',
			trainable=True
		),
		bias=tf.Variable(
			lambda: tf.zeros(dtype=tf.float32, shape=CLASS_COUNT),
			name='bias',
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
	set_model_weights(model, weights)
	return model


def set_model_weights(model: models.Model, weight_list):
	for i, symbolic_weights in enumerate(model.weights):
		weight_values = weight_list[i]
		K.set_value(symbolic_weights, weight_values)
