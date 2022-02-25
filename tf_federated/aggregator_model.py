import collections
import tensorflow as tf
import tensorflow_federated as tff
from setup import INPUT_SHAPE
from tf_federated.keras_utils import create_variables, forward_pass, predict_on_batch, aggregate_metrics, \
	get_local_metrics


class AggregatorModel(tff.learning.Model):
	def __init__(self):
		self._variables = create_variables()

	@property
	def trainable_variables(self):
		return [self._variables.weights, self._variables.bias]

	@property
	def non_trainable_variables(self):
		return []

	@property
	def local_variables(self):
		return [
			self._variables.num_examples, self._variables.loss_sum,
			self._variables.accuracy_sum
		]

	@property
	def input_spec(self):
		return collections.OrderedDict(
			x=tf.TensorSpec(INPUT_SHAPE, tf.float32),
			y=tf.TensorSpec([None, 1], tf.int32))

	@tf.function
	def predict_on_batch(self, x, training=True):
		return predict_on_batch(self._variables, x)

	@tf.function
	def forward_pass(self, batch, training=True):
		del training
		loss, predictions = forward_pass(self._variables, batch)
		num_examples = tf.shape(batch['x'])[0]
		return tff.learning.BatchOutput(
			loss=loss, predictions=predictions, num_examples=num_examples)

	@tf.function
	def report_local_outputs(self):
		return get_local_metrics(self._variables)

	@property
	def federated_output_computation(self):
		return aggregate_metrics
