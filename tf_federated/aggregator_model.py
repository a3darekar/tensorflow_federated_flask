import collections
from typing import Callable, List, OrderedDict

import tensorflow as tf
import tensorflow_federated as tff
from setup import INPUT_SHAPE
from tf_federated.keras_utils import create_variables, forward_pass, predict_on_batch, aggregate_metrics, \
	get_local_metrics


class AggregatorModel(tff.learning.Model):
	def __init__(self):
		super(tff.learning.Model, self).__init__()
		self._variables = create_variables()

	@property
	def trainable_variables(self):
		return {
			'dense_kernel_0': self._variables.dense_kernel_0.numpy().tolist(),
			'dense_bias_0': self._variables.dense_bias_0.numpy().tolist(),
			'dense_1_kernel_0': self._variables.dense_1_kernel_0.numpy().tolist(),
			'dense_1_bias_0': self._variables.dense_1_bias_0.numpy().tolist(),
			'dense_2_kernel_0': self._variables.dense_2_kernel_0.numpy().tolist(),
			'dense_2_bias_0': self._variables.dense_2_bias_0.numpy().tolist()
		}

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

	@tf.function
	def report_local_unfinalized_metrics(
			self) -> OrderedDict[str, List[tf.Tensor]]:
		"""Creates an `OrderedDict` of metric names to unfinalized values."""
		return collections.OrderedDict(
			num_examples = [self._variables.num_examples],
			loss = [self._variables.loss_sum, self._variables.num_examples],
			accuracy = [self._variables.accuracy_sum, self._variables.num_examples]
		)

	def metric_finalizers(
			self) -> OrderedDict[str, Callable[[List[tf.Tensor]], tf.Tensor]]:
		"""Creates an `OrderedDict` of metric names to finalizers."""
		return collections.OrderedDict(
			num_examples=tf.function(func=lambda x: x[0]),
			loss=tf.function(func=lambda x: x[0] / x[1]),
			accuracy=tf.function(func=lambda x: x[0] / x[1])
		)
