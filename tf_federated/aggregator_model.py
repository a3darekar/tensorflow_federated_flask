import collections
import os
from typing import Callable, List, OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from setup import AGGR_MODEL_FILE, AGGR_OPTIMIZER_LEARNING_RATE, CLIENT_OPTIMIZER_LEARNING_RATE
from tf_federated.keras_utils import create_variables, forward_pass, predict_on_batch, aggregate_metrics, \
	get_local_metrics, create_keras_model, input_spec


class AggregatorModel(tff.learning.Model):
	def __init__(self):
		super(tff.learning.Model, self).__init__()
		self._variables = create_variables()

	def get_model(self):
		model = create_keras_model()
		if os.path.isfile(AGGR_MODEL_FILE):
			print("Loading model with saved weights")
			weights = np.load('weights.npy', allow_pickle=True)
			model.set_weights(weights)
		return tff.learning.from_keras_model(
			model,
			input_spec=self.input_spec,
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
		)

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
		return input_spec

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

	def init_aggregator(self, process=None):
		if process == 'SGD':
			# init avg aggregator
			aggregator_process = tff.learning.build_federated_sgd_process(
				model_fn=self.get_model,
				client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=CLIENT_OPTIMIZER_LEARNING_RATE),
				server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=AGGR_OPTIMIZER_LEARNING_RATE),
			)
		else:
			# init avg aggregator
			aggregator_process = tff.learning.build_federated_averaging_process(
				model_fn=self.get_model,
				client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=CLIENT_OPTIMIZER_LEARNING_RATE),
			)
		init_state = aggregator_process.initialize()
		print(init_state)
		return aggregator_process, init_state
