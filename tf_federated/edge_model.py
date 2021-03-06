import os
from typing import Callable

from tensorflow.python.keras import models
import numpy as np
import tensorflow.python.keras.losses

from .keras_utils import get_rid_of_the_models


class Client:
	def __init__(self, client_id: int):
		self.client_id = client_id
		self.model: models.Model = None
		self.x_train = None
		self.y_train = None

	def init_model(self, model_fn: Callable, model_weights=None):
		model = model_fn()
		if model_weights:
			if isinstance(model_weights, list):
				model.set_weights(model_weights)
			elif os.path.isfile(model_weights):
				print("Loading model with saved weights")
				weights = np.load(model_weights, allow_pickle=True)
				model.set_weights(weights)
			else:
				print("model_weights", model_weights)
				# print("Unrecognizable weights type: ". type(model_weights))
				pass
		else:
			print("Loading fresh model")
		model.compile(
			loss=tensorflow.keras.losses.categorical_crossentropy,
			optimizer=tensorflow.keras.optimizers.Adadelta(),
			metrics=['accuracy']
			)
		print(model.summary())
		self.model = model

	def receive_data(self, x, y):
		self.x_train = x.astype("float32") / 255
		self.y_train = y

	def receive_and_init_model(self, model_fn: Callable, model_weights):
		self.init_model(model_fn, model_weights)

	def local_train(self, client_train_dict: dict):
		if self.model is None:
			raise ValueError("Model is not created for client: {0}".format(self.client_id))

		hist = self.model.fit(self.x_train, self.y_train, **client_train_dict)
		return hist

	def edge_train(self, client_train_dict: dict):
		import tensorflow as tf
		if self.model is None:
			raise ValueError("Model is not created for client: {0}".format(self.client_id))
		x = self.x_train[:2]
		y = self.y_train[:2]
		predictions, flat_labels = tf.cast(self.model.predict(x), tf.float32), tf.cast(y, tf.float32)
		loss = -tf.reduce_mean(
			tf.reduce_sum(flat_labels * predictions, axis=[1])
		)
		num_examples = tf.cast(y.shape[0], tf.float32)
		accuracy = tf.reduce_sum(
			tf.cast(tf.equal(tf.math.argmax(predictions, 1), tf.math.argmax(flat_labels, 1)), tf.float32)
		) / num_examples
		loss_sum = loss * num_examples
		forward_pass_metrics = {
			'predictions': predictions.numpy().tolist(),
			'loss': str(loss.numpy()),
			'num_examples': str(num_examples.numpy()),
			'loss_sum': str(loss_sum.numpy()),
			'accuracy': str(accuracy.numpy()),
		}
		return forward_pass_metrics

	def reset_model(self):
		get_rid_of_the_models(self.model)

	def evaluate(self, x, y):
		try:
			return self.model.evaluate(x, y)
		except AttributeError as e:
			print("Model not found")
		return None

	def save_local_weights(self):
		weights = self.model.get_weights()
		np.save(f'weights_{self.client_id}', weights)
		# np.save('weights', weights)
		print("Saved Local weights")