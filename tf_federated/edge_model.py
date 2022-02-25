import os
from typing import Callable

from tensorflow.python.keras import models
import numpy as np
import tensorflow.python.keras.losses

from keras_utils import set_model_weights, get_rid_of_the_models


class Client:
	def __init__(self, client_id: int):
		self.client_id = client_id
		self.model: models.Model = None
		self.x_train = None
		self.y_train = None

	def _init_model(self, model_fn: Callable, model_weights=None):
		model = model_fn()
		if model_weights and os.path.isfile(model_weights):
			set_model_weights(model, model_weights)
		model.compile(
			loss=tensorflow.keras.losses.categorical_crossentropy,
			optimizer=tensorflow.keras.optimizers.Adadelta(),
			metrics=['accuracy']
			)
		self.model = model

	def receive_data(self, x, y):
		self.x_train = x
		self.y_train = y

	def receive_and_init_model(self, model_fn: Callable, model_weights):
		self._init_model(model_fn, model_weights)

	def edge_train(self, client_train_dict: dict):
		if self.model is None:
			raise ValueError("Model is not created for client: {0}".format(self.client_id))

		hist = self.model.fit(self.x_train, self.y_train, **client_train_dict)
		return hist

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
		np.save('weights', weights)
		print("Saved Local weights")