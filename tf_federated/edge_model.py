from typing import Callable

from tensorflow.python.keras import models


class Client:
	def __init__(self, client_id: int):
		self.client_id = client_id
		self.model: models.Model = None
		self.x_train = None
		self.y_train = None

	def _init_model(self, model_fn: Callable, model_weights=None):
		model = model_fn()
		if model_weights:
			fed_learn.set_model_weights(model, model_weights)
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
		fed_learn.get_rid_of_the_models(self.model)