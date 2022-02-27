import os
import sys
import time
import pickle

import numpy as np
import socketio
from tensorflow.python.keras.utils.np_utils import to_categorical

from setup import NODE_ID, APP_URL, DATA_FILE, status
from tf_federated.edge_model import Client
from tf_federated.keras_utils import create_keras_model
from sklearn.model_selection import train_test_split

model = Client(NODE_ID)
sock = socketio.Client()
NODE_STATUS = status['init']


def assemble_data():
	return {'nodeID': NODE_ID, 'node_status': NODE_STATUS}


def evaluate_model(x, y):
	global NODE_STATUS
	NODE_STATUS = status['eval']
	return model.evaluate(x, y)


def fetch_model():
	message("fetch_model", "NONE")


@sock.event
def connect():
	global NODE_STATUS
	NODE_STATUS = status['idle']
	data = assemble_data()
	message('join', data)
	print("connected!")


def reconnect():
	print("Attempting Connection")
	try:
		sock.connect(APP_URL)
	except Exception as e:
		...


@sock.event
def connect_error(error_message):
	print("Problem establishing connection Trying in 5 seconds")
	try:
		time.sleep(5)
	except KeyboardInterrupt:
		print("KeyboardInterrupt occurred.")
		print("Disconnecting!")
		print('Session Interrupted by User. Program terminated')
		print("-----------------------------------Program terminated-----------------------------------")
		exit()
	reconnect()


@sock.event
def disconnect():
	global NODE_STATUS
	NODE_STATUS = status['down']
	print("Disconnected!")


@sock.on('fetch_model')
def receive_model(json):
	weights = []
	for key, value in json['model_variables'].items():
		weights.append(np.array(value))
	if len(weights) > 0:
		model.init_model(create_keras_model, weights)
	print("model fetched successfully!")


def message(event, data):
	sock.emit(event, data)


def await_reconnection_command():
	while True:
		reconnect_request = input(
			"Attempt reconnection? (y|Y): \t")
		if reconnect_request.strip() == 'Y' or 'y':
			return True
		elif reconnect_request.strip() == '0':
			print("-----------------------------------Program terminated-----------------------------------")
			exit(0)
		else:
			print("Invalid input")


def fetch_data():
	try:
		f = open(f"{DATA_FILE}", 'rb')
		data = pickle.load(f)
		data = data[f'node_{NODE_ID}']
		return np.array([t[0] for t in data]), np.array([t[1] for t in data])
	except OSError:
		print(f"Error accessing data. Please check if {DATA_FILE} file exists")   # return [list(t) for t in zip(*data[f'node_{NODE_ID}'])]
		sys.exit()


def init_model(_model):

	x_data, labels = fetch_data()
	x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33)
	y_train, y_test = to_categorical(y_train), to_categorical(y_test)
	_model.init_model(create_keras_model, model_weights=f"weights_{NODE_ID}.npy")
	_model.receive_data(x_train, y_train)
	global NODE_STATUS
	NODE_STATUS = status['idle']
	return _model, x_test, y_test


def run():
	global model
	model, x_test, y_test = init_model(model)
	try:
		if not NODE_ID:
			print("ERROR! Could not load Node identity")
			print("Initialize the Node identity with environment variable 'NODE_ID'. Look at README.txt for more info.")
			print("-----------------------------------Program terminated-----------------------------------")
			exit(0)
		while True:
			if not sock.connected:
				print("Attempting Connection")
				reconnect()

			else:
				print("Press Ctrl + C to terminate.")
				print("1. Fetch Global Model \n2. Evaluate on validation set.\n3. Train Model locally.\n4. Save Model locally.\n")
				task = input("Input Task to perform. \nType in (d|D) to disconnect: \n")
				if task.strip().lower() == 'd':
					sock.disconnect()
					await_reconnection_command()
				if task.strip().lower() == '1':
					print("Fetching Global Model")
					fetch_model()
				if task.strip().lower() == '2':
					score = evaluate_model(x_test, y_test)
					if score:
						print("Evaluation Accuracy: %.2f" % score[1])
				if task.strip().lower() == '3':
					training_dict = {'batch_size': 64, 'epochs': 10, 'verbose': 1, 'validation_split': 0.2}
					# training_dict = {'epochs': 10, 'validation_split': 0.33}
					hist = model.edge_train(client_train_dict=training_dict)
					print(f"Hist: {hist}")
				if task.strip().lower() == '4':
					model.save_local_weights()
				print("\n\n\n")
	except KeyboardInterrupt:
		print("KeyboardInterrupt occurred.")
		print('Session Interrupted by User.')
		print("-----------------------------------Program terminated-----------------------------------")
		exit(0)