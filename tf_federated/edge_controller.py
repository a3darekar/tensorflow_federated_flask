import time
import pickle
import socketio
from setup import NODE_ID, APP_URL
from tf_federated.edge_model import Client
from keras_utils import create_keras_model
from sklearn.model_selection import train_test_split

model = Client(NODE_ID)
sock = socketio.Client()


def assemble_data():
	node_id = NODE_ID
	return {'nodeID': NODE_ID}


def evaluate_model(x, y):
	return model.evaluate(x, y)


def fetch_model():
	message("fetch_model", "NONE")


def fetch_data():
	with open('data/data.txt', 'rb') as f:
		data = pickle.load(f)
	return [list(t) for t in zip(*data[f'node_{NODE_ID}'])]


@sock.event
def connect():
	data = assemble_data()
	message('join', data)
	print("connected!")


def reconnect():
	print("Attempting Connection")
	try:
		sock.connect(APP_URL)
	except Exception as e:
		print(e)


@sock.event
def connect_error(error_message):
	print(error_message + " Trying to reconnect in 5 seconds")
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
	print("Disconnected!")


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
	with open('data/data.txt', 'rb') as f:
		data = pickle.load(f)
	return [list(t) for t in zip(*data[f'node_{NODE_ID}'])]


def run():
	x_data, labels = fetch_data()
	x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33)
	model._init_model(create_keras_model, model_weights='weights')
	model.receive_data(x_train, y_train)
	print(model.client_id)
	try:
		if not NODE_ID:
			print("ERROR! Could not load Node identity")
			print("Initialize the Node identity with environment variable 'NODE_ID'. Look at README.txt for more info.")
			print("-----------------------------------Program terminated-----------------------------------")
			exit(0)
		while True:
			if not sock.connected:
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
					print(f"Evaluation Accuracy: {score}")
				print("\n\n\n")
	except KeyboardInterrupt:
		print("KeyboardInterrupt occurred.")
		print('Session Interrupted by User.')
		print("-----------------------------------Program terminated-----------------------------------")
		exit(0)