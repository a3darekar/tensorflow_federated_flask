import time
import socketio
from setup import NODE_ID, APP_URL
from tensorflow_federated.edge_model import Client

model = Client(NODE_ID)
sock = socketio.Client()


def assemble_data():
	node_id = NODE_ID
	return {'nodeID': NODE_ID}


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
		reconnect_request = input("Press Ctrl + C to terminate. \t Attempt reconnection? (y|Y): \t")
		if reconnect_request.strip() == 'Y' or 'y':
			return True
		elif reconnect_request.strip() == '0':
			print("-----------------------------------Program terminated-----------------------------------")
			exit(0)
		else:
			print("Invalid input")


def run():
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
				task = input("Input Task to perform. \nType in (d|D) to disconnect: \n")
				if task.strip().lower() == 'd':
					sock.disconnect()
					await_reconnection_command()
	except KeyboardInterrupt:
		print("KeyboardInterrupt occurred.")
		print('Session Interrupted by User.')
		print("-----------------------------------Program terminated-----------------------------------")
		exit(0)