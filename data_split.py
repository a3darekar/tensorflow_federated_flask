import pickle
import random

epochs = 5
batch_size = 4096
buffer = 100
pref_buffer = 10


def fetch_data():
	with open('data/x_data.txt', 'rb') as f:
		x_data = pickle.load(f)
	with open('data/y_data.txt', 'rb') as f:
		labels = pickle.load(f)
	return x_data, labels


def create_clients(data, labels, num_clients=3, initial='clients'):
	client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

	data = list(zip(data, labels))
	random.shuffle(data)

	size = len(data) // num_clients
	shards = [data[i: i + size] for i in range(0, size * num_clients, size)]
	print(f"Generated {len(shards)} equal parts of data")
	assert (len(shards) == num_clients)

	return {client_names[i]: shards[i] for i in range(num_clients)}


x_data, labels = fetch_data()
data = create_clients(x_data, labels, 4, 'node')
with open('data/data.txt', 'wb') as f:
	pickle.dump(data, f)
