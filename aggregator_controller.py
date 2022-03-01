from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit

from tf_federated.aggregator_model import AggregatorModel
# from keras_utils import

edgeNodes, inactiveNodes, sid_mapper = {}, {}, {}
app = Flask(__name__)
socket = SocketIO(app)
aggregator = AggregatorModel()


@socket.on('join')
def welcome_call(json):
	json['sid'] = request.sid
	edge_node = int(json['nodeID'])
	sid_mapper.update({request.sid: edge_node})
	if edge_node in inactiveNodes.keys():
		inactiveNodes.pop(edge_node)
		edgeNodes[edge_node] = json
	edgeNodes[edge_node] = json
	return


@socket.on('disconnect')
def disconnected():
	nodeID = sid_mapper[request.sid]
	inactive_node = edgeNodes.get(nodeID)
	inactiveNodes.update({nodeID: inactive_node})
	edgeNodes.pop(nodeID)


def get_global_model():
	return aggregator.trainable_variables


@socket.on('fetch_model')
def fetch_model(*args, **kwargs):
	print("fetch_model")
	emit("fetch_model", get_global_model())


@socket.on('eval_results')
def eval_report_handler(json):
	edge_node = sid_mapper[request.sid]
	edgeNodes[edge_node].update({"eval_report": json})


""" 
	Flask server methods. Use browser to access each of these methods.
	'/' 		=> index method: Displays lists of active Edge nodes as well as inactive nodes.
"""


@app.route('/json')
def get_status():
	registered_users = {'active': edgeNodes, 'inactive': inactiveNodes, "sid_mapper": sid_mapper}
	return jsonify(registered_users)


@app.route('/', methods = ["GET", "POST"])
def index():
	if request.method == 'POST':
		for nodeID, node in edgeNodes.items():
			emit('evaluate_edge', "ping", namespace='/', to=node['sid'])
	return render_template('index.html')


@app.route('/eval')
def fetch_eval():
	for nodeID, node in edgeNodes.items():
		emit("evaluate_edge", "ping", namespace='/', to=node['sid'])
	return jsonify({"status": 200, "response": "success"})


@app.route('/send')
def send_global_model():
	model_data = get_global_model()
	for nodeID, node in edgeNodes.items():
		emit("fetch_model", model_data, namespace='/', to=node['sid'])
	return jsonify({"status": 200, "response": "success"})