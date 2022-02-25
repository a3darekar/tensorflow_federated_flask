from flask import Flask, jsonify, request
from flask_socketio import SocketIO

edgeNodes, inactiveNodes, sid_mapper = {}, {}, {}
app = Flask(__name__)
socket = SocketIO(app)


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


""" 
	Flask server methods. Use browser to access each of these methods.
	'/' 		=> index method: Displays lists of active Edge nodes as well as inactive nodes.
"""


@app.route('/')
def index():
	registered_users = {'active': edgeNodes, 'inactive': inactiveNodes}
	return jsonify(registered_users)
