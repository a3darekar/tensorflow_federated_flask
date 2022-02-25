from flask import Flask, jsonify, request
from flask_socketio import SocketIO

edgeNodes, inactiveNodes, sid_mapper = {}, {}, {}
app = Flask(__name__)
socket = SocketIO(app)


@app.route('/')
def index():
	return "Aggregator initialized"
