from setup import NODE, DEBUG, PORT, NODE_ID

if NODE != 'edge':
	print(f"STARTING AGGREGATOR NODE ON PORT: {PORT}")
	from aggregator_controller import socket, app
	socket.run(app, debug=DEBUG)
else:
	print(f"STARTING EDGE NODE: {NODE_ID}")
	from edge_controller import run
	run()
