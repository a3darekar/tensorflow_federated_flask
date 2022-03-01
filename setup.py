import os

NODE = str(os.getenv('NODE', 'EDGE')).lower()  # 'edge'
NODE_ID = int(os.getenv('NODE_ID', 1))  # 'edge'
if NODE != 'edge':
	PORT = 5000  # 'admin'
else:
	PORT = 5000 + NODE_ID  # 'edge'

DEBUG = bool(os.getenv('DEBUG', False))
DATA_FILE = str(os.getenv('DATA_FILE', False))
AGGR_MODEL_FILE = str(os.getenv('AGGR_MODEL_FILE', 'weights/weights.npy'))
INPUT_SHAPE = list(map(int, str(os.getenv('INPUT_SHAPE', False)).split()))
MODEL_WEIGHTS_FILE = str(os.getenv('MODEL_WEIGHTS_FILE', f"weights/weights_{NODE_ID}.npy"))
CLASS_COUNT = int(os.getenv('CLASS_COUNT', False))
APP_URL = str(os.getenv('ADMIN_URL', "http://127.0.0.1:5000"))


os.environ["CUDA_VISIBLE_DEVICES"] = ""

status = {
	'init': "INITIALIZING",
	'idle': "IDLE",
	'busy': "BUSY",
	'train': "TRAINING",
	'eval': "EVALUATING",
	'down': "DISCONNECTED"
}
