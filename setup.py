import os

DEBUG = bool(os.getenv('DEBUG', False))
INPUT_SHAPE = bool(os.getenv('INPUT_SHAPE', False))
CLASS_COUNT = bool(os.getenv('CLASS_COUNT', False))
APP_URL = str(os.getenv('ADMIN_URL', "http://127.0.0.1:5000"))

NODE = str(os.getenv('NODE', 'EDGE')).lower()  # 'edge'
NODE_ID = int(os.getenv('NODE_ID', 1))  # 'edge'
if NODE != 'edge':
	PORT = 5000  # 'admin'
else:
	PORT = 5000 + NODE_ID  # 'edge'
