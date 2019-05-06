"""Simple Client to sent events 
Colin Dietrich 2019"""

import socket
import atexit

import config


class Telemetry:
	def __init__(self, server_ip):
		
		self.server_ip = server_ip
		self.server_port = config.server_port
		self.s = None

	def connect(self):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.connect((self.server_ip, self.server_port))
		atexit.register(self.s.close)

	def send(self, m, verbose=False):
		self.s.send(m.encode('utf-8'))

	def recv(self):
		return self.s.recv(config.buffer_size)
		
if __name__ == "__main__":
	import time
	t = Telemetry(server_ip = '192.168.86.47')
	t.connect()
	for n in range(10):
		#t.send("Counter value: {}".format(n))
		t.send("Prediction Start")
		t.send("Prediction End")
		time.sleep(0.1)
	t.send('done_transmit')
