"""Server to send profiler events

2019 Colin Dietrich"""

import time
import socket
import atexit

import config


class PowerProfiler:
    def __init__(self):
        self.server_ip = config.server_ip
        self.server_port = config.server_port
        self.s = None
        self.conn = None
        self.addr = None

    def serve(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((config.server_ip, config.server_port))
        self.s.listen(1)

        self.conn, self.addr = self.s.accept()
        atexit.register(self.s.close)

    def recv(self):
        data = self.conn.recv(config.buffer_size)
        return data


if __name__ == "__main__":
    pp = PowerProfiler()
    pp.serve()
    while True:
        with open('profile_output.txt', 'w') as f:
            x = pp.recv()
            x = x.decode('utf-8')
            if 'profile' in x:
                d = "{},{}".format(time.ctime(), x)
                print(d)            
            if 'profile_end' in x:
                pp.s.close()                
                break
            
    pp.s.close()

