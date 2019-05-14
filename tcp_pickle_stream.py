import pickle
import struct
import socket


class listener:

    def __init__(self, host='', port=8089):
        """
        Listener class. Starts a listening TCP socket to receive pickled images (server on protocol level).

        Args:
          host: String
           IP address.
          port: int
           Port.
        """
        self.host = host
        self.port = port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen()
        print("listening on", (self.host, self.port))

        try:
            print("waiting for connection ...")
            self.connection = self.socket.accept()[0].makefile('rb')
        except KeyboardInterrupt:
            print("terminated by user, close stream")
            self.close()

    def get_frame(self):
        msg_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        data = self.connection.read(msg_len)
        frame = pickle.loads(data)
        return frame

    def close(self):
        self.connection.close()
        self.connection = None
        self.socket.close()
        self.socket = None


class streamer:

    def __init__(self, host, port=8089):
        """
        Streamer class. Starts a streaming TCP socket to send pickled images (client on protocol level).

        Args:
          host: String
           IP address.
          port: int
           Port.
        """
        self.host = host
        self.port = port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("trying to connect ...")
        self.socket.connect((self.host, self.port))
        self.connection = self.socket.makefile('wb')
        print("connected to", (self.host, self.port))

    def send_frame(self, frame):
        data = pickle.dumps(frame)
        self.connection.write(struct.pack('<L', len(data)))
        self.connection.flush()
        self.connection.write(data)
        self.connection.flush()

    def close(self):
        self.connection.close()
        self.connection = None
        self.socket.close()
        self.socket = None