import socket
import pickle

# For use with Python 3.10

class Server():
	""" This class enables for data transfer to another instance of Python """

	__func = None

	def __init__(self, host: str, port: int, pickleProtocol: int, bufferSize: int):
		self.host = host
		self.port = port
		self.pickleProtocol = pickleProtocol
		self.bufferSize = bufferSize
		self.isConnected = False

	def startServer(self):
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			print("Starting server")
			s.bind((self.host, self.port))
			s.listen()
			conn, addr = s.accept()
			with conn:
				self.isConnected = True

				print("Entering connection loop")
				while self.isConnected:
					data: None

					try:
						dataRecieved = conn.recv(self.bufferSize)
						data = pickle.loads(dataRecieved, encoding="bytes")
					except Exception as error:
						print("Pickle error:", error)

					if data:
						if self.__func:
							# Takes the sent data, unpacks it, processes it and then returns the result
							dataToSend = self.__func(data)
							#print("Data to send:", dataToSend)
							conn.sendall(pickle.dumps(dataToSend, protocol=self.pickleProtocol))
							continue
					if not data:
						self.isConnected = False
						conn.close()
						print("Connection ended")
						break

					conn.sendall(pickle.dumps(100))

				print("Exiting connection loop")

	def hasSubscriber(self):
		return self.__func != None

	def subscribe(self, function):
		self.__func = function

	def unsubscribe(self):
		self.__func = None