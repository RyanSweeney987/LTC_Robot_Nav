import socket
import pickle

# For use with Python 2.7

class Client():
	""" This class enables for data transfer to another instance of Python """

	__func = None

	def __init__(self, host, port, bufferSize):
		self.host = host
		self.port = port
		self.bufferSize = bufferSize
		self.isConnected = False

	def connectServer(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		retry = True
		counter = 0
		while counter < 100 and retry:
			retry = False

			try:
				print("Connecting to server")
				s.connect((self.host, self.port))
				self.isConnected = True

				print("Connected")
				# Initial send to the loop off
				print("Sending initial data")
				s.sendall(pickle.dumps((100)))
				print("Initial data sent")
				print("Entering connection loop")
				while self.isConnected:
					if self.__func != None:
						data = None

						try:
							pickledData = s.recv(self.bufferSize)
							data = pickle.loads(pickledData)
						except Exception as error:
							print("Pickle error:", error)

						# sends the data to the subscribed function and returns the result to be sent off
						dataToSend = self.__func(data)
						s.sendall(pickle.dumps(dataToSend)) # Recieve server response, process response, send reply
					else:
						s.sendall(pickle.dumps(100))
			except Exception as error:
				print("Connection error:", error)
				retry = True
			finally:
				print("Exiting connection loop")
				s.close()
				self.isConnected = False
				counter += 1

	def stopConnection(self):
		self.isConnected = False

	def hasSubscriber(self):
		return self.__func != None

	def subscribe(self, function):
		self.__func = function

	def unsubscribe(self):
		self.__func = None