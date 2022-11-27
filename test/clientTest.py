from client import Client

count = 0

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
s = Client(HOST, PORT)

def processRequest(data):
	print("Client recieved:", data)
	count = data

	if count > 250:
		s.stopConnection()

	print("Client sending:", count + 1)
	return count + 1

def main():
	s.subscribe(processRequest)
	s.connectServer()

# import socket
# import pickle

# def main():
# 	HOST = "127.0.0.1"  # The server's hostname or IP address
# 	PORT = 65432  # The port used by the server

# 	print("Starting")

# 	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 	count = 0
# 	try:
# 		s.connect((HOST, PORT))
# 		while count < 1:
# 			array = [1, 2, 3, 4, 5, 6, 7]
# 			#s.sendall(b"Hello, world " + str(count))
# 			print(pickle.dumps(array))
# 			s.sendall(pickle.dumps(array))
# 			data = s.recv(1024)
# 			print("Received " + str(data))
# 			count += 1
# 	except Exception as error:
# 		print(error)
# 	finally:
# 		s.shutdown(1)
# 		s.close()	

if __name__ == "__main__":
	main()