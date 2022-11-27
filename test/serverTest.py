
from server import Server

def processRequest(data):
	print("Server recieved:", data)
	print("Server sending:", data + 1)
	return data + 1

def main():
	HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
	PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

	s = Server(HOST, PORT, 2)
	s.subscribe(processRequest)
	s.startServer()

# import socket
# import pickle

# def main():
# 	HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
# 	PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

# 	print("Starting")

# 	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
# 		s.bind((HOST, PORT))
# 		s.listen()
# 		conn, addr = s.accept()
# 		with conn:
# 			print(f"Connected by {addr}")
# 			while True:
# 				data = conn.recv(262144)
# 				if data:
# 					print("Recieved - " + str(data))
# 					print("Recieved -", pickle.loads(data))
# 				if not data:
# 					break
# 				conn.sendall(data)

if __name__ == "__main__":
	main()