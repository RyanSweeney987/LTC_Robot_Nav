import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp
from kerasncp import wirings
from kerasncp.tf import LTCCell
import models
from scripts.server import Server
import base64
import cv2
import sys

class AISystem:
	""" This class processes images and returns the movement """

	def __init__(self):
		self.model = None

	def loadModel(self, modelPath: str):
		print("Loading model")
		modelDir = os.path.dirname(modelPath)
		self.model = keras.models.load_model(modelDir)
		print("Model loaded")

	def evaluateImage(self, image: np.ndarray, labels: np.ndarray):
		data = self.prepareImage(image)
		return self.model.evaluate(data, labels)

	def predictImage(self, imgData):
		result = ([0, 0, 0, 0])

		# If the data is the starting data, return default result
		if imgData == 100:
			return result

		# Prepare the image from the recieved image data
		image = self.prepareImage(imgData)
		# Return an array of the prediction results
		result = np.asarray(self.model.predict(image)).flatten()
		return result

	def prepareImage(self, imgData):
		# Decode byte array
		decodedImg = base64.b64decode(imgData)
		# Create image buffer
		imgBuffer = np.frombuffer(decodedImg, dtype=np.uint8)
		# Create image from buffer
		image = cv2.imdecode(imgBuffer, flags=1)

		# Preview what is recieved
		cv2.imshow("Sever Image", image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
				self.isConnected = False
		
		# Turn colour values into linear values
		image = image.astype(np.float32)/255.
		# Prepare for model prediction
		data = tf.expand_dims([image], 0)
		return data


def main():
	serverIpAddress = "127.0.0.1"
	serverPort = 65432

	# Image consisting of 3 channels (byte per channel) and dimensions of 256 by 256 and multiplied to add more space
	bufferSize = 3 * 256 * 256 * sys.getsizeof(np.int8)
	server = Server(serverIpAddress, serverPort, 2, bufferSize)

	aiSystem = AISystem()
	aiSystem.loadModel("./models/saved_models/t1_CNN3_4_345/")

	server.subscribe(aiSystem.predictImage)
	server.startServer()

if __name__ == "__main__":
	main()