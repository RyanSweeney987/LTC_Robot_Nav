#serial number ap990236i02y63100735 or ap990236l02y63100735

from inaoqi import ALMemoryProxy
from naoqi import ALModule, ALProxy
from scripts.client import Client
from scripts.commandprocessor import CommandProcessor
import copy
import cv2
import numpy as np
import vision_definitions as vd
from PIL import Image
import base64
import sys
import time

def handleVoiceCommand(value):
	""" Mandatory docstring. comment needed to create a bound method """
	print("-----", value)
	if value[2] == "modifiable_grammar":
		phrase = value[0]
		confidence = value[1]


	#result = self.commandPorcessor.processWord2(value)
	# if stop
	# call pepper stop
	# if turn
	# 	if right
	#		call pepper move
	#	if left
	#		call pepper move
	# if move
	#	if forward
	#		call pepper move
	#	if back
	#		call pepper move
	# if follow
	#	call pepper following?????

class PepperMiddleware():
	""" A middleware class to handle communications between the pepper robot and outside """

	def __init__(self, pepperConnection, acceptancePercentage):
		self.pepperConnection = pepperConnection
		# self.commandPorcessor = commandProcessor
		self.acceptancePercentage = acceptancePercentage
		self.errorCounter = 0
		self.previousPred = None

	def setPepperConnection(self, pepperConnection):
		self.pepperConnection = pepperConnection

	# def setCommandProcessor(self, commandProcessor):
	# 	self.commandPorcessor = commandProcessor

	def handleServerRequest(self, data):
		pred = data if data is not None else [0., 0., 0., 1.]
		#print("Prediction:", pred)

		# 0.4 is just under 39 degrees
		if pred[0] > 0.5: # turn left
			self.pepperConnection.move(0.0, 0.0, 0.4, True)
			print("Turning left", pred)
		elif pred[1] > 0.75: # move forward
			self.pepperConnection.move(0.5, 0.0, 0.0, True)
			print("Moving forward", pred)
		elif pred[2] > 0.5: # turn right
			self.pepperConnection.move(0.0, 0.0, -0.4, True)
			print("Turning right", pred)
		elif pred[3] > self.acceptancePercentage: # stop
			#self.pepperConnection.stop()
			print("Stopping")
		else: # If the prediction was poor, count the errors, if there's too many just stop
			maxIndex = np.argmax(pred)
			movements = ["turn left", "move forward", "turn right", "stop/do nothing"]
			print("Error - top value: " + str(pred[maxIndex]) + " - " + movements[maxIndex])

		# When movement is complete, get next frame and process it
		image = cv2.cvtColor(self.pepperConnection.getVideoFrame(), cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (256, 256))

		# Convert to jpg
		isSuccess, imgBuffer = cv2.imencode(".jpg", image)
		# Convert to base64 encoding
		imgJpg = base64.b64encode(imgBuffer)

		# Return base64 encoded jpg
		return imgJpg

class PepperConnection():
	""" The primary class for controlling the Pepper Robot """

	def __init__(self):
		self.isConnected = False
		# Memory
		self.memoryService = None
		# Speech recognition
		self.speechRecog = None
		self.speechRecogService = None
		# Video
		self.cameras = None
		self.videoSession = None
		self.videoService = None
		self.videoHandle = None
		# Navigation
		#self.navigation = None
		#self.motion = None
		self.navigationService = None
		self.motionService = None
		# Posture
		self.postureService = None
		# Text to speech
		self.textToSpeech = None
		# 
		self.voiceSub = None
		# Qi session for the robot connection
		self.session = None
		# Awareness
		self.awarenessService = None

	def connectRobot(self, ipAddress, port):
		print("Connecting to Pepper")
		self.__initMemory(ipAddress, port)
		#self.__initSpeechRecog()
		self.__initCameras()
		self.__initNavigation()
		self.__initPosture()
		self.__initAwareness()
		self.isConnected = True

	def setRobotVocabulary(self, vocabulary):
		# self.speechRecog.pause(True)
		# try:
		# 	self.speechRecog.setVocabulary(vocabulary, False)
		# except Exception as e:
		# 	print("Error updating vocabulary: ", e)
		# finally:
		# 	self.speechRecog.pause(False)
		return
		self.speechRecogService.pause(True)
		try:
			self.speechRecogService.setVocabulary(vocabulary, False)
		except Exception as e:
			print("Error updating vocabulary: ", e)
		finally:
			self.speechRecogService.pause(False)

	def getVideoFrame(self):
		imgTemp = None
		while(imgTemp is None):
			# Get the image from storage
			imgTemp = self.videoService.getImageRemote(self.videoHandle)
		# Copy the image to local
		image = copy.deepcopy(self.videoService.getImageRemote(self.videoHandle))
		# Release the image from storage
		self.videoService.releaseImage(self.videoHandle)
		# Convert the copied image into a more usable format and return it
		byteString = image
		imageWidth = byteString[0]
		imageHeight = byteString[1]
		byteString = str(bytearray(byteString[6]))
		image = np.asarray(Image.frombytes("RGB", (imageWidth, imageHeight), byteString))
		return image
	
	def move(self, x, y, theta, wait):
		""" x is forward vector, y is right vector, theta is +-PI, wait is whether this is a blocking call """

		if self.awarenessService.isRunning() or self.awarenessService.isEnabled():
			self.awarenessService.setEnabled(False)

		#move = self.motion.move(x, y, theta)

		#self.motionService.wakeUp()
		#self.motionService.moveToward(x, y, theta, [["Frequency", 1.0]])

		result = self.motionService.moveTo(x, y, theta)
		
		#result = self.navigationService.navigateTo(x, y, theta)
		print("Move result", result)
		# if wait:
		# 	self.motion.waitUntilMoveIsFinished()

		#return move

	def stop(self):
		self.motionService.stopMove()
		#return self.motion.stopMove()

	def unscubscribeAll(self):
		self.cameras.unsubscribe(self.videoHandle)

	def __initMemory(self, ipAddress, port):
		memory = ALProxy("ALMemory", ipAddress, port)

		self.session = memory.session()
		self.memoryService = self.session.service("ALMemory")

		self.voiceSub = self.memoryService.subscriber("WordRecognizedAndGrammar")
		self.voiceSub.signal.connect(handleVoiceCommand)	

		print("Memory connected!")

	def __initSpeechRecog(self):
		#self.speechRecog = ALProxy("ALSpeechRecognition", ipAddress, port)
		#self.speechRecog.setLanguage("English")
		
		self.speechRecogService = self.session.service("ALSpeechRecognition")
		self.speechRecogService.setLanguage("English")
		print("Speech recognition connected!")
		
	def __initCameras(self):
		# top camera, 640x480, BBGGRR, 30Fps
		self.videoService = self.session.service("ALVideoDevice")
		self.videoHandle = self.videoService.subscribe("video_service", vd.kVGA, vd.kRGBColorSpace, 30)
		print("Camera connected!", self.videoHandle)

	def __initNavigation(self):
		self.navigationService = self.session.service("ALNavigation")
		self.motionService = self.session.service("ALMotion")
		self.motionService.stiffnessInterpolation("Body", 1.0, 1.0)
		#self.motionService.wbEnableEffectorControl("Body", False)
		print("Navigation connected!")

	def __initPosture(self):
		self.postureService = self.session.service("ALRobotPosture")
		self.postureService.goToPosture("StandInit", 1.0)
		print("Posture connected!")

	def __initAwareness(self):
		self.awarenessService = self.session.service("ALBasicAwareness")
		self.awarenessService.pauseAwareness()
		self.awarenessService.setEnabled(False)
		print("Awareness connected!")

class PepperOffline():
	""" A standing of the Pepper robot that simulates the results """
	
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.isConnected = False
		self.imgWidth = 256
		self.imgHeight = 256

	def connectRobot(self, ipAddress, port):
		print("Simulating connecting to Pepper")
		self.isConnected = True

	def setRobotVocabulary(self, vocabulary):
		# Set the voice recognition values
		print("Simulating setting vocabulary")

	def getVideoFrame(self):
		#Capture frame-by-frame
		print("Getting video frame")
		ret, frame = self.cap.read()	
		image = copy.deepcopy(frame)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		cv2.imshow("Robot Image", image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
				self.isConnected = False

		return image
	
	def move(self, x, y, theta, wait):
		print("Moving: ", x, ", ", y, ", ", theta, ", should wait: ", wait)
		return True

	def stop(self):
		print("Stopping")
		return True

	def unscubscribeAll(self):
		print("Unsubscribing")

	def talk(self, text):
		print("Saying: ", text)


##### Load and test using my webcam
##### Load up voice recognition AI and use that instead

def main():
	pepperIpAddress = "169.254.15.73"
	pepperPort = 9559
	serverIpAddress = "127.0.0.1"
	serverPort = 65432

	pepper = PepperConnection()
	#pepper = PepperOffline()
	pepper.connectRobot(pepperIpAddress, pepperPort)
	#pepper.setRobotVocabulary(["move forward", "move back", "turn left", "turn right", "stop", "follow"])

	print("Setting up middleware")
	middle = PepperMiddleware(pepper, 0.75)

	useServer = True
	# Array of 4 elements of 4 bytes each element
	bufferSize = sys.getsizeof(np.int32) * 4
	print("Setting up client socket")
	server = Client(serverIpAddress, serverPort, bufferSize)
	server.subscribe(middle.handleServerRequest)
	if useServer:
		server.connectServer()

	isRunning = True

	while isRunning:
		if not pepper.isConnected:
			server.stopConnection()
			isRunning = False

		if useServer:
			if not server.isConnected:
				isRunning = False
		else:
			image = middle.handleServerRequest([0, 0, 0, 0])

			cv2.imshow("Pepper View", image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				isRunning = False

if __name__ == "__main__":
	main()