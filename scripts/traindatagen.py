import sys
import os, os.path
from typing import Type
import cv2
import numpy as np
import pandas as pd
from numpy.typing import _32Bit
from scripts.dataloader import DataLoader
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import snapshot
import tensorflow_hub as hub
from tensorflow import keras
# Import shared modules for wirings, datasets,...
import kerasncp as kncp
# Import framework-specific binding
from kerasncp.tf import LTCCell      # Use TensorFlow binding
import matplotlib.pyplot as plt
import concurrent.futures
import requests
import threading
import time
import pickle
from sklearn.model_selection import train_test_split
import csv

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")	



## Refactoring
class Bounds():
	""""""

	def __init__(self, xMin: float, xMax: float, yMin: float, yMax: float):
		self.xMin = xMin
		self.xMax = xMax
		self.yMin = yMin
		self.yMax = yMax
		self.width = xMax - xMin
		self.height = yMax - yMin
		self.area = self.width * self.height
		self.centroid = self.__getCentroid()

	def __getCentroid(self):
		centerX = (self.xMin + self.xMax) / 2
		centerY = (self.yMin + self.yMax) / 2
		return (int(centerX), int(centerY))

class ImageLabeller():
	""""""

	def __init__(self):
		self.objectDector = ObjectDetector()
	
	def labelImage(self, image, imageBounds: Bounds, rangeFromCentre: int, occupyPrecentage: int):
		""" Returns the label text, one hot list, prediction confidence score, person bounds """
		labels, boxes, scores = self.objectDector.detectObjects(image)
		
		result = ("Not moving", [0, 0, 0, 1], 1, Bounds(0, 0, 0, 0))

		#loop throughout the detections and place a box around it  
		for score, (yMin, xMin, yMax, xMax), label in zip(scores, boxes, labels):
			if score < 0.5 or label != "person":
				continue

			personBounds = Bounds(xMin, xMax, yMin, yMax)
			percentageOccupied = personBounds.area / imageBounds.area
			
			if percentageOccupied < occupyPrecentage:
				frameCentre = imageBounds.centroid[0]
				if personBounds.centroid[0] >= (frameCentre - rangeFromCentre) and personBounds.centroid[0] <= (frameCentre + rangeFromCentre):
					result = ("Moving forward", [0, 1, 0, 0], score, personBounds)
				elif personBounds.centroid[0] < (frameCentre - rangeFromCentre):
					result = ("Turning left", [1, 0, 0, 0], score, personBounds)
				elif personBounds.centroid[0] > (frameCentre + rangeFromCentre):
					result = ("Turning right", [0, 0, 1, 0], score, personBounds)
			break
		
		return result

	def drawLabellingBounds(self, image, imageBounds: Bounds, rangeFromCentre: int, objectBounds: Bounds):
		# Draw bounds around person
		cv2.rectangle(image, (objectBounds.xMin, objectBounds.yMax), (objectBounds.xMax, objectBounds.yMin), (0,255,0), -1)
		# Draw centroid of person bounds
		cv2.rectangle(image, (objectBounds.centroid[0] - 1, objectBounds.centroid[1] - 1), (objectBounds.centroid[0] + 1, objectBounds.centroid[1] + 1),(0,0,255), 1)
		# Draw seperation lines
		imgCentroidX = imageBounds.centroid[0]
		cv2.line(image, (int(imgCentroidX - rangeFromCentre), int(0)), (int(imgCentroidX - rangeFromCentre), int(imageBounds.yMax)), (0, 0, 255), 1)
		cv2.line(image, (int(imgCentroidX + rangeFromCentre), int(0)), (int(imgCentroidX + rangeFromCentre), int(imageBounds.yMax)), (0, 0, 255), 1)
		return image

class ObjectDetector():
	""""""

	def __init__(self):
		self.detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
		self.labels = pd.read_csv('./data/labels.csv',sep=';',index_col='ID')
		self.labels = self.labels['OBJECT (2017 REL.)']

	def detectObjects(self, image):
		#Is optional but i recommend (float convertion and convert img to tensor image)
		rgb_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
		#Add dims to rgb_tensor
		rgb_tensor = tf.expand_dims(rgb_tensor , 0)
		
		boxes, scores, classes, numDetections = self.detector(rgb_tensor)
		
		labels = classes.numpy().astype('int')[0]
		labels = [self.labels[i] for i in labels]
		boxes = boxes.numpy()[0].astype('int')
		scores = scores.numpy()[0]
		return (labels, boxes, scores)

class ImageCapture():
	""""""

	def __init__(self):
		self.capture = cv2.VideoCapture(0)
		self.frameSize = (np.int32(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), np.int32(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self.image = None

	def captureImage(self, width: int, height: int):
		ret, self.image = self.capture.read()		
		if(width != self.frameSize[0] or height != self.frameSize[1]):
			self.image = cv2.resize(self.image, (width, height))
		return self.image

	def release(self):
		self.capture.release()
##


def getCentroid(currentBox: np.ndarray):
	xMin = currentBox[0]
	yMin = currentBox[1]
	xMax = currentBox[2]
	yMax = currentBox[3]
	# halfWidth = (xMax - xMin) / 2
	# halfHeight = (yMax - yMin) / 2
	# centerX = xMin + halfWidth
	# centerY = yMin + halfHeight

	centerX = (xMin + xMax) / 2
	centerY = (yMin + yMax) / 2

	return np.int32((centerX, centerY))

def getPercentageOccupied(currentBox: np.ndarray, frameSize: np.ndarray):
	xMin = currentBox[0]
	yMin = currentBox[1]
	xMax = currentBox[2]
	yMax = currentBox[3]
	width = xMax - xMin
	height = yMax - yMin
	frameArea = frameSize[0] * frameSize[1]
	boxArea = width * height
	return (boxArea / frameArea) 

def getMove(currentBox: np.ndarray, frameSize: np.ndarray, moveRange: np.int32, stopArea: np.float64):
	area = getPercentageOccupied(currentBox, frameSize)

	result = ("Not moving", [0, 0, 0, 1])
	# The area suggests how close we are to the camera
	if area < stopArea:
		centerX, centerY = getCentroid(currentBox)
		frameCenterX = frameSize[0] / 2	
		range = frameSize[0] / moveRange
		halfRange = range / 2

		if centerX >= (frameCenterX - halfRange) and centerX <= (frameCenterX + halfRange):
			result = ("Moving forward", [0, 1, 0, 0])
		elif centerX < (frameCenterX - halfRange):
			result = ("Turning left", [1, 0, 0, 0])
		elif centerX > (frameCenterX + halfRange):
			result = ("Turning right", [0, 0, 1, 0])
	
	return result

def getCSVLineString(filename_col: str, filename_sin: str, movement: list, bounds: list):
	dataArray = np.append([filename_col, filename_sin, movement[0]], movement[1])
	dataArray = np.append(dataArray, [bounds[0], bounds[1], bounds[2], bounds[3]]) 
	return dataArray

def currentSecTime():
    return time.time_ns() / 1_000_000_000

def calcDataCounts(data: np.ndarray):
	total = np.sum(data)

	leftMov = data[0] / total
	forwardMov = data[1] / total
	rightMov = data[2] / total
	stopMov = data[3] / total
	return (total, leftMov, forwardMov, rightMov, stopMov)


# Weighted Exponential Moving Average
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6708545
def getWeightExpMovingAverage(movingAverage: np.matrix, currentBox: np.ndarray, weights: np.ndarray):
	if movingAverage.shape[0] == 0:
		return currentBox

	a = 2 / (movingAverage.shape[0] + 1)
	weightedAverage = getWeightedAverage(movingAverage, currentBox, weights)

	expWeightAverage = (a * currentBox) + ((1 - a) * weightedAverage)

	return expWeightAverage.astype(dtype=np.int32)
	
def getExonentialAverage(movingAverage: np.matrix, currentBox: np.ndarray):
	if movingAverage.shape[0] == 0:
		return currentBox

	a = 2 / (movingAverage.shape[0] + 1)
	prev = movingAverage[movingAverage.shape[0] - 1]
	
	expAverage = (a * currentBox) + ((1 - a) * prev)

	return expAverage.astype(dtype=np.int32)

def getWeightedAverage(movingAverage: np.matrix, currentBox: np.ndarray, weights: np.ndarray):
	if movingAverage.shape[0] == 0:
		return currentBox
	
	movingAverage = np.vstack((movingAverage[1:,:], currentBox))
	movingAverage = movingAverage.T * weights
	movingAverage = np.sum(movingAverage.T, axis=0)

	return movingAverage.astype(dtype=np.int32)

# Basic moving average
def getMovingAverage(movingAverage: np.matrix, currentBox: np.ndarray):
	if movingAverage.shape[0] == 0:
		return currentBox
	
	rowCount = movingAverage.shape[0]
	
	earliestBox = np.array(movingAverage[0,:] / rowCount).round(decimals=0)
	currentBox = np.array(currentBox / rowCount).round(decimals=0)
	previousBox = movingAverage[rowCount - 1,:]

	return (currentBox + previousBox - earliestBox).astype(dtype=np.int32)

def saveAllData(colFile: str, matFile: str, csvFile: str, rgbImg, matrixImg, csvData, saveCol: bool, saveMat: bool, saveCSV: bool):
	if saveCol:
		cv2.imwrite(colFile, rgbImg)
	if saveMat:
		cv2.imwrite(matFile, matrixImg)
	if saveCSV:
		saveCSVLine(csvFile, csvData)

def saveCSVLine(filename: str, data):
	with open(filename, "at", newline='') as file:
		writer = csv.writer(file)
		writer.writerow(data)

def loadExistingCSV(filename: str):
	data = np.asarray(pd.read_csv(filename))
	return data



def objectDetection():
	maxArraySize = 5
	boxArray = np.array(np.zeros((1, 4), dtype=np.int32))
	# The sum of weights for the weighted average
	weightDivider = np.arange(1, maxArraySize + 1, 1)[::-1]
	weightsSum = 0
	for i in weightDivider:
		weightsSum += i
	# Precalculation of the weights
	weights = np.array([])
	for value in weightDivider:
		weights = np.hstack((weights, [value / weightsSum]))

	# For use if detection is lost
	isDetected = False
	isLastKnown = False
	lastKnownTime = 5

	# Carregar modelos
	detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
	labels = pd.read_csv('./data/labels.csv',sep=';',index_col='ID')
	labels = labels['OBJECT (2017 REL.)']

	cap = cv2.VideoCapture(0)

	frameSize = [np.int32(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), np.int32(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
	print("Frame default resolution: ", frameSize[0], ", ", frameSize[1])

	width = 256
	height = 256
	convolutionSize = [width, height]

	# Current data
	csvData = loadExistingCSV("./data/training_data.csv")
	dataAmounts = np.sum(csvData[:, 3:7], axis=0)
	print(csvData.shape)
	total, forwardMov, leftMov, rightMov, stopMov = calcDataCounts(dataAmounts) if csvData.shape[0] > 1 else (0, 0, 0, 0, 0)
	print("")
	print("Current data stats")
	print("Data amounts", dataAmounts)
	print("Total", total)
	print("Forward Move", forwardMov)
	print("Left Move", leftMov)
	print("Right Move", rightMov)
	print("Stop Move", stopMov)
	print("")

	# For saving images
	saveImageEverySec = 0.25
	imageDirectory = "./images/cv2_images/"
	subDirectories = ["colour/", "single/"]
	currentTime = currentSecTime()

	# Write header for data file
	#saveCSVLine(["colour_image_dir", "box_image_dir", "label_text", "move_forward_val", "turn_left_val", "turn_right_val", "stop_val", "xmin", "ymin", "xmax", "ymax"])

	# Options
	drawGuidelines = True
	drawCenter = True
	saveImages = False
	printText = True
	fillBoxes = True
	drawRectangles = True
	useMovingAverage = True
	ensureEvenCount = False
	averagingMethod = 3
	moveRange = 4

	font = cv2.FONT_HERSHEY_SIMPLEX

	matrixView = None

	correctPred = 0
	totalPred = 0

	while(True):
		#Capture frame-by-frame
		ret, frame = cap.read()	
		#Resize to respect the input_shape
		inp = cv2.resize(frame, (width , height ))
		#Convert img to RGB
		#rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
		rgb = inp

		#Is optional but i recommend (float convertion and convert img to tensor image)
		rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

		#Add dims to rgb_tensor
		rgb_tensor = tf.expand_dims(rgb_tensor , 0)
		
		boxes, scores, classes, num_detections = detector(rgb_tensor)
		
		pred_labels = classes.numpy().astype('int')[0]
		
		pred_labels = [labels[i] for i in pred_labels]
		pred_boxes = boxes.numpy()[0].astype('int')
		pred_scores = scores.numpy()[0]

		cv2.imshow("Raw", rgb)

		#loop throughout the detections and place a box around it  
		for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
			if score < 0.5 or label != "person":
				continue
				
			score_txt = f'{100 * round(score,0)}'
			
			if useMovingAverage:
				currentBox = np.array([xmin, ymin, xmax, ymax])		
				# If we have reached the size limit, add the moving average whilst dropping the oldest
				if boxArray.shape[0] == maxArraySize:
					# Get the moving average of the box
					movingAverage = getWeightedAverage(boxArray, currentBox, weights) if averagingMethod == 1 else getExonentialAverage(boxArray, currentBox) if averagingMethod == 2 else getWeightExpMovingAverage(boxArray, currentBox, weights) if averagingMethod == 3 else getMovingAverage(boxArray, currentBox)
					#movingAverage = getMovingAverage(boxArray, currentBox)
					#movingAverage = getWeightedAverage(boxArray, currentBox, weights)
					#movingAverage = getExonentialAverage(boxArray, currentBox)
					#movingAverage = getWeightExpMovingAverage(boxArray, currentBox, weights)

					boxArray = np.vstack((boxArray[1:,:], movingAverage))

					xmin = movingAverage[0]
					ymin = movingAverage[1]
					xmax = movingAverage[2]
					ymax = movingAverage[3]
				else:
					# Otherwise just add the row to the array
					boxArray = np.vstack((boxArray, currentBox))
			
			# Get movement based on box locaiton
			movement = getMove([xmin, ymin, xmax, ymax], convolutionSize, moveRange, 0.5)

			# if useLTC:
			# 	pred = predict(model, rgb)
			# 	print(movement[0], movement[1])
			# 	totalPred += 1
			# 	m = movement[1]
			# 	if (m[0] == 1 and pred[0] > 0.85) or (m[1] == 1 and pred[1] > 0.85) or (m[2] == 1 and pred[2] > 0.85) or (m[3] == 1 and pred[3] > 0.85):
			# 		correctPred += 1

			# 	print("Total:", totalPred)
			# 	print("Successful rate:", correctPred / totalPred)

			displayBoxes = np.copy(rgb)
			imageCopy = np.copy(rgb)

			# Draws green square for positional data
			matrixView = cv2.rectangle(imageCopy, (0, 0), (convolutionSize[0], convolutionSize[1]), (0,0,0), -1)
			cv2.rectangle(matrixView, (xmin, ymax), (xmax, ymin), (0,255,0), -1)

			if printText:
				cv2.putText(displayBoxes, movement[0], (xmin + 5, ymin + 15), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

			if drawRectangles:
				displayBoxes = cv2.rectangle(displayBoxes, (xmin, ymax), (xmax, ymin), (0,255,0), -1 if fillBoxes else 1)

			if drawGuidelines:
				frameCenterX = np.int32(convolutionSize[0] / 2)
				frameCenterY = np.int32(convolutionSize[1] / 2)
				range = np.int32(convolutionSize[0] / 4)
				halfRange = np.int32(range / 2)
		
				leftColour = (0, 255, 0) if leftMov <= 0.25 else (0, 0, 255)
				rightColour = (0, 255, 0) if rightMov <= 0.25 else (0, 0, 255)
				forwardColour = (0, 255, 0) if forwardMov <= 0.25 else (0, 0, 255)
				stopColour = (0, 255, 0) if stopMov <= 0.25 else (0, 0, 255)

				cv2.line(displayBoxes, (frameCenterX, 0), (frameCenterX, convolutionSize[1]), forwardColour, 1)
				cv2.line(displayBoxes, (frameCenterX - halfRange, 0), (frameCenterX - halfRange, convolutionSize[1]), leftColour, 1)
				cv2.line(displayBoxes, (frameCenterX + halfRange, 0), (frameCenterX + halfRange, convolutionSize[1]), rightColour, 1)
				cv2.line(displayBoxes, (0, frameCenterY),  (convolutionSize[0], frameCenterY), stopColour, 1)

			if drawCenter:
				center = getCentroid(currentBox)
				cv2.rectangle(displayBoxes, (center[0] - 1, center[1] - 1), (center[0] + 1, center[1] + 1),(0,0,255), 1)

			if printText:
				cv2.putText(displayBoxes, label, (xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
				cv2.putText(displayBoxes, score_txt, (xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

				area = f'{round(getPercentageOccupied(currentBox, convolutionSize), 2)}'
				cv2.putText(displayBoxes, area, (xmin, ymax-25), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

			#Display the resulting frame
			cv2.imshow("Object Detection", displayBoxes)
			cv2.imshow("Matrix View", matrixView)

			if saveImages:
				newTime = currentSecTime()
				if (newTime - currentTime) > saveImageEverySec:
					filename_colour = "cv2_colour_" + str(total - 0) + ".png" # For the full colour image
					filename_single = "cv2_single_" + str(total - 0) + ".png" # For the bounding box image

					# The data array will consist of the following
					# [colour_image_dir, box_image_dir, label_text, move_forward_val, turn_left_val, turn_right_val, stop_val, xmin, ymin, xmax, ymax]
					csvLine = getCSVLineString(filename_colour, filename_single, movement, [xmin, ymin, xmax, ymax])

					total += 1
					currentTime = newTime

					if ensureEvenCount:
						if True if movement[0] == "Turning left" and leftMov <= 0.25 else (True if movement[0] == "Turning right" and rightMov <= 0.25 else (True if movement[0] == "Moving forward" and forwardMov <= 0.25 else (True if movement[0] == "Not moving" and stopMov <= 0.25 else False))):
							print("")
							print("Saving: images - " + str(total))
							print(csvLine)
							saveAllData(imageDirectory + subDirectories[0] + filename_colour, 
								imageDirectory + subDirectories[1] + filename_single, 
								"./data/temp.csv", 
								rgb, 
								matrixView, 
								csvLine, 
								False, False, False)
							
							dataAmounts = np.sum(np.vstack((dataAmounts, movement[1])), axis=0)
							print(dataAmounts)
							total, leftMov, forwardMov, rightMov, stopMov = calcDataCounts(dataAmounts)
							print(total, leftMov, forwardMov, rightMov, stopMov)
					else:
						print("")
						print("Saving: images - " + str(total))
						print(csvLine)
						saveAllData(imageDirectory + subDirectories[0] + filename_colour, 
							imageDirectory + subDirectories[1] + filename_single, 
							"./data/temp.csv", 
							rgb, 
							matrixView, 
							csvLine, 
							False, False, False)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def main():
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
	print("OpenCV version:", cv2.__version__)

	# Capture a single frame from the webcam
	#ic = ImageCapture()
	#image = ic.captureImage(256, 256)
	#ic.release()

	# Load a saved image
	dl = DataLoader()
	image = dl.loadImage("./images/figures/", "cv2_colour_166.png")

	cv2.imshow("Base Image", image)
	cv2.waitKey(1)

	imageBounds = Bounds(0, 256, 0, 256)
	range = 256 / 8
	percentage = 0.5

	im = ImageLabeller()
	labelText, val, score, personBounds = im.labelImage(image, imageBounds, range, percentage)
	
	boundsImage = np.copy(image)
	boundsImage = im.drawLabellingBounds(boundsImage, imageBounds, range, personBounds)

	cv2.imshow("Bounds Image", boundsImage)
	cv2.waitKey(1)

	cv2.imwrite("./images/figures/cv2_colour_166_boundries.png", boundsImage)

	#input("Press Enter to continue...")

	#objectDetection()

if __name__ == "__main__":
	main()