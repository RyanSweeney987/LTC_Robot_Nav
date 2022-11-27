import numpy as np
from numpy.typing import _32Bit
import pandas as pd
import matplotlib.pyplot as plt
import os, os.path
import time

def currentSecTime():
    return time.time_ns()/1000000000

# Weighted Exponential Moving Average
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6708545
def getWeightExpMovingAverage(movingAverage: np.ndarray, currentBox, weights):
	a = 2 / (movingAverage.size + 1)
	weightedAverage = getWeightedAverage(movingAverage, currentBox, weights)

	expWeightAverage = (a * currentBox) + ((1 - a) * weightedAverage)

	return expWeightAverage
	
def getExonentialAverage(movingAverage: np.ndarray, currentBox):	
	a = 2 / (movingAverage.size + 1)
	prev = movingAverage[movingAverage.size - 1]

	expAverage = (a * currentBox) + ((1 - a) * prev)

	return expAverage

def getWeightedAverage(movingAverage: np.ndarray, currentBox, weights):
	movingAverage = np.append(movingAverage[1:], currentBox)
	
	for index, average in enumerate(movingAverage):
		movingAverage[index] = average * weights[index]

	return np.sum(movingAverage)

# Basic moving average
def getMovingAverage(movingAverage: np.ndarray, currentBox):
	rowCount = movingAverage.size
	
	earliestBox = np.array(movingAverage[0] / rowCount)
	currentBox = np.array(currentBox / rowCount)
	previousBox = movingAverage[rowCount - 1]

	return (currentBox + previousBox - earliestBox)

def main():
	saveImageEverySec = 1
	imageCount = 0
	currentTime = currentSecTime()

	while(imageCount < 20):
		newTime = currentSecTime()
		#print("New time: " + str(newTime))
		#print("Diff: " + str(newTime - currentTime))
		if (newTime - currentTime) > saveImageEverySec:
			print("Running save - " + str(imageCount))
			currentTime = newTime
			imageCount = imageCount + 1

	return
	data = np.asarray(pd.read_csv("./data/tempurature.csv")["Temp"])
	indeces = np.arange(0, data.shape[0], 1)

	maxArraySize = 10
	movingAverageArray = np.zeros(maxArraySize)

	# The sum of weights for the weighted average
	weightDivider = np.arange(1, maxArraySize + 1, 1)[::-1]
	weightsSum = 0
	for i in weightDivider:
		weightsSum += i
	weights = np.asarray([])
	for value in weightDivider:
		weights = np.append(weights, value / weightsSum)

	sma = np.asarray([])
	wma = np.asarray([])
	ema = np.asarray([])
	wema = np.asarray([])

	for value in data:
		movingAverage = getMovingAverage(movingAverageArray, value)
		movingAverageArray = np.append(movingAverageArray[1:], movingAverage)
		sma = np.append(sma, movingAverage)

	movingAverageArray = np.zeros(maxArraySize)

	for value in data:
		movingAverage = getWeightedAverage(movingAverageArray, value, weights)
		movingAverageArray = np.append(movingAverageArray[1:], movingAverage)
		wma = np.append(wma, movingAverage)

	movingAverageArray = np.zeros(maxArraySize)

	for value in data:
		movingAverage = getExonentialAverage(movingAverageArray, value)
		movingAverageArray = np.append(movingAverageArray[1:], movingAverage)
		ema = np.append(ema, movingAverage)

	movingAverageArray = np.zeros(maxArraySize)

	for value in data:
		movingAverage = getWeightExpMovingAverage(movingAverageArray, value, weights)
		movingAverageArray = np.append(movingAverageArray[1:], movingAverage)
		wema = np.append(wema, movingAverage)

	plt.figure(figsize=(12, 8), dpi=80)
	plt.plot(indeces, data, '-')
	#plt.plot(indeces, sma, "-")
	plt.plot(indeces, wma, "-")
	#plt.plot(indeces, ema, "-")
	#plt.plot(indeces, wema, "-")
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.xlabel("Time")
	plt.ylabel("Temperature")
	plt.tick_params(axis='x', which='major', labelsize=10)
	plt.show()

if __name__ == "__main__":
	main()