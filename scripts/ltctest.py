import os

from scripts.confusionmatix import ConfusionMatrix
from scripts.dataloader import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp
from kerasncp import wirings
from kerasncp.tf import LTCCell
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

import confusionmatix
import dataloader

class ModelTest():
	""" Makes loading and testing models with a series of images & labels easier """
	
	def __init__(self, modelPath: str):
		self.model = None
		self.x = None
		self.y = None
		self.__x = None
		self.__y = None
		self.__count = 0
		self.prediction = None
		self.loadModel(modelPath)

	def loadModel(self, modelPath: str):
		modelDir = os.path.dirname(modelPath)
		self.model = keras.models.load_model(modelDir)
		return self.model

	def loadData(self, images: list, data: list):
		self.x, self.y = DataLoader().loadData(images, data)
		self.__count = self.y.shape[0]
		self.__x = tf.expand_dims(self.x, 0)

	def evaluate(self):
		print("")
		print("Trained evaluation")
		conv, pred = self.model.evaluate(self.__x, self.__y)
		print(conv, pred)
		print("Evaluation complete")
		print("")

	def predict(self):
		preds = np.empty((1, 4))
		totalCorrect = 0

		print("")
		print("Predicting starting")
		for i in range(self.__count):
			p = self.model.predict(self.__x[:, i:i+1, :, :, :])
			preds = np.vstack((preds, p.flatten()))
			
			if tf.math.top_k(p).indices.numpy() == tf.math.top_k(self.y[i]).indices.numpy():
				totalCorrect += 1

			print("Predicting", str(i + 1) + " of " + str(self.__count))
		print("Prediction complete")
		print("")
		
		preds = np.delete(preds, (0), axis=0)

		percentCorrect = (totalCorrect / len(preds)) * 100.0
		print("Accuracy: " + str(percentCorrect) + "%")

		print("")
		print("Creating confusion matrix")
		ConfusionMatrix(["Turn Left", "Move Forward", "Turn Right", "Stop/Do Nothing"], self.y, preds).plot()

def main():
	mt = ModelTest("./models/saved_models/t1_CNN3_4_345/")
	mt.loadData(["./images/archive/batch_test_1/colour/", "./images/archive/batch_test_2/colour/"],["./data/training_data_test_1.csv", "./data/training_data_test_2.csv"])
	mt.evaluate()
	mt.predict()

if __name__ == "__main__":
	main()
