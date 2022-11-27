import itertools
import matplotlib.pyplot as plt
import sklearn.metrics
import tensorflow as tf
import numpy as np

class ConfusionMatrix():
	""" A class to encapsulate the functions to plot a confusion matrix and display it """

	def __init__(self, classNames, trueLabels, predictedLabels):
		self.classNames = classNames
		self.trueLabels = trueLabels
		self.predictedLabels = predictedLabels

	def plot(self):
		cm = sklearn.metrics.confusion_matrix(tf.math.top_k(self.trueLabels).indices.numpy(), tf.math.top_k(self.predictedLabels).indices.numpy())

		print(cm)

		figure = plt.figure(figsize=(8, 8))
		plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
		plt.title("Confusion matrix")
		plt.colorbar()
		tick_marks = np.arange(len(self.classNames))
		plt.xticks(tick_marks, self.classNames, rotation=45)
		plt.yticks(tick_marks, self.classNames)

		# Compute the labels from the normalized confusion matrix.
		labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

		# Use white text if squares are dark; otherwise black.
		threshold = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			color = "white" if cm[i, j] > threshold else "black"
			plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()