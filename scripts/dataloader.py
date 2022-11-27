import numpy as np
import pandas as pd
import cv2

class DataLoader:
	""" A helper class to make the data loading easier """

	def __init__(self):
		pass

	def loadData(self, imageDirectories: list, labelFiles: list):
		""" Returns a tuple containing the dataset (images, labels) """
		dataset = list((None, None))
		for i in range(len(labelFiles)):
			data = pd.read_csv(labelFiles[i]).values
			labels = np.asarray(data[:, 3:7]).astype(np.int32)
			images = np.asarray(self.__loadImages(imageDirectories[i], data[:,0], False))

			dataset[0] = images if dataset[0] is None else np.vstack((dataset[0], images))
			dataset[1] = labels if dataset[1] is None else np.vstack((dataset[1], labels))

		return tuple(dataset)

	def loadImage(self, dir: str, filename: str):
		return self.__loadImages(dir, [filename])[0]

	def __loadImages(self, dir: str, filenames: list, resize: bool = False):
		images = []
		for filename in filenames:
			img = cv2.imread(dir + filename)
			img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA) if resize else img
			#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = img.astype(np.float32)/255.0
			images.append(img)
		return images