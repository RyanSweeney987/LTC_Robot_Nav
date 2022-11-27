import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp
from kerasncp import wirings
from kerasncp.tf import LTCCell
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds
import cv2
from sklearn.model_selection import train_test_split
import datetime
import models
# LTC is fundamentally an RNN

def drawWiringModel(wirings):
	sns.set_style("white")
	plt.figure(figsize=(12, 12))
	legend_handles = wirings.draw_graph(layout='shell',neuron_colors={"command": "tab:cyan"})
	plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
	sns.despine(left=True, bottom=True)
	plt.tight_layout()
	plt.show()

def loadImages(dir: str, filenames: list, resize: bool):
	images = []
	for filename in filenames:
		img = cv2.imread(dir + filename)
		img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA) if resize else img
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.astype(np.float32)/255.0
		images.append(img)
	return images

def printDataInfo(data: np.ndarray):
	moveForwardCount = np.count_nonzero(data == "Moving forward", axis=0)[2]
	turnLeftCount = np.count_nonzero(data == "Turning left", axis=0)[2]
	turnRightCount = np.count_nonzero(data == "Turning right", axis=0)[2]
	stopCount = np.count_nonzero(data == "Not moving", axis=0)[2]

	print("Data count")
	print("Move forward: " + str(moveForwardCount))
	print("Turn left: " + str(turnLeftCount))
	print("Turn right: " + str(turnRightCount))
	print("Stop Moving: " + str(stopCount))
	print("\n")

def main():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	print("Num GPUs: ", len(gpus))
	if gpus:
		# Restrict TensorFlow to only allocate 7GB of memory on the first GPU
		try:
			tf.config.set_visible_devices([], 'GPU')
			tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
			#tf.config.experimental.set_memory_growth(gpus[0], True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Virtual devices must be set before GPUs have been initialized
			print(e)

	tf.compat.v1.RunOptions(
		report_tensor_allocations_upon_oom = True
	)

	drawWiring = True
	loadCheckpoint = False
	saveCheckpoints = True
	logForTensorboard = True
	earlyStop = False
	saveModel = True

	#dataA = pd.read_csv("./data/training_data_1.csv").values
	#dataB = pd.read_csv("./data/training_data_2.csv").values
	dataC = pd.read_csv("./data/training_data_3.csv").values
	dataD = pd.read_csv("./data/training_data_4.csv").values
	dataE = pd.read_csv("./data/training_data_5.csv").values


	#yA = np.asarray(dataA[:,3:7]).astype(np.int32)
	#yB = np.asarray(dataB[:,3:7]).astype(np.int32)
	yC = np.asarray(dataC[:,3:7]).astype(np.int32)
	yD = np.asarray(dataD[:,3:7]).astype(np.int32)
	yE = np.asarray(dataE[:,3:7]).astype(np.int32)
	#print(yC.shape)
	#y = np.vstack((y, yB))
	#y = np.vstack((y, yC))
	y = np.vstack((yC, yD))
	y = np.vstack((y, yE))
	#y = yE

	#xA = np.asarray(loadImages("./images/archive/batch_1/colour/", dataA[:,0], False))
	#xB = np.asarray(loadImages("./images/archive/batch_2/colour/", dataB[:,0], False))
	xC = np.asarray(loadImages("./images/archive/batch_3/colour/", dataC[:,0], False))
	xD = np.asarray(loadImages("./images/archive/batch_4/colour/", dataD[:,0], False))
	xE = np.asarray(loadImages("./images/archive/batch_5/colour/", dataE[:,0], False))
	#print(xC.shape)
	#x = np.vstack((x, xB))
	#x = np.vstack((x, xC)) 
	x = np.vstack((xC, xD)) 
	x = np.vstack((x, xE)) 
	#x = xE

	#[ 19 155  52  70] 3 = 296
	#[136 277 191 221] all = 825
	#[ 85 221 118 134] 1 & 3 = 558
	#[ 70 211 125 157] 2 & 3 = 563
	columnCounts = np.sum(y, axis=0)
	print("")

	minValue = np.min(columnCounts)
	minIndex = np.where(columnCounts == minValue)

	print(columnCounts, minValue, minIndex)

	print("")
	print("X Shape", x.shape)
	print("Y Shape", y.shape)
	print("")

	x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

	x_train = tf.expand_dims(x_train, 0)
	x_valid = tf.expand_dims(x_valid, 0)

	y_train = tf.expand_dims(y_train, 0)
	y_valid = tf.expand_dims(y_valid, 0)

	print("")
	print("x_valid", str(x_valid.shape))
	print("y_valid", str(y_valid.shape))
	print("x_train", str(x_train.shape))
	print("y_train", str(y_train.shape))
	print("")

	inputs = (None, x_train.shape[2], x_train.shape[3], x_train.shape[4])
	print(inputs)

	#model, wirings = getVGG19Model(inputs)
	#model, wirings = getFFCNN(inputs)
	#model, wirings = models.getCNNModel1(inputs)
	#model, wirings = getCNNModel1_2(inputs)
	#model, wirings = getCNNModel1_3(inputs)
	##model, wirings = getCNNModel2(inputs) no good
	#model, wirings = models.getCNNModel3(inputs)
	model, wirings = models.getCNNModel3_4(inputs)
	#model, wirings = getCNNModel4(inputs)
	#model, wirings = getCNNModel5(inputs)
	#model, wirings = getCNNModel6(inputs)

	model.summary(line_length=100)

	if drawWiring:
		drawWiringModel(wirings)

	return

	#training_modelname_iteration_batches 
	#iteration for each adjustment
	#t1_CNN1_1_23
	#t1_CNN3_1
	#t1_CNN4_1
	#t1_FFCNN_1
	#t1_VGG16_1
	#t1_VGG19_1

	modelTrainingName = "t3_CNN3_1_5"

	# Training checkpoints
	checkpointPath = "./models/training_checkpoints/" + modelTrainingName + "/cp-{epoch:04d}.ckpt"
	checkpointDir = os.path.dirname(checkpointPath)
	
	if loadCheckpoint:
		print("")
		print("Checking for checkpoint in", checkpointDir)
		# Load the previously saved weights
		latest = tf.train.latest_checkpoint(checkpointDir)
		if latest:
			print("Checkpoint found. Loading checkpoint")
			print(latest)
			model.load(latest)
			print("Checkpoint loaded")
		else:
			print("No checkpoint found")
		print("")

	# Check performance on the validation before training
	print("")
	print("Pre-trained evaluation")
	#loss, acc = model.evaluate(x_valid, y_valid)
	#print(loss, acc)
	#loss, acc, auc = model.evaluate(x_valid, y_valid)
	#print(loss, acc, auc)
	#loss, truePos, trueNeg, falsePos, falseNeg = model.evaluate(x_valid, y_valid)
	#print(loss, truePos, trueNeg, falsePos, falseNeg)
	loss, acc, auc, truePos, trueNeg, falsePos, falseNeg = model.evaluate(x_valid, y_valid)
	print(loss, acc, auc, truePos, trueNeg, falsePos, falseNeg)
	print("Evaluation complete")
	print("")

	callbacksList = []
	if saveCheckpoints:
		cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointDir, save_weights_only=True, verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)
		callbacksList.append(cb)

	if earlyStop:
		cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", min_delta=1, patience=100)
		callbacksList.append(cb)

	if logForTensorboard:
		log_dir = "./models/logs/fit/" + modelTrainingName + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		callbacksList.append(cb)
	
	# Train the model
	hist = None
	print("")
	print("Training")
	hist = model.fit(x=x_train, y=y_train, batch_size=32, epochs=300, validation_data=(x_valid, y_valid), callbacks=callbacksList)
	print("Training complete")
	print("")

	print(hist.history.keys())

	# Let's visualize the training loss
	sns.set()
	plt.figure(figsize=(6, 4))
	plt.plot(hist.history["loss"], label="Training loss")
	plt.plot(hist.history["val_loss"], label="Validation loss")
	plt.legend(loc="upper right")
	plt.xlabel("Training steps")
	plt.show()

	plt.figure(figsize=(6, 4))
	plt.plot(hist.history["accuracy"], label="Training accuracy")
	plt.plot(hist.history["val_accuracy"], label="Validation accuracy")
	plt.legend(loc="upper right")
	plt.xlabel("Training steps")
	plt.show()

	# Evaluate after training
	print("")
	print("Evaluating model")
	model.evaluate(x_valid, y_valid)
	print("")

	if saveModel:
		print("Saving model")
		model.load(checkpointDir)
		print("loaded1")
		model = keras.models.load_model(checkpointDir)
		print("loaded2")
		#model.save("./models/saved_models/" + modelTrainingName)
		print("Model saved") 


if __name__ == "__main__":
	main()