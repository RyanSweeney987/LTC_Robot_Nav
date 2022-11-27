from os import name
import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp
from kerasncp import wirings
from kerasncp.tf import LTCCell

#metricsUsed = ["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
#metricsUsed = [tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
#metricsUsed = ["accuracy", tf.keras.metrics.AUC()]
metricsUsed = ["accuracy"]

def getCNNModel6(inputs):
	""" Model 6 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=10,  # Number of inter neurons
		command_neurons=10,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=5,  # How many outgoing synapses has each sensory neuron
		inter_fanout=5,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=6,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (5, 5), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel5(inputs):
	""" Model 5 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=10,  # Number of inter neurons
		command_neurons=10,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=5,  # How many outgoing synapses has each sensory neuron
		inter_fanout=5,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=6,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(128, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.25)),
			keras.layers.TimeDistributed(keras.layers.Dense(64, activation="relu", kernel_regularizer='l1_l2')),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.125)),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel4(inputs):
	""" Model 4 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=10,  # Number of inter neurons
		command_neurons=10,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=5,  # How many outgoing synapses has each sensory neuron
		inter_fanout=5,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=6,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(128, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.25)),
			keras.layers.TimeDistributed(keras.layers.Dense(64, activation="relu", kernel_regularizer='l1_l2')),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.125)),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel3_5(inputs):
	""" Model 3.5 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=64,  # Number of inter neurons
		command_neurons=64,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=32,  # How many outgoing synapses has each sensory neuron
		inter_fanout=16,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=8,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel3_4(inputs):
	""" Model 3.4 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"), name="conv_layer_1"),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"), name="max_pool_2d_1"),
			keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"), name="conv_layer_2"),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"), name="max_pool_2d_2"),
			keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"), name="conv_layer_3"),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"), name="max_pool_2d_3"),
			keras.layers.TimeDistributed(keras.layers.Conv2D(128, (5, 5), activation="relu", padding="same"), name="conv_layer_4"),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"), name="max_pool_2d_4"),
			keras.layers.TimeDistributed(keras.layers.Flatten(), name="flatten"),
			keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"), name="dense_relu"),
			keras.layers.RNN(ncp_cell, return_sequences=True, name="ltc_layer"),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax"), name="softmax"),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel3_3(inputs):
	""" Model 3.3 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.01),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel3_2(inputs):
	""" Model 3.2 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel3(inputs):
	""" Model 3 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"), name="conv_layer_1"),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"), name="max_pool_2d_1"),
			keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"), name="conv_layer_2"),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"), name="max_pool_2d_2"),
			keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"), name="conv_layer_3"),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"), name="max_pool_2d_3"),
			keras.layers.TimeDistributed(keras.layers.Flatten(), name="flatten"),
			keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"), name="dense_relu"),
			keras.layers.RNN(ncp_cell, return_sequences=True, name="ltc_layer"),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax"), name="softmax"),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel2(inputs):
	""" Model 2 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=1,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel1_3(inputs):
	""" Model 1.3 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel1_2(inputs):
	""" Model 1.2 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (7, 7), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getCNNModel1(inputs):
	""" Model 1 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)
	# LTC model
	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(16, (5, 5), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(8, (3, 3), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getVGG16Model(inputs):
	""" Model VGG16 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)

	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			# keras.layers.TimeDistributed(
			# 	keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			# ),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			# keras.layers.TimeDistributed(
			# 	keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu")
			# ),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			# keras.layers.TimeDistributed(
			# 	keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			# ),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(4096, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dense(2048, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.5)),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

def getVGG19Model(inputs):
	""" Model VGG19 """
	#Neural Circuit Policy wiring
	ncp_arch = wirings.NCP(
		inter_neurons=20,  # Number of inter neurons
		command_neurons=20,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
		inter_fanout=10,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=6,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=8,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)

	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dense(4096, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dense(2048, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.5)),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
		metrics=metricsUsed
	)

	return (model, ncp_cell)

# Based on https://www.nature.com/articles/s42256-020-00237-3.epdf?sharing_token=xHsXBg2SoR9l8XdbXeGSqtRgN0jAjWel9jnR3ZoTv0PbS_e49wmlSXvnXIRQ7wyir5MOFK7XBfQ8sxCtVjc7zD1lWeQB5kHoRr4BAmDEU0_1-UN5qHD5nXYVQyq5BrRV_tFa3_FZjs4LBHt-yebsG4eQcOnNsG4BenK3CmBRFLk%3D
def getFFCNN(inputs):
	""" Model FFCNN """
	ncp_arch = wirings.NCP(
		inter_neurons=30,  # Number of inter neurons
		command_neurons=30,  # Number of command neurons
		motor_neurons=4,  # Number of motor neurons
		sensory_fanout=15,  # How many outgoing synapses has each sensory neuron
		inter_fanout=15,  # How many outgoing synapses has each inter neuron
		recurrent_command_synapses=12,  # Now many recurrent synapses are in the
		# command neuron layer
		motor_fanin=16,  # How many incoming synapses has each motor neuron
	)
	ncp_cell = LTCCell(ncp_arch)

	model = keras.models.Sequential(
		[
			keras.layers.InputLayer(input_shape=inputs),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(48, (3, 3), strides=(2, 2),activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), strides=(1, 1),activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(
				keras.layers.Conv2D(32, (3, 3), strides=(1, 1),activation="relu", padding="same")
			),
			keras.layers.TimeDistributed(keras.layers.Flatten()),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.5)),
			keras.layers.TimeDistributed(keras.layers.Dense(1000, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.5)),
			keras.layers.TimeDistributed(keras.layers.Dense(100, activation="relu")),
			keras.layers.TimeDistributed(keras.layers.Dropout(0.3)),
			keras.layers.TimeDistributed(keras.layers.Dense(1)),
			keras.layers.RNN(ncp_cell, return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='categorical_crossentropy',
 		metrics=metricsUsed
	)

	return (model, ncp_cell)