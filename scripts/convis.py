import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp
from kerasncp import wirings
from kerasncp.tf import LTCCell
from tensorflow.python.keras.activations import softmax


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer, ZeroPadding2D
from collections import defaultdict
import visualkeras
from PIL import ImageFont

def visProposedModel():
	image_size = 256
	model = Sequential()
	model.add(InputLayer(input_shape=(image_size, image_size, 3)))

	model.add(Conv2D(16, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Conv2D(32, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))

	# Now visualize the model!

	color_map = defaultdict(dict)
	color_map[Conv2D]['fill'] = 'orange'
	color_map[ZeroPadding2D]['fill'] = 'gray'
	color_map[Dropout]['fill'] = 'pink'
	color_map[MaxPooling2D]['fill'] = 'red'
	color_map[Dense]['fill'] = 'green'
	color_map[Flatten]['fill'] = 'teal'

	font = ImageFont.truetype("arial.ttf", 32)

	path = "./images/visuals/model"

	visualkeras.layered_view(model, to_file=path + '.png', type_ignore=[visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_legend.png', type_ignore=[visualkeras.SpacingDummyLayer],
							legend=True, font=font)
	visualkeras.layered_view(model, to_file=path + '_spacing_layers.png', spacing=0)
	visualkeras.layered_view(model, to_file=path + '_type_ignore.png',
							type_ignore=[ZeroPadding2D, Dropout, Flatten, visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_color_map.png',
							color_map=color_map, type_ignore=[visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_flat.png',
							draw_volume=False, type_ignore=[visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_scaling.png',
							scale_xy=1, scale_z=1, max_z=1000, type_ignore=[visualkeras.SpacingDummyLayer])

def visVGG19():
	image_size = 224
	model = Sequential()
	model.add(InputLayer(input_shape=(image_size, image_size, 3)))

	model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Dense(4096, activation='relu'))
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(1000, activation='relu'))

	# Now visualize the model!

	color_map = defaultdict(dict)
	color_map[Conv2D]['fill'] = 'orange'
	color_map[ZeroPadding2D]['fill'] = 'gray'
	color_map[Dropout]['fill'] = 'pink'
	color_map[MaxPooling2D]['fill'] = 'red'
	color_map[Dense]['fill'] = 'green'
	color_map[Flatten]['fill'] = 'teal'

	font = ImageFont.truetype("arial.ttf", 32)

	path = "./images/visuals/vgg19_model"

	visualkeras.layered_view(model, to_file=path + '.png', type_ignore=[visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_legend.png', type_ignore=[visualkeras.SpacingDummyLayer],
							legend=True, font=font)
	visualkeras.layered_view(model, to_file=path + '_spacing_layers.png', spacing=0)
	visualkeras.layered_view(model, to_file=path + '_type_ignore.png',
							type_ignore=[ZeroPadding2D, Dropout, Flatten, visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_color_map.png',
							color_map=color_map, type_ignore=[visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_flat.png',
							draw_volume=False, type_ignore=[visualkeras.SpacingDummyLayer])
	visualkeras.layered_view(model, to_file=path + '_scaling.png',
							scale_xy=1, scale_z=1, max_z=1000, type_ignore=[visualkeras.SpacingDummyLayer])

if __name__ == "__main__":
	visVGG19()