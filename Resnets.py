# Dependencies
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization
from keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing import image
import pickle
import numpy as np
from PIL import Image



class Resnet_train(object):

	def __init__(self, path, epochs, batch_size):

		"""
		The Resnet_train class is initialized with its arguments when an 
		instance of the class is created.
		"""

		X_train, X_val, Y_train, Y_val = self._load_dataset(path)

		model = self._Resnet50(input_shape = (64, 64, 3), classes = 10)
		model.summary()
		checkpointer = ModelCheckpoint(filepath="./data/model.h5", verbose=0, save_best_only=True)
		tensorboard = TensorBoard(log_dir='data/./logs', histogram_freq=0, write_graph=True, write_images=True)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,shuffle=True, 
		validation_data=(X_val, Y_val), verbose=1, callbacks=[checkpointer, tensorboard]).history




	def _indentity_block(self, X, filters, f, stage, block):

		"""
		The identity blocks are standard blocks used in Resnets where the input activation a[l] 
		has the same dimension as the output activation a[l+2]. 
											  ----> X
							-->										-->
			-->																			-->
		
		X ----> |Conv2D|BatchNorm|Relu| ----> |Conv2D|BatchNorm|Relu| ----> |Conv2D|BatchNorm|	+	|Relu|-->

		Arguments:
		X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev) 
		filters -- list of integers defining the number of filters in the conv layers
		f -- with of filter used in the conv layers
		stage -- interger representation of the stage the operation is taking place in the Resnet network
		block -- string representation of the block at which an operation taking place at a given stage in the Resnet

		Returns:
		X -- output of the identity block, tensor of shape (n_H, n_W, n_C) 
		"""
		conv_layer_name = 'res' + str(stage) + block + '_branch'
		bn_layer_name = 'bm' + str(stage) + block + '_branch'

		X_shortcut = X

		F1, F2, F3 = filters

		# First component of main path
		X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
			name = conv_layer_name + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis = 3, name = bn_layer_name + '2a')(X)
		X = Activation('relu')(X)

		# Second component of main path
		X = Conv2D(filters = F2, kernel_size = (1, 1), strides = (1, 1), padding = 'same',
			name = conv_layer_name + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis = 3, name = bn_layer_name + '2b')(X)
		X = Activation('relu')(X)

		# Third component of main path
		X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
			name = conv_layer_name + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis = 3, name = bn_layer_name + '2c')(X)

		# Final step : adding the shortcut componet to X and applying 'relu' activation on the combination 
		X = Add()([X_shortcut,X])
		X = Activation('relu')(X)

		return X



	def _convolution_block(self, X, filters, f, s, stage, block):

		"""
		Convolutional blocks are used when the input activation a[l] is different from the
		output activation a[l+2]
										 --> X * |Conv2D|BatchNorm|
						  -->										   -->
			-->																			-->
		
		X ----> |Conv2D|BatchNorm|Relu| ----> |Conv2D|BatchNorm|Relu| ----> |Conv2D|BatchNorm|	+	|Relu|-->

		Arguments:
		X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev
		filters -- a list of integers indicating the number of filters at each convolutional layer
		f -- an integer indicating the dimension of the filter
		s -- an integer indicating the stride value 
		  
	
		Returns:
		X -- output tensor of shape (m, n_H, n_W, n_C)
		"""

		conv_layer_name = 'res' + str(stage) + block + '_branch'
		bn_layer_name = 'bn' + str(stage) + block + '_branch'

		X_shortcut = X 

		F1, F2, F3 = filters

		# First component of the main path 
		X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), padding = 'valid',
			name = conv_layer_name + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis = 3, name = bn_layer_name + '2a')(X)
		X = Activation('relu')(X)

		# Second component of the main path 
		X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same',
			name = conv_layer_name + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis = 3, name = bn_layer_name + '2b')(X)
		X = Activation('relu')(X)

		# Third component of the main path 
		X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid',
			name = conv_layer_name + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis = 3, name = bn_layer_name + '2c')(X)
		
		# Shortcut path
		X_shortcut = Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s), padding = 'valid',
			name = conv_layer_name + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
		X_shortcut = BatchNormalization(axis = 3, name = bn_layer_name + '1')(X_shortcut)

		# Final step : adding the shortcut component to the activation tensor from the third component of the main path
		X = Add()([X, X_shortcut])
		X = Activation('relu')(X)

		return X



	def _Resnet50(self, input_shape, classes):

		"""
		Implementation of Resnet50 network, with 50 layers follows the following architecture:
		CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
		-> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

		Argumments:
		input_shape -- shape of the input tensor (m, n_H_prev, n_W_prev, n_C_prev)
		classes -- number of classes the dataset contains 


		Note:
		IDBLOCKx2 means IDBLOCK twice 
		"""

		# input tensor and zero padding
		X_input = Input(input_shape)
		X = ZeroPadding2D(padding = (3,3))(X_input)

		# stage 1
		X = Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), 
			kernel_initializer = glorot_uniform(seed=0), name = 'conv1')(X)
		X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
		X = Activation('relu')(X)
		X = MaxPooling2D(pool_size = (3,3), strides = (2,2))(X)

		# stage 2
		X = self._convolution_block(X, filters = [64,64,256], f = 3, s = 1, stage = 2, block = 'a')
		X = self._indentity_block(X, filters = [64,64,256], f = 3, stage = 2, block = 'b')
		X = self._indentity_block(X, filters = [64,64,256], f = 3, stage = 2, block = 'c')

		# stage 3
		X = self._convolution_block(X, filters = [128,128,512], f = 3, s = 2, stage = 3, block = 'a')
		X = self._indentity_block(X, filters = [128,128,512], f = 3, stage = 3, block = 'b')
		X = self._indentity_block(X, filters = [128,128,512], f = 3, stage = 3, block = 'c')
		X = self._indentity_block(X, filters = [128,128,512], f = 3, stage = 3, block = 'd')

		# stage 4
		X = self._convolution_block(X, filters = [256,256,1024], f = 3, s = 2, stage = 4, block = 'a')
		X = self._indentity_block(X, filters = [256,256,1024], f = 3, stage = 4, block = 'b')
		X = self._indentity_block(X, filters = [256,256,1024], f = 3, stage = 4, block = 'c')
		X = self._indentity_block(X, filters = [256,256,1024], f = 3, stage = 4, block = 'd')
		X = self._indentity_block(X, filters = [256,256,1024], f = 3, stage = 4, block = 'e')
		X = self._indentity_block(X, filters = [256,256,1024], f = 3, stage = 4, block = 'f')

		# stage 5
		X = self._convolution_block(X, filters = [512, 512, 2048], f = 3, s = 1, stage = 5, block = 'a')
		X = self._indentity_block(X, filters = [512, 512, 2048], f = 3, stage = 5, block = 'b')
		X = self._indentity_block(X, filters = [512, 512, 2048], f = 3, stage = 5, block = 'c')

		# Average pool, flatten and FC
		X = AveragePooling2D(pool_size = (2,2), name = 'avg_pool')(X)
		X = Flatten()(X)
		X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), 
			kernel_initializer = glorot_uniform(seed=0))(X)

		model = Model(inputs = X_input, outputs = X, name = "Resnet50")

		return model



	def _reshape(self, data):

		"""
		A helper fuction to deconstruct the CIFAR 10 dataset into an appropriate tensor

		Arguements:
		data -- an instance from the dataset to deconstruct  

		Returns :
		d -- deconstructed instance of shape (64,64,3)
		"""

		d = np.zeros((32,32,3))
		d_r = data[0:1024].reshape(32,32)
		d_g = data[1024:2048].reshape(32,32)
		d_b = data[2048:].reshape(32,32)

		for h in range(32):
		    for w in range(32):
		        for c in range(3):

		            if c == 0 : d[h,w,c] = d_r[h,w]
		            elif c == 1 : d[h,w,c] = d_g[h,w]
		            else : d[h,w,c] = d_b[h,w]

		array = np.array(d, dtype=np.uint8)
		img = Image.fromarray(array)
		temp = img.resize(size = (64,64))
		d = image.img_to_array(temp)

		#plt.imshow(d)
		#plt.show()
		return d



	def _load_dataset(self, path):

		"""
		This function loads the dataset and splits it into training and validation
		sets

		Arguments:
		path -- the directory in which the training datasets are to be found

		Returns:
		X_train -- training set of shape (40000, 64, 64, 3)
		X_val -- validation set of shape (40000, 64, 64, 3)
		Y_train -- target values of the training set of shape (40000, 10)
		Y_val -- target values of the validation set of shape (40000, 10)

		"""
		while True:
			
			try:
				X_train = np.load("data/X_train.npy")
				X_val = np.load("data/X_val.npy")
				Y_train = np.load("data/Y_train.npy")
				Y_val = np.load("data/Y_val.npy")
				break

			except FileNotFoundError:

				data_temp = np.zeros((50000,64,64,3))
				label_temp = []

				for i in range(5):

					file = path + str(i+1)
					with open(file, 'rb') as fo:
						temp_element = pickle.load(fo, encoding='bytes')

					temp_data = temp_element[b'data']
					label_temp.extend(temp_element[b'labels'])

					for j in range(10000):
						data_temp[j+(i*10000)] = self._reshape(temp_data[j])

				label_temp = np.eye(10)[np.array(label_temp)]

				np.random.seed(123)
				permutations = list(np.random.permutation(50000))
				X = data_temp[permutations, :, : , :] 
				Y = label_temp[permutations, :]
				X_train = X[0:40000, :, :, :] 
				Y_train = Y[0:40000, :]
				X_val = X[40000:50000, :, :, :] 
				Y_val = Y[40000:50000, :]

				np.save("./data/X_train", X_train)
				np.save("./data/X_val", X_val)
				np.save("./data/Y_train", Y_train)
				np.save("./data/Y_val", Y_val)
				break

		return X_train, X_val, Y_train, Y_val



class Resnet_infer(object):

	def __init__(self, mode, path):

		"""
		The Resnet_infer class is initialized with its arguments when an 
		instance of the class is created.
		"""

		model = load_model('data/model.h5') 

		if mode == "test":

			X_test, Y_test = self._load_dataset(path)
			preds = model.evaluate(X_test, Y_test)
			print ("Loss = " + str(preds[0]))
			print ("Test Accuracy = " + str(preds[1]))


		elif mode == "predict":			
			
			label_dict = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
			'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

			img = image.load_img(path, target_size=(64, 64))
			x = image.img_to_array(img)
			x = np.reshape(x, (1,64,64,3))
			temp_pred = model.predict(x)
			idx = np.argmax(temp_pred)
			
			print("The object detected in the picture is a(n) : " + 
				list(label_dict.keys())[list(label_dict.values()).index(idx)])


	
	def _load_dataset(self, path):

		"""
		This function loads the test dataset

		Arguments:
		path -- the directory in which the test dataset can be found

		Returns:
		X_test -- test dataset of shape (10000, 64, 64, 3)
		Y_test -- target values of the test dataset of shape (10000, 10)
		"""
		while True:
			
			try:
				X_test = np.load("data/X_test.npy")
				Y_test = np.load("data/Y_test.npy")
				break

			except FileNotFoundError:

				X_test = np.zeros((10000,64,64,3))
				Y_test = []

				
				with open(path, 'rb') as fo:
					temp_element = pickle.load(fo, encoding='bytes')

				temp_data = temp_element[b'data']
				Y_test.extend(temp_element[b'labels'])

				for j in range(10000):
					X_test[j] = self._reshape(temp_data[j])

				Y_test = np.eye(10)[np.array(Y_test)]
				
				np.save("./data/X_test", X_test)
				np.save("./data/Y_test", Y_test)

				break


		return X_test, Y_test



	def _reshape(self, data):

			"""
			A helper fuction to deconstruct the CIFAR 10 dataset into an appropriate tensor

			Arguements:
			data -- a particular instance from the dataset to deconstruct  

			Returns :
			d -- deconstructed instance of shape (64,64,3)
			"""

			d = np.zeros((32,32,3))
			d_r = data[0:1024].reshape(32,32)
			d_g = data[1024:2048].reshape(32,32)
			d_b = data[2048:].reshape(32,32)

			for h in range(32):
			    for w in range(32):
			        for c in range(3):

			            if c == 0 : d[h,w,c] = d_r[h,w]
			            elif c == 1 : d[h,w,c] = d_g[h,w]
			            else : d[h,w,c] = d_b[h,w]

			array = np.array(d, dtype=np.uint8)
			img = Image.fromarray(array)
			temp = img.resize(size = (64,64))
			d = image.img_to_array(temp)

			#plt.imshow(d)
			#plt.show()
			return d



def main(path, mode, epochs = None, batch_size = None):

	if mode == 'train':

		resnet_train = Resnet_train(path, epochs, batch_size)

	else: resnet_infer = Resnet_infer(mode, path)



if __name__ == '__main__':

	path  = 'bird.png' # change path w.r.t mode selected
	mode = 'predict' # train, predict or test
	epochs = 100
	batch_size = 64

	main(path, mode, epochs, batch_size)




