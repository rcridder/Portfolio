import numpy as np
import sys, os
from time import time 
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization

''' stuff from:
code:
https://elitedatascience.com/keras-tutorial-deep-learning-in-python

saving models:
http://machinelearningmastery.com/save-load-keras-deep-learning-models/
'''
#run with python <filename> >> <logfile.log> 2>&1

def main(): 
	print ('main (import success)')
	# 4. Load pre-shuffled MNIST data into train and test sets
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	"""
	####TEST
	X_train = X_train[:3]
	Y_train = Y_train[:3]
	X_test = X_test[:3]
	Y_test = Y_test[:3]
	"""

	print "before ", X_train.shape
	# 5. Preprocess input data
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	

	# 6. Preprocess class labels
	Y_train = np_utils.to_categorical(Y_train, 10)
	Y_test = np_utils.to_categorical(Y_test, 10)
	#Y_tune = np_utils.to_categorical(Y_tune, 10)

	
	# 7. Define model architecture
	CNN1(X_train, Y_train, X_test, Y_test, 3, 3, 128, 32)
	CNN1(X_train, Y_train, X_test, Y_test, 3, 5, 128, 32)
	CNN1(X_train, Y_train, X_test, Y_test, 3, 11, 128, 32)
	CNN1(X_train, Y_train, X_test, Y_test, 3, 3, 64, 32)
	CNN1(X_train, Y_train, X_test, Y_test, 3, 3, 256, 32)
	CNN1(X_train, Y_train, X_test, Y_test, 3, 3, 384, 32)
	CNN1(X_train, Y_train, X_test, Y_test, 3, 3, 128, 64)
	CNN1(X_train, Y_train, X_test, Y_test, 3, 3, 128, 16)
	CNN1(X_train, Y_train, X_test, Y_test, 16, 3, 128, 32)
	CNN1(X_train, Y_train, X_test, Y_test, 32, 3, 128, 32)
	########CNN1(X_train, Y_train, X_test, Y_test, 64, 3, 128, 32) #lan #memory error
	
	CNN1(X_train, Y_train, X_test, Y_test, 32, 3, 128, 32, True) #True batch normalization



def CNN1(X1, Y1, X2, Y2, K, F, D, B, batch_norm=False): #K = number of kernels, F = size of kernels, D = depth, B = batch size
	#define file convention for saving weights and data
	filename = ('models/%s_%s_%s_%s' %(K, F, D, B))
	if batch_norm:
		filename+='_withBatchNormalization'
	filejson = (filename+'.json')
	fileweights = filename+'_weights.h5'
	filetime = filename+'_time.txt'
	file_exist = False
	print filename

	if os.path.isfile(filejson) and os.path.isfile(fileweights) and os.path.isfile(filetime):
		file_exist=True
		fittime = open(filetime, 'r').readlines()
		json_file = open(filejson, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights(fileweights)
		print("Loaded model from disk")
		 
	else:
		print ('calculating model')
		model = Sequential()
		 
		model.add(Convolution2D(K, (F, F), activation='relu', input_shape=(28,28, 1)))
		if batch_norm:
			model.add(BatchNormalization())

		model.add(Convolution2D(K, (F, F), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		 
		model.add(Flatten())

		# Output size of the layer with one hidden layer of size D
		model.add(Dense(D, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10, activation='softmax'))

	# manually adjust the adam optimizer, with a larger learning rate
	adam_opt = Adam(lr=.01, beta_1=0.9, beta_2=0.999, epsilon=.001, decay=0.0)
	# 8. Compile model
	model.compile(loss='categorical_crossentropy',
	              optimizer=adam_opt,
	              metrics=['accuracy'])

	print ('model compiled')

	if not file_exist:
		t0 = time()	 
		# 9. Fit model on training data
		model.fit(X1, Y1, nb_epoch = 10, batch_size = B)#, verbose = 0) #I think it should be 2 for one log line per epoch, but when I do that, nothing prints for at least 15 minutes (not even the print line from main)
		          #batch_size=B, nb_epoch=2, verbose=1)
		fittime = time()-t0
		with open(filetime, 'w') as f:
			f.write(str(fittime))

	print (model.summary())
	t1 = time()
	# Evaluate on training data
	print("Score for CNN1 train K=%d, F=%d, D=%d, B=%d is: %s (%s)" %(K, F, D, B, model.evaluate(X1, Y1, verbose=0), model.metrics_names))
	t2 = time()
	# 10. Evaluate model on test data
	print("Score for CNN1 test K=%d, F=%d, D=%d, B=%d is: %s (%s)" %(K, F, D, B, model.evaluate(X2, Y2, verbose=0), model.metrics_names))
	t3 = time()
	print "Time fit: ", fittime
	print "Time eval training data: ", t2-t1
	print "Time eval test data: ", t3-t2

	if not file_exist:
		# serialize model to JSON
		model_json = model.to_json()
		with open(filejson, "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(fileweights)
		print("Saved model to disk")

	print "________________________________________"


if __name__=='__main__':
	main()
	sys.exit()