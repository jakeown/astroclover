import numpy as np
from astropy.io import fits
import numpy
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Activation, Dropout, Flatten, concatenate
from sklearn.utils import shuffle
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import accuracy_score
import tensorflow as tf
import h5py

def get_train_multi(type_name='train'):
	# Quicker data loading method
	# Data is stored and loaded in two h5 files
	# One with training data, other with training labels

	print 'Loading Training Data...'
	with h5py.File('three_class_gauss_'+type_name+'1.h5', 'r') as hf:
		X = hf['data'][:]
	hf.close()
	with h5py.File('labels_three_class_gauss_'+type_name+'1.h5', 'r') as hf:
		y = hf['data'][:]
	hf.close()
	return X, y

def get_train_nh3(type_name='train'):
	# Quicker data loading method
	# Data is stored and loaded in two h5 files
	# One with training data, other with training labels
	if type_name=='train':
		xx = '300000'
		yy = 'test_'
	elif type_name=='gas_train':
		xx = 'GAS_train'
		yy = ''
	elif type_name=='gas_val':
		xx = 'GAS_val'
		yy = ''
	elif type_name=='gas_test':
		xx = 'GAS_test'
		yy = ''
	else:
		xx = 'val'
		yy = ''
	print 'Loading Training Data...'
	with h5py.File('nh3_three_class_'+yy+xx+'.h5', 'r') as hf:
		X = hf['data'][:]
	hf.close()
	with h5py.File('labels_nh3_three_class_'+xx+'.h5', 'r') as hf:
		y = hf['data'][:]
	hf.close()
	return X, y

# Define models

def branch_conv1d_mod():
	#model = Sequential()
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last", input_shape=(500,1)))
	#model.add(MaxPooling1D(pool_size=2))
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	#model.add(Conv1D(36, kernel_size=3, activation='relu', padding='same'))
	#model.add(MaxPooling1D(pool_size=2))
	#model.add(Flatten())
	b1_input = Input(shape=(500,1))
	b1_conv1 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b1_input)
	b1_conv2 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(b1_conv1)
	#b1_conv3 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(b1_conv2)
	#b1_conv4 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(b1_conv3)
	b1_flat = Flatten()(b1_conv2)
	b2_input = Input(shape=(500,1))
	b2_conv1 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b2_input)
	b2_conv2 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(b2_conv1)
	#b2_conv3 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(b2_conv2)
	#b2_conv4 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(b2_conv3)
	b2_flat = Flatten()(b2_conv2)
	merge1 = concatenate([b1_flat, b2_flat])
	#dense1 = Dense(1000, activation='relu')(merge1) # tried 3000 
	dense1 = Dense(3000, activation='relu')(merge1) #regular 3class
	#dense2 = Dense(500, activation='relu')(dense1) # tried 3000
	dense2 = Dense(3000, activation='relu')(dense1) #regular 3class
	dense3 = Dense(3, activation='softmax')(dense2)
	model = Model(inputs=[b1_input, b2_input], outputs=dense3)
	#model.add(Dense(1000, activation='relu'))
	#model.add(Dropout(0.25)) # 0.25
	#model.add(Dense(1000, activation='relu'))
	#model.add(Dropout(0.25)) # 0.25
	#model.add(Dense(128, activation='relu')) # maybe remove
	#model.add(Dropout(0.25)) #
	#model.add(Dense(64, activation='relu')) #
	#model.add(Dropout(0.25)) #
	#model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

def branch_conv1d_mod_nh3():
	#model = Sequential()
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last", input_shape=(500,1)))
	#model.add(MaxPooling1D(pool_size=2))
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	#model.add(Conv1D(36, kernel_size=3, activation='relu', padding='same'))
	#model.add(MaxPooling1D(pool_size=2))
	#model.add(Flatten())
	b1_input = Input(shape=(1000,1))
	b1_conv1 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b1_input)
	b1_conv2 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b1_conv1)
	b1_flat = Flatten()(b1_conv2)
	b2_input = Input(shape=(1000,1))
	b2_conv1 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b2_input)
	b2_conv2 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b2_conv1)
	b2_flat = Flatten()(b2_conv2)
	merge1 = concatenate([b1_flat, b2_flat])
	dense1 = Dense(3000, activation='relu')(merge1) 
	#dense1 = Dense(512, activation='relu')(merge1) #regular 3class
	dense2 = Dense(3000, activation='relu')(dense1)
	#dense2 = Dense(512, activation='relu')(dense1) #regular 3class
	dense3 = Dense(3, activation='softmax')(dense2)
	model = Model(inputs=[b1_input, b2_input], outputs=dense3)
	#model.add(Dense(1000, activation='relu'))
	#model.add(Dropout(0.25)) # 0.25
	#model.add(Dense(1000, activation='relu'))
	#model.add(Dropout(0.25)) # 0.25
	#model.add(Dense(128, activation='relu')) # maybe remove
	#model.add(Dropout(0.25)) #
	#model.add(Dense(64, activation='relu')) #
	#model.add(Dropout(0.25)) #
	#model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

def conv1d_multi():
	model = Sequential()
	model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last", input_shape=(1000,1)))
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last", input_shape=(500,2)))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	#model.add(Dropout(0.25)) # 0.25
	model.add(Dense(512, activation='relu')) # maybe remove
	#model.add(Dropout(0.25)) #
	#model.add(Dense(64, activation='relu')) #
	#model.add(Dropout(0.25)) #
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

def conv1d_multi_local():
	model = Sequential()
	model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last", input_shape=(500,1)))
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last", input_shape=(500,2)))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	#model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(3000, activation='relu'))
	#model.add(Dropout(0.25)) # 0.25
	model.add(Dense(3000, activation='relu')) # maybe remove
	#model.add(Dropout(0.25)) #
	#model.add(Dense(64, activation='relu')) #
	#model.add(Dropout(0.25)) #
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

my_callbacks = [EarlyStopping(monitor="val_loss", mode='min', patience=2, verbose=1)]

def eval_mod(predicted, y_val_new):	
	# Final evaluation of the model
	predicted = np.argmax(predicted, axis=1) # find highest probability class
	score = accuracy_score(np.argmax(y_val_new,axis=1), predicted)

	print("Accuracy: %.2f%%" % (score*100))

# Fit for two-branch case
# Train an ensemble of models, average later
def train_ensemble(X_train_new, y_train_new, my_callbacks, mod_num, X_v, y_v): 
	for i in range(mod_num):
		es = EarlyStopping(monitor="val_loss", mode='min', patience=5, verbose=1)
		mc = ModelCheckpoint("model_cnn_3class"+ str(i) +"_gauss_3000_2conv_GAS.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
		model = branch_conv1d_mod()
		X_v = [X_v[:,:,0].reshape(X_v.shape[0], X_v.shape[1], 1), X_v[:,:,1].reshape(X_v.shape[0], X_v.shape[1], 1)]
		X_train_new, y_train_new = shuffle(X_train_new, y_train_new)
		hist = model.fit([X_train_new[:,:,0].reshape(X_train_new.shape[0], X_train_new.shape[1], 1), X_train_new[:,:,1].reshape(X_train_new.shape[0], X_train_new.shape[1], 1)], y_train_new, validation_data = (X_v, y_v), epochs=40, batch_size=100, callbacks=[es, mc])
		#model.save("model_cnn_3class"+ str(i) +"_gauss.h5")
		#model.save("model_cnn_3class.h5")
		#model.save("model_cnn_3class_1000fc.h5")
		print("Load best model")
		model = load_model("model_cnn_3class"+ str(i) +"_gauss_3000_2conv_GAS.h5")
		X_val_new, y_val_new = get_train_multi(type_name='test')
		predicted = model.predict([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)])
		eval_mod(predicted, y_val_new)
		
def train_merged(X_train_new, y_train_new, my_callbacks):
	# Fit for 1000 channel, merged spectrum case
	model = conv1d_multi()
	hist = model.fit(numpy.concatenate([X_train_new[:,:,0], X_train_new[:,:,1]], axis=1).reshape(X_train_new.shape[0], X_train_new.shape[1]*2, 1), y_train_new, epochs=20, batch_size=100, callbacks=my_callbacks)
	model.save("model_cnn_3class_mergedspec.h5")
	print("Saved model to disk")
	X_val_new, y_val_new = get_train_multi(type_name='test_gauss')
	predicted = model.predict(numpy.concatenate([X_val_new[:,:,0], X_val_new[:,:,1]], axis=1).reshape(X_val_new.shape[0], X_val_new.shape[1]*2, 1))
	eval_mod(predicted, y_val_new)

def train_local(X_train_new, y_train_new, my_callbacks, X_v, y_v, glob=False):
	if glob:
		tx = 'global'
	else:
		tx = 'local'
	model = conv1d_multi_local()
	es = EarlyStopping(monitor="val_loss", mode='min', patience=5, verbose=1)
	mc = ModelCheckpoint("model_cnn_3class_gauss_3000_2conv_"+tx+".h5", monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	X_v = X_v[:,:,int(glob)].reshape(X_v.shape[0], X_v.shape[1], 1)
	X_train_new, y_train_new = shuffle(X_train_new, y_train_new)
	hist = model.fit(X_train_new[:,:,int(glob)].reshape(X_train_new.shape[0], X_train_new.shape[1], 1), y_train_new, validation_data = (X_v, y_v), epochs=40, batch_size=100, callbacks=[es, mc])
	print("Load best model")
	model = load_model("model_cnn_3class_gauss_3000_2conv_"+tx+".h5")
	X_val_new, y_val_new = get_train_multi(type_name='test')
	predicted = model.predict(X_val_new[:,:,int(glob)].reshape(X_val_new.shape[0], X_val_new.shape[1], 1))
	eval_mod(predicted, y_val_new)

def train_nh3(X_train_new, y_train_new, my_callbacks, X_v, y_v):
	es = EarlyStopping(monitor="val_acc", mode='max', patience=4, verbose=1)
	mc = ModelCheckpoint("model_cnn_3class_nh3_sep_short_valacc_GAS.h5", monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	model = branch_conv1d_mod_nh3()
	X_v = [X_v[:,:,0].reshape(X_v.shape[0], X_v.shape[1], 1), X_v[:,:,1].reshape(X_v.shape[0], X_v.shape[1], 1)]
	hist = model.fit([X_train_new[:,:,0].reshape(X_train_new.shape[0], X_train_new.shape[1], 1), X_train_new[:,:,1].reshape(X_train_new.shape[0], X_train_new.shape[1], 1)], y_train_new, validation_data = (X_v, y_v), epochs=20, batch_size=100, callbacks=[es, mc])
	#model.save("model_cnn_3class_nh3_sep_short.h5")
	print("Load best model")
	model = load_model("model_cnn_3class_nh3_sep_short_valacc_GAS.h5")
	X_val_new, y_val_new = get_train_nh3(type_name='gas_test')
	predicted = model.predict([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)])
	eval_mod(predicted, y_val_new)

def fit_cnn(nh3=False, plot=False, local=False, glob=False):
# Load training and validation data
	if nh3:
		# Train for ammonia hyperfine case
		X_train_new, y_train_new = get_train_nh3(type_name='gas_train')
		X_v, y_v = get_train_nh3(type_name='gas_val')
	else:
		# Train for non-hyperfine case
		X_train_new, y_train_new = get_train_multi(type_name='train')
		X_v, y_v = get_train_multi(type_name='val')


	print X_train_new.shape
	print y_train_new.shape
	
	if plot:
		for i in X_train_new[-100:]:
			plt.plot(range(len(i[:,0])), i[:, 0])
			plt.plot(range(len(i[:,0])), i[:, 1]+1)
			plt.show()

	# Shuffle the training set order
	X_train_new, y_train_new = shuffle(X_train_new, y_train_new, random_state=0)

	if nh3:
		train_nh3(X_train_new, y_train_new, my_callbacks, X_v=X_v, y_v=y_v)
	elif local:
		train_local(X_train_new, y_train_new, my_callbacks, X_v=X_v, y_v=y_v, glob=glob)
	elif glob:
		train_local(X_train_new, y_train_new, my_callbacks, X_v=X_v, y_v=y_v, glob=glob)
	else:
		train_ensemble(X_train_new, y_train_new, my_callbacks, mod_num=1, X_v=X_v, y_v=y_v)

#fit_cnn(nh3=True, plot=False)
#fit_cnn(nh3=False, plot=False)
#fit_cnn(nh3=False, plot=False, local=True)
#fit_cnn(nh3=False, plot=False, local=False, glob=True)
