import numpy as np
from astropy.io import fits
import numpy
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import load_model, Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Activation, Dropout, Flatten, concatenate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import tensorflow as tf
import os
import sys
import h5py
import struct

# Define the double Gaussian profile
def p_eval2(x, a, x0, sigma, a1, x1, sigma1):
	one = a*np.exp(-(x-x0)**2/(2*sigma**2))
	two = a1*np.exp(-(x-x1)**2/(2*sigma1**2))
	return one+two

def test_model():
	model = load_model('model_cnn_tpeak_gauss_3000_2conv.h5')
	X_val_new, y_val_new = get_train_set2(type_name='test')
	print X_val_new.shape
	#X_val_new = X_val_new.reshape(X_val_new.shape[0], img_rows, img_depth)
	#score = model.score(X_val_new, y_val_new)
	#print("Test Score: %.2f%%" % (score))
	preds = model.predict([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)])
	results = mean_absolute_error(y_val_new, preds)
	print("MAE: " + str(results))
	f,(ax1,ax2,ax3,ax4,ax5, ax6) = plt.subplots(1,6)
	ax1.scatter(preds[:,0], y_val_new[:,0], marker='.', alpha=0.3, rasterized=True)
	#heatmap, xedges, yedges = np.histogram2d(preds[:,0], y_val_new[:,0], bins=100)
	#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	#ax1.imshow(heatmap.T, extent=extent, origin='lower')
	ax1.plot([min(preds[:,0]),max(preds[:,0])],[min(preds[:,0]),max(preds[:,0])], linestyle='dashed', color='black')
	ax2.scatter(preds[:,1], y_val_new[:,1], marker='.', alpha=0.3, label='CNN', rasterized=True)
	ax2.plot([min(preds[:,1]),max(preds[:,1])],[min(preds[:,1]),max(preds[:,1])], linestyle='dashed', color='black')
	ax2.legend()
	ax3.scatter(preds[:,2], y_val_new[:,2], marker='.', alpha=0.3, rasterized=True)
	ax3.plot([min(preds[:,2]),max(preds[:,2])],[min(preds[:,2]),max(preds[:,2])], linestyle='dashed', color='black')
	ax4.scatter(preds[:,3], y_val_new[:,3], marker='.', alpha=0.3, rasterized=True)
	ax4.plot([min(preds[:,3]),max(preds[:,3])],[min(preds[:,3]),max(preds[:,3])], linestyle='dashed', color='black')
	ax5.scatter(preds[:,4], y_val_new[:,4], marker='.', alpha=0.3, rasterized=True)
	ax5.plot([min(preds[:,4]),max(preds[:,4])],[min(preds[:,4]),max(preds[:,4])], linestyle='dashed', color='black')
	ax6.scatter(preds[:,5], y_val_new[:,5], marker='.', alpha=0.3, rasterized=True)
	ax6.plot([min(preds[:,5]),max(preds[:,5])],[min(preds[:,5]),max(preds[:,5])], linestyle='dashed', color='black')
	ax1.set_ylabel('Ground Truth')
	ax1.set_xlabel('Predicted V$_{LSR}$')
	ax2.set_xlabel('Predicted V$_{LSR}$')
	ax3.set_xlabel('Predicted $\sigma$')
	ax4.set_xlabel('Predicted $\sigma$')
	ax5.set_xlabel('Predicted T$_{peak}$')
	ax6.set_xlabel('Predicted T$_{peak}$')
	ax1.set_title('V1')
	ax2.set_title('V2')
	ax3.set_title('W1')
	ax4.set_title('W2')
	ax5.set_title('T1')
	ax6.set_title('T2')
	if 3==3:
		ax4.set_xlim([-1,20])
		ax4.set_ylim([1,12])
		ax3.set_xlim([-1,20])
		ax3.set_ylim([1,12])
		ax5.set_xlim([0,1.4])
		ax5.set_ylim([0,1.4])
		ax6.set_xlim([0,1.4])
		ax6.set_ylim([0,1.4])
		ax2.set_xlim([-1.0,1.2])
		ax2.set_ylim([-0.6,0.8])
		ax1.set_xlim([-1.2,1.0])
		ax1.set_ylim([-0.8,0.6])
	f.set_size_inches(10,4)
	f.suptitle('        MAE: '+ str(round(results,3)))
	f.tight_layout()
	plt.show()

	for i,j,k in zip(X_val_new, preds, y_val_new): 
			spec = i[:,0]
			xaxis = numpy.linspace(-1,1, len(spec))
			step = abs(xaxis[23]-xaxis[24])
			gauss = p_eval2(xaxis, j[4], j[0], j[2]*step, j[5], j[1], j[3]*step)
			gauss2 = p_eval2(xaxis, k[4], k[0], k[2]*step, k[5], k[1], k[3]*step)
			plt.plot(xaxis, gauss2, zorder=20, color='black', linestyle='--')
			plt.plot(xaxis, gauss, zorder=11, color='orange')
			plt.plot(xaxis, spec)
			plt.scatter([k[0],k[1]], [k[4],k[5]], color='black', label='Ground Truth', marker='.', zorder=9)
			plt.scatter([j[0],j[1]], [j[4],j[5]], color='orange', zorder=10, alpha=0.7, label='CNN-prediction', marker='.')
			plt.plot([j[0]-(j[2]*step/2), j[0]+(j[2]*step/2)], [0,0], color='orange')
			plt.plot([j[1]-(j[3]*step/2), j[1]+(j[3]*step/2)], [0.1,0.1], color='orange')
			plt.plot([k[0]-(k[2]*step/2), k[0]+(k[2]*step/2)], [0.05,0.05], color='black')
			plt.plot([k[1]-(k[3]*step/2), k[1]+(k[3]*step/2)], [0.15,0.15], color='black')
			plt.xlabel('VLSR')
			plt.ylabel('Normalized Intensity')
			plt.legend()
			plt.show()

def get_train_set2(type_name='train'):
	# Quicker data loading method
	# Data is stored and loaded in two h5 files
	# One with training data, other with training labels

	print 'Loading Training Data...'
	with h5py.File('three_class_gauss_'+type_name+'1_reg.h5', 'r') as hf:
		X = hf['data'][:]
	hf.close()
	with h5py.File('params_three_class_gauss_'+type_name+'1_reg.h5', 'r') as hf:
		y = hf['data'][:]
	hf.close()
	return X, y

def test_chi(use_cnn=False):
	X_val_new, y_val_new = get_train_set2(type_name='test')
	ax = numpy.linspace(-1,1,500)
	step = abs(ax[23]-ax[24])
	if use_cnn:
		preds = numpy.load('chi_predictions_3class_reg_local_tpeak_cnn.npy')
		preds = numpy.column_stack((preds[:,1], preds[:,2], preds[:,4], preds[:,5], preds[:,0], preds[:,3]))
		preds[:,1] = preds[:,1]/step
		preds[:,3] = preds[:,3]/step
		tx = 'CNN+$\chi^2$'
	else:
		preds = numpy.load('chi_predictions_3class_reg_local_tpeak.npy')
		tx = '$\chi^2$'
	print min(preds[:,0])
	print min(preds[:,1])
	print min(preds[:,2])
	print min(preds[:,3])
	print preds[0:5]

	count=0
	for i in preds:
		preds[count]=[min([i[0],i[2]]), max([i[0],i[2]]), [i[1],i[3]][numpy.argmin([i[0],i[2]])], [i[1],i[3]][numpy.argmax([i[0],i[2]])], [i[4],i[5]][numpy.argmin([i[0],i[2]])] , [i[4],i[5]][numpy.argmax([i[0],i[2]])]]
		count+=1

	if use_cnn==False:
		preds[:,0] = 2.*((preds[:,0]-0.)/(499.-0.))-1
		preds[:,1] = 2.*((preds[:,1]-0.)/(499.-0.))-1
		preds[:,2] = (preds[:,2])
		preds[:,3] = (preds[:,3])

	results = mean_absolute_error(y_val_new, preds)
	print("MAE: " + str(results))
	f,(ax1,ax2,ax3,ax4, ax5, ax6) = plt.subplots(1,6)
	ax1.scatter(preds[:,0], y_val_new[:,0], marker='.', alpha=0.3, color='green', rasterized=True)
	ax1.plot([min(preds[:,0]),max(preds[:,0])],[min(preds[:,0]),max(preds[:,0])], linestyle='dashed', color='black')
	ax2.scatter(preds[:,1], y_val_new[:,1], marker='.', alpha=0.3, color='green', label=tx, rasterized=True)
	ax2.plot([min(preds[:,1]),max(preds[:,1])],[min(preds[:,1]),max(preds[:,1])], linestyle='dashed', color='black')
	ax2.legend()
	ax3.scatter(preds[:,2], y_val_new[:,2], marker='.', alpha=0.3, color='green', rasterized=True)
	ax3.plot([min(preds[:,2]),max(preds[:,2])],[min(preds[:,2]),max(preds[:,2])], linestyle='dashed', color='black')
	ax4.scatter(preds[:,3], y_val_new[:,3], marker='.', alpha=0.3, color='green', rasterized=True)
	ax4.plot([min(preds[:,3]),max(preds[:,3])],[min(preds[:,3]),max(preds[:,3])], linestyle='dashed', color='black')
	ax5.scatter(preds[:,4], y_val_new[:,4], marker='.', alpha=0.3, color='green', rasterized=True)
	ax5.plot([min(preds[:,4]),max(preds[:,4])],[min(preds[:,4]),max(preds[:,4])], linestyle='dashed', color='black')
	ax6.scatter(preds[:,5], y_val_new[:,5], marker='.', alpha=0.3, color='green', rasterized=True)
	ax6.plot([min(preds[:,5]),max(preds[:,5])],[min(preds[:,5]),max(preds[:,5])], linestyle='dashed', color='black')
	if use_cnn!=3:
		ax4.set_xlim([-1,20])
		ax4.set_ylim([1,12])
		ax3.set_xlim([-1,20])
		ax3.set_ylim([1,12])
		ax5.set_xlim([0,1.4])
		ax5.set_ylim([0,1.4])
		ax6.set_xlim([0,1.4])
		ax6.set_ylim([0,1.4])
		ax2.set_xlim([-1.0,1.2])
		ax2.set_ylim([-0.6,0.8])
		ax1.set_xlim([-1.2,1.0])
		ax1.set_ylim([-0.8,0.6])
	ax1.set_ylabel('Ground Truth')
	ax1.set_xlabel('Predicted V$_{LSR}$')
	ax2.set_xlabel('Predicted V$_{LSR}$')
	ax3.set_xlabel('Predicted $\sigma$')
	ax4.set_xlabel('Predicted $\sigma$')
	ax5.set_xlabel('Predicted T$_{peak}$')
	ax6.set_xlabel('Predicted T$_{peak}$')
	ax1.set_title('V1')
	ax2.set_title('V2')
	ax3.set_title('W1')
	ax4.set_title('W2')
	ax5.set_title('T1')
	ax6.set_title('T2')
	f.suptitle('        MAE: '+ str(round(results,3)))
	f.set_size_inches(10,4)
	f.tight_layout()
	plt.show()

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
	b1_conv2 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b1_conv1)
	b1_flat = Flatten()(b1_conv2)
	b2_input = Input(shape=(500,1))
	b2_conv1 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b2_input)
	b2_conv2 = Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last")(b2_conv1)
	b2_flat = Flatten()(b2_conv2)
	merge1 = concatenate([b1_flat, b2_flat])
	#dense1 = Dense(1000, activation='relu')(merge1) 
	dense1 = Dense(3000, activation='relu')(merge1) #regular 3class
	#dense2 = Dense(500, activation='relu')(dense1)
	dense2 = Dense(3000, activation='relu')(dense1) #regular 3class
	dense3 = Dense(6)(dense2)
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
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()
	return model

def conv1d_mod():
	model = Sequential()
	model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', data_format="channels_last", input_shape=(500,2)))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
	#model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(3000, activation='relu'))
	model.add(Dropout(0.1)) # 0.25
	model.add(Dense(3000, activation='relu'))
	#model.add(Dense(128, activation='relu')) # maybe remove
	model.add(Dropout(0.1)) #
	#model.add(Dense(64, activation='relu')) #
	#model.add(Dropout(0.5)) #
	model.add(Dense(6))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()
	return model

def train_model():
	X_train_new, y_train_new = get_train_set2()
	X_v, y_v = get_train_set2(type_name='val')
	X_v = [X_v[:,:,0].reshape(X_v.shape[0], X_v.shape[1], 1), X_v[:,:,1].reshape(X_v.shape[0], X_v.shape[1], 1)]
	print X_train_new.shape

	model = branch_conv1d_mod()

	es = EarlyStopping(monitor="val_loss", mode='min', patience=5, verbose=1)
	mc = ModelCheckpoint("model_cnn_tpeak_gauss_3000_2conv.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
	#print numpy.shape(X_train_new)
	#print numpy.shape(y_train_new)
	#hist = model.fit_generator(generate_arrays_from_file(), epochs=10, steps_per_epoch=1000)
	hist = model.fit([X_train_new[:,:,0].reshape(X_train_new.shape[0], X_train_new.shape[1], 1), X_train_new[:,:,1].reshape(X_train_new.shape[0], X_train_new.shape[1], 1)], y_train_new, validation_data = (X_v, y_v), epochs=40, batch_size=100, callbacks=[es,mc])
	#hist = model.fit(X_train_new, y_train_new, epochs=40, batch_size=32, callbacks=my_callbacks, 	validation_split=0.3)
	#model.save("model_cnn_tpeak2.h5")
	#print("Saved model to disk")
	# Final evaluation of the model
	model = load_model("model_cnn_tpeak_gauss_3000_2conv.h5")
	X_val_new, y_val_new = get_train_set2(type_name='test')
	X_val_new = [X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)]
	#score = model.score(X_val_new, y_val_new)
	#print("Test Score: %.2f%%" % (score))
	results = mean_squared_error(y_val_new, model.predict(X_val_new))
	print("MSE: %.5f%%" % (results))

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = numpy.array(yhats)
	# weighted sum across ensemble members
	#summed = numpy.tensordot(yhats, weights, axes=((0),(0)))
        summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	result = numpy.argmax(summed, axis=1)
	return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(testX, nh3=False):
	members = get_members(nh3=nh3)
	weights = [1.0/len(members) for _ in range(len(members))]
	# make prediction
	yhat = ensemble_predictions(members, weights, testX)
	return yhat

def get_members(nh3=True):
	members=[]
	if nh3:
		members.append(load_model("model_cnn_3class_nh3_sep_short_valloss_GAS_0.h5"))
		members.append(load_model("model_cnn_3class_nh3_sep_short_valloss_GAS_1.h5"))
		members.append(load_model("model_cnn_3class_nh3_sep_short_valloss_GAS_2.h5"))
		members.append(load_model("model_cnn_3class_nh3_sep_short_valloss_GAS_3.h5"))
		members.append(load_model("model_cnn_3class_nh3_sep_short_valloss_GAS_4.h5"))
		members.append(load_model("model_cnn_3class_nh3_sep_short_valloss_GAS.h5"))
		#members.append(load_model("model_cnn_3class_nh3_sep_short_valacc_GAS.h5"))
	else:
		members.append(load_model("model_cnn_3class0_gauss_3000_2conv_GAS.h5"))
		members.append(load_model("model_cnn_3class1_gauss_3000_2conv_GAS.h5"))
		members.append(load_model("model_cnn_3class2_gauss_3000_2conv_GAS.h5"))
		members.append(load_model("model_cnn_3class3_gauss_3000_2conv_GAS.h5"))
		members.append(load_model("model_cnn_3class4_gauss_3000_2conv_GAS.h5"))
		members.append(load_model("model_cnn_3class5_gauss_3000_2conv_GAS.h5"))
	return members

def test_data(f='CygX_N_13CO_conv_test_smooth_clip.fits', plot=False, compare=False):
	# c is the class of the test data (0=single, 1=multi)
	data = fits.getdata(f)
	header = fits.getheader(f)
	print data.shape
	# Create a 2D array to place ouput predictions
	out_arr = data[0].copy()
	out_arr[:]=numpy.nan
	out_arr2 = out_arr.copy()
	out_arr3 = out_arr.copy()
	out_arr4 = out_arr.copy()
	out_arr5 = out_arr.copy()
	out_arr6 = out_arr.copy()
	
	window_shape = [data.shape[0],3,3]
	X_val_new = []
	X_val_full = []
	indices = []
	Tmax = []
	for index, x in numpy.ndenumerate(data[0]):
		z = data[:, index[0]-1:index[0]+2, index[1]-1:index[1]+2]
		if z.shape==(500, 3,3):
			indices.append(index)
			local0 = z[:,1,1].reshape(500,1) # central pixel
			Tmax.append(numpy.max(local0))
			local0 = local0/numpy.max(local0)
			local1 = numpy.mean(z[:,:,:].reshape(500,9), axis=1) #3x3 pixel average
			local1 = local1/numpy.max(local1)
			#if max(local1)/numpy.std(local0[0:50])>6.0:
			#	plt.plot(range(len(local1)), local1)
			#	plt.plot(range(len(glob1)), glob1, alpha=0.5)
			#	plt.show()
			z = numpy.column_stack((local0,local1))
			X_val_new.append(z)
	X_val_new = numpy.array(X_val_new)
	indices = numpy.array(indices)

	print X_val_new.shape

	#count = 0
	#for i in X_val_new:
	#	X_val_new[count] = i*(1/numpy.max(i))

	# load model
	new_model = load_model("model_cnn_tpeak_gauss_3000_2conv.h5")
	print "Loaded model from disk"

	# load model
	#model = load_model("model_cnn_3class0_gauss_3000_2conv_GAS.h5")
	#print "Loaded model from disk"
	
	# Make prediction on each pixel and output as 2D fits image
	#pred_class = model.predict([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)], verbose=0)
	pred_class = evaluate_ensemble([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)], nh3=False)
	
	# Make prediction on each pixel and output as 2D fits image
	predictions = new_model.predict([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)], verbose=0)

	# Reshape to get back 2D structure
	xax = numpy.linspace(-1,1, 500)
	step = header['CDELT3']
	cubeax = numpy.array(SpectralCube.read(f).spectral_axis)
	counter=0
	for i,j,k in zip(predictions,indices, pred_class):
		ind = int(k)
		#ind = numpy.argmax(k)
		if ind==2:
			out_arr[j[0], j[1]] = (max(cubeax)-min(cubeax))*(i[0]-abs(-1))/(1--1) + max(cubeax) 
			out_arr2[j[0],j[1]] = (max(cubeax)-min(cubeax))*(i[1]-abs(-1))/(1--1) + max(cubeax)
			out_arr3[j[0], j[1]] = i[2]*step 
			out_arr4[j[0],j[1]] = i[3]*step
			out_arr5[j[0], j[1]] = i[4]*Tmax[counter]
			out_arr6[j[0], j[1]] = i[5]*Tmax[counter]
		counter+=1
	# Format 3D header for 2D data
	del header['NAXIS3']
	#del header['LBOUND3']
	#del header['OBS3']
	del header['CRPIX3']
	del header['CDELT3']
	del header['CUNIT3']
	header['WCSAXES']=2
	del header['CTYPE3']
	del header['CRVAL3']
	# Write to fits file
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_vlsr1.fits', data=out_arr, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_vlsr2.fits', data=out_arr2, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_sig1.fits', data=out_arr3, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_sig2.fits', data=out_arr4, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_tpeak1.fits', data=out_arr5, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_tpeak2.fits', data=out_arr6, header=header, overwrite=True)
	if compare:
		#numpy.flip(X_val_new,axis=1)
		for i,j,k in zip(X_val_new, predictions, pred_class):
			ind = numpy.argmax(k)
			spec = i[:,0]
			max_ch = np.argmax(spec)
			Tpeak = spec[max_ch]
			# Use the velocity of the brightness Temp peak as 
			# initial guess for Gaussian mean
			vpeak = numpy.linspace(-1,1, len(spec))[max_ch]
			err = np.std(np.append(spec[0:50], spec[-50:]))
			if (ind==2) and ((numpy.max(spec)/err)>30):
				xaxis = numpy.linspace(-1,1, len(spec))
				step = abs(xaxis[23]-xaxis[24]) 
				coeffs = get_guess(vpeak, xaxis, spec, err)
				print coeffs
				gauss2 = np.array(p_eval2(xaxis,coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]))	
				if plot:
					V1=numpy.min([coeffs[1],coeffs[4]])
					V2=numpy.max([coeffs[1],coeffs[4]])
					W1=[coeffs[2], coeffs[5]][numpy.argmin([coeffs[1],coeffs[4]])]
					W2=[coeffs[2], coeffs[5]][numpy.argmax([coeffs[1],coeffs[4]])]
					plt.plot(xaxis, spec)
					plt.scatter([V1, V2], [0.05,0.15], color='black', label='$\chi^2$-fit', marker='.')
					plt.scatter([j[0],j[1]], [0,0.1], color='orange', zorder=10, alpha=0.7, label='CNN-prediction', marker='.')
					plt.plot([j[0]-(j[2]*step/2), j[0]+(j[2]*step/2)], [0,0], color='orange')
					plt.plot([j[1]-(j[3]*step/2), j[1]+(j[3]*step/2)], [0.1,0.1], color='orange')
					plt.plot([V1-(W1/2), V1+(W1/2)], [0.05,0.05], color='black')
					plt.plot([V2-(W2/2), V2+(W2/2)], [0.15,0.15], color='black')
				
					gauss = p_eval2(xaxis, j[4], j[0], j[2]*step, j[5], j[1], j[3]*step)
					plt.plot(xaxis, gauss2, linestyle='dotted', color='black', label='$\chi^2$-fit', zorder=11)
					plt.plot(xaxis, gauss, linestyle='-', color='orange', label='CNN-prediction', zorder=9)
					
					plt.xlabel('Normalized Velocity')
					plt.ylabel('Normalized Intensity')
					plt.xlim((-0.3,0.5))
					plt.legend()
					plt.show()
def get_guess(vpeak, xaxis, spec, err):
	guesses = []
	err1 = numpy.zeros(len(spec))+err
	for xx in numpy.arange(0.1,0.3, 0.01):
		g1 = [1.0, vpeak, 0.01, 0.9, vpeak-xx, 0.05]
		g2 = [1.0, vpeak, 0.01, 0.9, vpeak+xx, 0.05]			
		guesses.append(g1)
		guesses.append(g2)

	gausses = []
	coeffs_out = []
	for gg in guesses:
		try:
			coeffs, covar_mat = curve_fit(p_eval2, xdata=xaxis, ydata=spec, p0=gg, maxfev=5000, sigma=err1, bounds=((0.001,-1, 0.001, 0.001,-1, 0.001), (10, 1, 100, 10, 1, 100)))
			try:
				gauss2 = np.array(p_eval2(xaxis,coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]))
				gausses.append(gauss2)
				coeffs_out.append(coeffs)
				params2 = numpy.ones(6)
			except IndexError:
				gausses.append(numpy.zeros(500))
				coeffs_out.append([0.,0.,0.,0.,0.,0.])	
		except RuntimeError:
			gausses.append(numpy.zeros(500))
			coeffs_out.append([0.,0.,0.,0.,0.,0.])
			#params2 = nan_array
	chis = []
	for xx in gausses:
		chi2 = aic(spec,xx,deg=6)
		chis.append(chi2)
	coeffs_best = coeffs_out[numpy.argmin(numpy.array(chis))]
	return coeffs_best

# Define the double Gaussian profile
def p_eval2(x, a, x0, sigma, a1, x1, sigma1):
	one = a*np.exp(-(x-x0)**2/(2*sigma**2))
	two = a1*np.exp(-(x-x1)**2/(2*sigma1**2))
	return one+two

def aic(ydata,ymod,deg=2,sd=None):  
      # Select only the channels where the model is non-zero

      # Chi-square statistic  
      if sd==None:  
           chisq=numpy.sum((ydata-ymod)**2)  
      else:  
           chisq=numpy.sum( ((ydata-ymod)**2/(sd**2)) )  
             
      # Number of degrees of freedom assuming 2 free parameters  
      #nu=ydata.size-1-deg 

      #AIC = 2*deg - 2*numpy.log(chisq) 
      AIC = chisq + 2*deg

      AICc = AIC + ((2.*deg*(deg+1)) / (len(ydata) - deg - 1.))
        
      return AICc

# Predicting Parameters for Real Cubes
#test_data(f='B18_HC5N_conv_test_smooth_clip.fits')
#test_data(f='CygX_N_13CO_conv_test_smooth_clip.fits')
#test_data(f='CygX_N_C18O_conv_test_smooth_clip2.fits')
#test_data(f='Oph_13CO_conv_test_smooth_clip.fits')
#test_data(f='Oph2_13CO_conv_test_smooth_clip.fits', plot=False, compare=False)
#test_data(f='W3Main_C18O_conv_test_smooth_clip.fits')
#test_data(f='NGC7538_C18O_conv_test_smooth_clip.fits')

# Load training data and reshape
#train_model()

# Test trained network on test set
#test_model()

# Compare network performance with traditional method
#test_chi(use_cnn=True)
#test_chi(use_cnn=False)
