import numpy
import time
import pprocess
from astropy.io import fits
import numpy as np
from spectral_cube import SpectralCube
import astropy.units as u

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX, nproc):
	if nproc>6:
		nproc=6

	def para_predict(modname):
		from keras.models import load_model
		model = load_model(modname)
		prediction = model.predict(testX)
		return prediction

	queue = pprocess.Queue(limit=nproc)
	calc = queue.manage(pprocess.MakeParallel(para_predict))

	yhats = []
	for i in members:
		calc(i)
		#yhats.append(para_predict(i,testX))

	for preds in queue:
		yhats.append(preds)

	# make predictions
	#yhats = [model.predict(testX) for model in members]
	yhats = numpy.array(yhats)
	# weighted sum across ensemble members
	#summed = numpy.tensordot(yhats, weights, axes=((0),(0)))
        summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	result = numpy.argmax(summed, axis=1)
	return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(testX, nh3=False, nproc=2):
	#members = get_members(nh3=nh3)
	if nh3:
		members = ["model_cnn_3class_nh3_sep_short_valloss_GAS_"+str(i)+".h5" for i in range(5)]
		members.append('model_cnn_3class_nh3_sep_short_valloss_GAS.h5')
	else:
		members = ["model_cnn_3class" + str(i) + "_gauss_3000_2conv_GAS.h5" for i in range(6)]
	#weights = [1.0/len(members) for _ in range(len(members))]
	# make prediction
	yhat = ensemble_predictions(members, testX, nproc)
	return yhat

def test_data(f='CygX_N_13CO_conv_test_smooth_clip.fits', nh3=False, nproc=2):
	if nh3:
		speclen = 1000
		regmod = "model_cnn_reg_nh3_1000_2conv.h5"
	else:
		speclen = 500
		regmod = "model_cnn_tpeak_gauss_3000_2conv.h5"

	tic = time.time()
	# c is the class of the test data (0=single, 1=multi)
	data = fits.getdata(f)
	header = fits.getheader(f)
	#print data.shape
	# Create a 2D array to place ouput predictions
	out_arr = data[0].copy()
	out_arr[:]=numpy.nan
	out_arr2 = out_arr.copy()
	out_arr3 = out_arr.copy()
	out_arr4 = out_arr.copy()
	out_arr5 = out_arr.copy()
	out_arr6 = out_arr.copy()
	out_class = out_arr.copy()
	
	window_shape = [data.shape[0],3,3]
	X_val_new = []
	X_val_full = []
	indices = []
	Tmax = []
	for index, x in numpy.ndenumerate(data[0]):
		z = data[:, index[0]-1:index[0]+2, index[1]-1:index[1]+2]
		if z.shape==(speclen, 3,3) and (numpy.isnan(z.flatten()).sum()==0):
			indices.append(index)
			local0 = z[:,1,1].reshape(speclen,1) # central pixel
			Tmax.append(numpy.max(local0))
			local0 = local0/numpy.max(local0)
			local1 = numpy.mean(z[:,:,:].reshape(speclen,9), axis=1) #3x3 pixel average
			local1 = local1/numpy.max(local1)
			#if max(local1)/numpy.std(local0[0:50])>6.0:
			#	plt.plot(range(len(local1)), local1)
			#	plt.plot(range(len(glob1)), glob1, alpha=0.5)
			#	plt.show()
			z = numpy.column_stack((local0,local1))
			X_val_new.append(z)
	X_val_new = numpy.array(X_val_new)
	indices = numpy.array(indices)

	print 'Number of pixels to predict:' + str(len(indices))
	
	# Make prediction on each pixel and output as 2D fits image
	#pred_class = model.predict([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)], verbose=0)
	pred_class = evaluate_ensemble([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)], nh3=nh3, nproc=nproc)
	
	# Make prediction on each pixel and output as 2D fits image
	# load model
	from keras.models import load_model
	new_model = load_model(regmod)
	#print "Loaded model from disk"
	predictions = new_model.predict([X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)], verbose=0)

	# Reshape to get back 2D structure
	cubeax = numpy.array(SpectralCube.read(f).with_spectral_unit(u.km / u.s, velocity_convention='radio').spectral_axis)
	step_real = abs(cubeax[23]-cubeax[24])

	if nh3:
		cube_km = SpectralCube.read('random_cube_NH3_11_0.fits')
		xax = cube_km.with_spectral_unit(u.km / u.s, velocity_convention='radio').spectral_axis.value
		# Normalize predictions between -1 and 1
    		predictions[:,0] = 2*((predictions[:,0]-min(xax))/(max(xax)-min(xax)))-1 
    		predictions[:,1] = 2*((predictions[:,1]-min(xax))/(max(xax)-min(xax)))-1
		step = abs(xax[23]-xax[24])
		factor = step_real/step
	else:
		factor = step_real

	counter=0
	for i,j,k in zip(predictions,indices, pred_class):
		ind = int(k)
		out_class[j[0], j[1]] = ind
		#ind = numpy.argmax(k)
		if ind==2:
			out_arr[j[0], j[1]] = (max(cubeax)-min(cubeax))*(i[0]-abs(-1))/(1--1) + max(cubeax) 
			out_arr2[j[0],j[1]] = (max(cubeax)-min(cubeax))*(i[1]-abs(-1))/(1--1) + max(cubeax)
			out_arr3[j[0], j[1]] = i[2]*factor 
			out_arr4[j[0],j[1]] = i[3]*factor
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
	fits.writeto(f.split('.fits')[0]+'_class.fits', data=out_class, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_vlsr1.fits', data=out_arr, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_vlsr2.fits', data=out_arr2, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_sig1.fits', data=out_arr3, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_sig2.fits', data=out_arr4, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_tpeak1.fits', data=out_arr5, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_tpeak2.fits', data=out_arr6, header=header, overwrite=True)

	print "\n %f s for computation." % (time.time() - tic)

#test_data(f='Oph2_13CO_conv_test_smooth_clip.fits', nproc=3)
#test_data(f='M17_NH3_11_ml_test.fits', nh3=True, nproc=3)
