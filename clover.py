#from test_cnn_3class import test_data as predict_class
#from train_cnn_tpeak import test_data as predict_gauss_reg
#from train_cnn_reg_nh3 import test_data as predict_nh3_reg
from parallel_predict import *
from spectral_cube import SpectralCube
import numpy

def check_spec(f, length):
	cube = SpectralCube.read(f)
	spec_length = len(cube.spectral_axis)	
	#print cube.shape
	out = numpy.zeros((length, cube.shape[1], cube.shape[2]))
	out[:] = numpy.nan
	if spec_length<length:
		print 'Spectral axis smaller than ' + str(length) + ' channels'
		print 'Correcting Spectral Axis'
		to_add = length - spec_length
		for index, x in numpy.ndenumerate(cube[0,:,:]):
			spec = cube[:, index[0], index[1]]
			rms = numpy.std(numpy.concatenate([spec[0:50], spec[-50:]]))
			noise = numpy.random.normal(scale=rms, size=to_add)
			spec2 = numpy.concatenate([noise[0:int(len(noise)/2)], spec, noise[int(len(noise)/2):]])
			out[:, index[0], index[1]] = spec2
		# correct the header variables
		cube.header['NAXIS3'] = length
		# Move CRPIX3 over by the number of channels
		# added to left of spectrum
		cube.header['CRPIX3'] = cube.header['CRPIX3']+int(len(noise)/2)
		cube = SpectralCube(data=out, wcs=cube.wcs)
		f = f.split('.fits')[0]+'_clover.fits'
		cube.write(f, overwrite=True)	
	elif spec_length>length:
		to_remove = spec_length-length
		cube = cube[int(to_remove/2.):-int(round(to_remove/2.)), :, :]
		f = f.split('.fits')[0]+'_clover.fits'
		cube.write(f, overwrite=True)
	return f

def predict(f='Oph2_13CO_conv_test_smooth_clip.fits', nh3=False, nproc=3):
	if nh3:
		type_name = 'nh3'
		f = check_spec(f, length=1000)
	else:
		type_name = 'branch'
		f = check_spec(f, length=500)
	
	# Segment cube into one-comp, two-comp, and noise
	#print 'Predicting Class of Each Pixel...'
	#predict_class(f=f, type_name=type_name)

	# Segment cube into one-comp, two-comp, and noise
	# and predict kinematics of two-comp pixels
	print 'Segmenting and Predicting Kinematics...'
	test_data(f=f, nh3=nh3, nproc=nproc)

#predict()
#predict(f='M17_NH3_11_ml_test.fits', nh3=True)
#predict(f='MonR2_NH3_11_ml_test.fits', nh3=True)
#predict(f='M17_clip.fits', nh3=True)
