import matplotlib.pyplot as plt
from astropy.io import fits
import numpy
import numpy as np
from scipy.stats import multivariate_normal
from spectral_cube import SpectralCube
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import sys
import h5py
from sklearn.utils import shuffle

# Setup the grid for 3D cubes: 
# 11x11 pixels with 500 spectral channels
#x, y, z = np.mgrid[-1.0:1.0:500j, -1.0:1.0:10j, -1.0:1.0:10j]
x, y, z = np.mgrid[-1.0:1.0:500j, -1.0:1.0:11j, -1.0:1.0:11j]

# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat, z.flat])

def get_train_single(num=100, multi=False, noise_only=False, close=False, local_views=False):
	# Creates a 10x10 sythetic spectral cube used as a single training example
	# Either a single gaussian, or two gaussians are injected into the frame
	# num: sets number of samples to create
	# multi: True if two gaussians are to be injected
	# noise_only: True if sample is just noise (no gaussian injected)
	# close: True if multi-comp sample is to have a small spectral velocity difference
	# local_views: if True, output only two spectra, one from the center pixel in the window
	# 	and the second an averaged spectrum from a 3x3 window around the center pixel

	if multi:
		filename = '3d_gauss_train_multi.h5'
	else:
		filename = '3d_gauss_train.h5'
	f = file(filename, mode='w')
	f.close()
	print "Creating Training Samples..."
	# Set the limits of the 3D gaussian parameters
	# sample parameters drawn randomly from these limits
	mu_range_spectral = [-0.6, 0.6]
	sigma_range_spectral = [0.01, 0.07]
	mu_range_xy = [-0.01, 0.01]
	sigma_range_xy = [0.8, 1.0]

	# Randomly select parameters for the samples
	mu_spectral = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(num,1))
	mu_xy = numpy.random.uniform(mu_range_xy[0], mu_range_xy[1], size=(num, 2))
	mu = numpy.column_stack((mu_spectral, mu_xy))

	sigma_spectral = numpy.random.uniform(sigma_range_spectral[0], sigma_range_spectral[1], size=(num,1))
	sigma_xy = numpy.random.uniform(sigma_range_xy[0], sigma_range_xy[1], size=(num, 2))
	sigma = numpy.column_stack((sigma_spectral, sigma_xy))

	# Make the second component zero if single-component sample
	mu2 = numpy.zeros(num)
	sigma2 = numpy.zeros(num)

	if multi:
		if not close:
			mu_spectral = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(num,1))
			mu_xy = numpy.random.uniform(mu_range_xy[0], mu_range_xy[1], size=(num, 2))
			mu2 = numpy.column_stack((mu_spectral, mu_xy))
		else:
			#mu_spectral = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(num,1))
			mu_xy = numpy.random.uniform(mu_range_xy[0], mu_range_xy[1], size=(num, 2))
			mu2 = numpy.column_stack((mu_spectral+sigma_spectral*numpy.random.uniform(2, 3, size=(num,1)), mu_xy))

		sigma_spectral = numpy.random.uniform(sigma_range_spectral[0], sigma_range_spectral[1], size=(num,1))
		sigma_xy = numpy.random.uniform(sigma_range_xy[0], sigma_range_xy[1], size=(num, 2))
		sigma2 = numpy.column_stack((sigma_spectral, sigma_xy))

	# Loop through parameters and generate 3D cubes
	counter = 0
	out = []
	mu_out = mu[:,0]
	sig_out = sigma[:,0]
	if multi:
		mu_out2 = mu2[:,0]
		sig_out2 = sigma2[:,0]
	else:
		mu_out2 = []
		sig_out2 = []
	for mu, sigma, mu2, sigma2, ind in zip(mu, sigma, mu2, sigma2, range(num)):
		z = grab_single(mu, sigma, mu2, sigma2, ind, multi, filename, noise_only=noise_only, local_views=local_views)
		counter+=1
		out.append(z)
		print str(counter) + ' of ' + str(num) + ' samples completed \r',
		sys.stdout.flush()
	#numpy.save(filename, numpy.array(out))
	return out, mu_out, mu_out2, sig_out, sig_out2

def grab_single(mu, sigma, mu2, sigma2, ind, multi=False, filename=False, noise_only=False, local_views=False):
	# Takes input gaussian parameters and generates a 3D cube
	covariance = np.diag(sigma**2)
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

	if multi:
		covariance2 = np.diag(sigma2**2)
		z2 = multivariate_normal.pdf(xy, mean=mu2, cov=covariance2)
		z = z+z2
		tx = '3d_gauss_train_multi/test'
	else:
		tx = '3d_gauss_train/test'

	z = z.reshape(x.shape)
	z = z*(1/numpy.max(z))
	if noise_only:
		z=z*0.
	z = add_noise(z)
	#z = z*(1/numpy.max(z))
	if local_views:
		local0 = z[:,5,5].reshape(500,1) # central pixel
		local0 = local0/numpy.max(local0) # normalize max to 1
		local1 = numpy.mean(z[:,4:7,4:7].reshape(500,9), axis=1) #3x3 pixel average
		local1 = local1/numpy.max(local1) # normalize max to 1
		#plt.plot(range(len(local1)), local1)
		#plt.show()
		z = numpy.column_stack((local0,local1))
	#f = file(filename, mode='a')
	#numpy.save(f, z)
	#f.close()
	#fits.writeto(tx+ str(ind) +'.fits', data=z, header=None, overwrite=True)
	return z

def add_noise(z, max_noise=0.5):
	# Adds noise to each synthetic spectrum
	# First, randomly select the noise strength (up to max_noise)
	mn = numpy.random.uniform(0.05, max_noise, size=1)
	for (i,j), value in np.ndenumerate(z[0]):
		# Next, add random noise with max strength as chosen above
		#noise=np.random.uniform(-mn[0],mn[0],len(z[:,i,j]))
		noise = np.random.randn(len(z[:,i,j])) * mn
		z[:,i,j] = z[:,i,j] + noise
	return z

def save_train_multi(size=50000, output='train_small', outdir='/nfs/copiapo1/home/jkeown/ml_multi_comp/three_class/', local_views=True):
	out0, mu1, mu2, s1, s2 = get_train_single(num=size, noise_only=False, local_views=local_views) # single class
	out1, mu1, mu2, s1, s2 = get_train_single(num=size, noise_only=True, local_views=local_views) # noise class 
	out2, mu1, mu2, s1, s2 = get_train_single(num=size/2, multi=True, noise_only=False, local_views=local_views) # multi_class
	# Add some closely separated velocity samples to multi component class
	out22, mu1, mu2, s1, s2 = get_train_single(num=size/2, multi=True, noise_only=False, close=True, local_views=local_views)
	out2.extend(out22)
	out0.extend(out1)
	out0.extend(out2)

	# Save training examples to h5 files for quicker loading
	# Classes are [single, noise, multi]
	
	with h5py.File(outdir+output+'_three_class.h5', 'w') as hf:
		hf.create_dataset('data', data=numpy.array(out0))
		hf.close()
	with h5py.File(outdir+'labels_'+ output +'_three_class.h5', 'w') as hf:
		d0=numpy.concatenate((numpy.ones(size), numpy.zeros(size), numpy.zeros(size)))
		d1=numpy.concatenate((numpy.zeros(size), numpy.ones(size), numpy.zeros(size)))
		d2=numpy.concatenate((numpy.zeros(size), numpy.zeros(size), numpy.ones(size)))
		hf.create_dataset('data', data=numpy.column_stack((d0,d1,d2)))
		hf.close()
	del out0
	del out2
	del out1 

def save_train_regression(size=50000, output='train_regression', outdir='/nfs/copiapo1/home/jkeown/ml_multi_comp/three_class/', local_views=True):
	out2, mu1, mu2, s1, s2 = get_train_single(num=size/2, multi=True, noise_only=False, local_views=local_views) # multi class
	# Add some closely separated velocity samples to multi component class
	out22, mu11, mu22, s11, s22 = get_train_single(num=size/2, multi=True, noise_only=False, close=True, local_views=local_views)
	out2.extend(out22)
	mu1 = numpy.append(mu1,mu11)
	mu2 = numpy.append(mu2,mu22)
	s1 = numpy.append(s1,s11)
	s2 = numpy.append(s2,s22)
	mu = numpy.column_stack((mu1,mu2,s1,s2))
	# Save training examples to h5 files for quicker loading
	# Classes are [single, multi, noise]
	
	with h5py.File(outdir+output+'_three_class.h5', 'w') as hf:
		hf.create_dataset('data', data=numpy.array(out2))
		hf.close()
	with h5py.File(outdir+'labels_'+ output +'_three_class.h5', 'w') as hf:
		hf.create_dataset('data', data=mu)
		hf.close()
	del out2

#save_train_regression(500000, local_views=True)
#save_train_regression(10000, output='test_regression', local_views=True)
#save_train_regression(500000, local_views=True)
#save_train_regression(10000, output='test_regression', local_views=True)

#save_train_multi(100000, output='train_gauss')
#save_train_multi(10000, output='test_gauss')

#save_train_multi(100000, output='test_10')
save_train_multi(30000, output='val')

