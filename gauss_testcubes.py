import pyspeckit.spectrum.models.inherited_gaussfitter as ig
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis
import os
import sys
import numpy as np
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
from astropy import log
import matplotlib.pyplot as plt
import h5py
import numpy
log.setLevel('ERROR')

# Load in the spectral axis in km/s

def generate_cubes(nCubes=90, nBorder=1, noise_rms=0.1, fix_vlsr=False,
	                   random_seed=None, remove_low_sep=False, remove_high_sep=False, remove_low_snr=False, set_name='GAS_reg', regression=False, plot=False):
    """
    This places nCubes random cubes into the specified output directory
    """

    #if not os.path.isdir(output_dir):
    #    os.mkdir(output_dir)
    cube_km = SpectralCube.read('random_cube_NH3_11_0.fits')
    xarr11 = cube_km.with_spectral_unit(u.km / u.s, velocity_convention='radio').spectral_axis.value[250:-250]

    xarr11 = numpy.flip(xarr11, axis=0) #flip so negative VLSR on left

    #xarr11 = spaxis((np.linspace(-500, 499, 500) * 5.72e-6
     #                + nh3con.freq_dict['oneone'] / 1e9),
      #              unit='GHz',
       #             refX=nh3con.freq_dict['oneone'] / 1e9,
        #            velocity_convention='radio', refX_unit='GHz')
    #xarr22 = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
    #                 + nh3con.freq_dict['twotwo'] / 1e9), unit='GHz',
    #                refX=nh3con.freq_dict['twotwo'] / 1e9,
    #                velocity_convention='radio', refX_unit='GHz')

    out_arr = []
    out_vels = []

    nDigits = int(np.ceil(np.log10(nCubes)))
    if random_seed:
        np.random.seed(random_seed)
    # Check that ncubes divisible by 3
    if (nCubes % 3 != 0):
	print 'Warning: Number of cubes not divisible by 3'
	print 'Setting ncubes to 90'
	nCubes = 90
    # Ensure output is balanced between each class
    nComps = np.concatenate((np.ones(nCubes/3).astype(int), np.zeros(nCubes/3).astype(int), np.zeros(nCubes/3).astype(int)+2))
    if regression:
	nComps = np.ones(nCubes)+1
    out_y1 = np.where(np.array(nComps)==1, 1, 0)
    out_y2 = np.where(np.array(nComps)==0, 1, 0)
    out_y3 = np.where(np.array(nComps)==2, 1, 0)

    mu_range_spectral = [-10, 10]
    sigma_range_spectral = [0.15, 0.8]
    #Width1NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))
    #Width2NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))

    #sigma1 = np.sqrt(Width1NT + 0.08**2)
    #sigma2 = np.sqrt(Width2NT + 0.08**2)
    #peak_range_spectral = [0.4, 6.0]
    mn = numpy.random.uniform(0.05, 0.25, size=nCubes) #rms noise added .01 - 0.1 # 0.05 - 0.2
    Voff1 = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(nCubes))
    sigma1 = numpy.random.uniform(sigma_range_spectral[0], sigma_range_spectral[1], size=(nCubes))
    #peak1 = numpy.random.uniform(peak_range_spectral[0], peak_range_spectral[1], size=(nCubes))
    #peak1 = numpy.random.uniform(0.25*3, 0.05*40, size=(nCubes))
    #peak1 = mn*3 + np.random.rand(nCubes) * mn*20
    peak1 = numpy.ones(nCubes)

    xoff = np.random.randint(2, size=nCubes)
    xoff[xoff==0]=-1
    sigma2 = numpy.random.uniform(sigma_range_spectral[0], sigma_range_spectral[1], size=(nCubes))
    Voff2 = Voff1 + numpy.max(numpy.column_stack((sigma1, sigma2)),axis=1)*numpy.random.uniform(1.5, 5, size=(nCubes))*xoff
    #Voff2 = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(nCubes))
    #peak2 = numpy.random.uniform(peak_range_spectral[0], peak_range_spectral[1], size=(nCubes))
    #peak2 = numpy.random.uniform(0.25*3, 0.05*40, size=(nCubes))
    #peak2 = mn*3 + np.random.rand(nCubes) * mn*20
    peak2 = numpy.random.uniform(2*mn, 1.0, size=(nCubes))

    #print numpy.where(numpy.absolute(peak1-peak2)<mn)

    #print np.where(np.abs(Voff1-Voff2)<np.max(np.column_stack((Width1, Width2)), axis=1))[0]

    if remove_low_sep:
        # Find where centroids are too close
        too_close = np.where(np.abs(Voff1-Voff2)<np.max(np.column_stack((sigma1, sigma2)), axis=1))
        # Move the centroids farther apart by the length of largest line width 
        min_Voff = np.min(np.column_stack((Voff2[too_close],Voff1[too_close])), axis=1)
        max_Voff = np.max(np.column_stack((Voff2[too_close],Voff1[too_close])), axis=1)
        Voff1[too_close]=min_Voff-np.max(np.column_stack((sigma1[too_close], sigma2[too_close])), axis=1)/2.
        Voff2[too_close]=max_Voff+np.max(np.column_stack((sigma1[too_close], sigma2[too_close])), axis=1)/2.

    if remove_high_sep:
        # Find where centroids are too close
        too_far = np.where(np.abs(Voff1-Voff2)>5.0)
        # Move the centroids farther apart by the length of largest line width 
        Voff2[too_far]=Voff1[too_far]-sigma2[too_far]*numpy.random.uniform(2, 3, size=(len(too_far)))

    if remove_low_snr:
	low_snr = np.where(np.min(np.column_stack((peak1,peak2)), axis=1)/mn<4.0)
	mn[low_snr] = np.min(np.column_stack((peak1,peak2)), axis=1)[low_snr]/4.0

    #print Voff1[too_close]
    #print Voff2[too_close]

    #print len(np.where(np.abs(Voff1-Voff2)<np.max(np.column_stack((Width1, Width2)), axis=1))[0])

    # Normalize centroids between -1 and 1
    Voff1_norm = 2*((Voff1-min(xarr11))/(max(xarr11)-min(xarr11)))-1 
    Voff2_norm = 2*((Voff2-min(xarr11))/(max(xarr11)-min(xarr11)))-1

    # Normalize dispersion to be number of channels
    step = abs(xarr11[23]-xarr11[24])
    sigma1_norm = sigma1/step
    sigma2_norm = sigma2/step

    scale = np.array([[0.05, 0.05, 0.05]]) # Tpeak, Sig, Voff
    gradX1 = np.random.randn(nCubes, 3) * scale
    gradY1 = np.random.randn(nCubes, 3) * scale
    gradX2 = np.random.randn(nCubes, 3) * scale
    gradY2 = np.random.randn(nCubes, 3) * scale

    for i in ProgressBar(range(nCubes)):
        xmat, ymat = np.indices((2 * nBorder + 1, 2 * nBorder + 1))
        cube11 = np.zeros((xarr11.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))

        for xx, yy in zip(xmat.flatten(), ymat.flatten()):
            T1 = peak1[i] * (1 + gradX1[i, 0] * (xx - 1)
                             + gradY1[i, 0] * (yy - 1))
            T2 = peak2[i] * (1 + gradX2[i, 0] * (xx - 1)
                             + gradY2[i, 0] * (yy - 1))

            W1 = np.abs(sigma1[i] * (1 + gradX1[i, 1] * (xx - 1)
                                     + gradY1[i, 1] * (yy - 1)))
            W2 = np.abs(sigma2[i] * (1 + gradX2[i, 1] * (xx - 1)
                                     + gradY2[i, 1] * (yy - 1)))
            V1 = Voff1[i] + (gradX2[i, 2] * (xx - 1) + gradY2[i, 2] * (yy - 1))
            V2 = Voff2[i] + (gradX2[i, 2] * (xx - 1) + gradY2[i, 2] * (yy - 1))

            if nComps[i] == 1:
                spec11 = ig.gaussian(xarr11, T1,V1,W1)
                #spec22 = ammonia.cold_ammonia(xarr22, T1,
                #                              ntot=N1,
                #                              width=W1,
                #                              xoff_v=V1)
                if (xx == nBorder) and (yy == nBorder):
                    Tmax11a = np.max(spec11)
		    Tmax11b = 0
                    Tmax11 = np.max(spec11)
            if nComps[i] == 2:
                spec11a = ig.gaussian(xarr11, T1,V1,W1)
                spec11b = ig.gaussian(xarr11, T2,V2,W2)
                spec11 = spec11a + spec11b

                if (xx == nBorder) and (yy == nBorder):
                    Tmax11a = np.max(spec11a)
                    Tmax11b = np.max(spec11b)
                    Tmax11 = np.max(spec11)
            if nComps[i]==0:
	        cube11[:, yy, xx] = np.zeros(500)
	    else:
		cube11[:, yy, xx] = spec11
            #cube22[:, yy, xx] = spec22
	if nComps[i]!=0:
		cube11 /= np.max(cube11)
        cube11 += np.random.randn(*cube11.shape) * mn[i]
        #cube22 += np.random.randn(*cube22.shape) * noise_rms
	
	loc11 = cube11[:,1,1].reshape(500,1)
	loc11 = loc11/np.max(loc11)
	glob11 = np.mean(cube11.reshape(500,9),axis=1)
	glob11 = glob11/np.max(glob11)

	loc11 = cube11[:,1,1].reshape(500,1)
	T1 = Tmax11a/np.max(loc11)
	T2 = Tmax11b/np.max(loc11)
	loc11 = loc11/np.max(loc11)
	glob11 = np.mean(cube11.reshape(500,9),axis=1)
	glob11 = glob11/np.max(glob11)
	z = np.column_stack((loc11,glob11))
	out_arr.append(z)
	V1 = Voff1_norm[i]
	V2 = Voff2_norm[i]
	W1 = sigma1_norm[i]
	W2 = sigma2_norm[i]
	
	out_vel = [min([V1,V2]), max([V1,V2]), [W1,W2][np.argmin([V1,V2])], [W1,W2][np.argmax([V1,V2])], [T1,T2][np.argmin([V1,V2])], [T1,T2][np.argmax([V1,V2])]]
        out_vels.append(out_vel)
	if plot and nComps[i]==2:
		if nComps[i]==1:
			tt = 'One-Component'
		elif nComps[i]==2:
			tt = 'Two-Component'
		else:
			tt = 'Noise-Only'
		plt.plot(xarr11, loc11) #, label='central pixel (local)'
		plt.plot(xarr11, glob11+1) #, label='3x3 average (global)'
		plt.errorbar([Voff1[i], Voff2[i]], [0,0.1], xerr=[sigma1[i],sigma2[i]], color='orange', fmt='none')
		plt.scatter([Voff1[i], Voff2[i]], [T1,T2], color='red', alpha=0.5)

		plt.xlabel('Synthetic $V_{LSR}$', size=14)
		plt.ylabel('Normalized Intensity', size=14)
		#plt.legend(fontsize=14)
		plt.title(tt, size=14)
		plt.show()

		#xxx = numpy.linspace(-1,1,500)
		#step = abs(xxx[1]-xxx[2])
		#plt.plot(xxx, loc11) #, label='central pixel (local)'
		#plt.plot(xxx, glob11+1) #, label='3x3 average (global)'
		#plt.errorbar([Voff1_norm[i], Voff2_norm[i]], [0,0.1], xerr=[sigma1_norm[i]*step,sigma2_norm[i]*step], color='orange', fmt='none')
		#plt.scatter([Voff1_norm[i], Voff2_norm[i]], [T1,T2], color='red', alpha=0.5)
		#plt.show()

    with h5py.File('three_class_' + set_name + '.h5', 'w') as hf:
	hf.create_dataset('data', data=np.array(out_arr))
	hf.close()
    with h5py.File('labels_three_class_' + set_name + '.h5', 'w') as hf:
	hf.create_dataset('data', data=np.column_stack((out_y1, out_y2, out_y3)))
	hf.close()
    with h5py.File('params_three_class_' + set_name + '.h5', 'w') as hf:
	hf.create_dataset('data', data=np.array(out_vels))
	hf.close()
    #print out_y, np.column_stack((out_y1, out_y2, out_y3))

#if __name__ == '__main__':
#    print(sys.argv)
#    if len(sys.argv) > 1:
#        generate_cubes(nCubes=int(sys.argv[1]))
#    else:
#        generate_cubes()

#generate_cubes(nCubes=300000, set_name='gauss_train1', regression=False, plot=False)
#generate_cubes(nCubes=90000, set_name='gauss_val1', regression=False)
#generate_cubes(nCubes=30000, set_name='gauss_test1', regression=False, plot=False)
generate_cubes(nCubes=300000, set_name='gauss_test2', regression=False, plot=False)

#generate_cubes(nCubes=300000, set_name='gauss_train1_reg', regression=True, plot=False)
#generate_cubes(nCubes=90000, set_name='gauss_val1_reg', regression=True)
#generate_cubes(nCubes=30000, set_name='gauss_test1_reg', regression=True, plot=False)
