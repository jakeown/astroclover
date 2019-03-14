import pyspeckit.spectrum.models.ammonia as ammonia
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
log.setLevel('ERROR')

# Load in the spectral axis in km/s
cube_km = SpectralCube.read('random_cube_NH3_11_0.fits')
xax = cube_km.with_spectral_unit(u.km / u.s, velocity_convention='radio').spectral_axis.value

def generate_cubes(nCubes=90, nBorder=1, noise_rms=0.1, fix_vlsr=False,
	                   random_seed=None, remove_low_sep=True, set_name='GAS_reg', regression=False):
    """
    This places nCubes random cubes into the specified output directory
    """

    #if not os.path.isdir(output_dir):
    #    os.mkdir(output_dir)
    xarr11 = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                     + nh3con.freq_dict['oneone'] / 1e9),
                    unit='GHz',
                    refX=nh3con.freq_dict['oneone'] / 1e9,
                    velocity_convention='radio', refX_unit='GHz')
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

    Temp1 = 8 + np.random.rand(nCubes) * 17
    Temp2 = 8 + np.random.rand(nCubes) * 17

    if fix_vlsr:
        Voff1 = np.zeros(nCubes)
    else:
	Voff1 = np.random.rand(nCubes) * 5 - 2.5
	
    Voff2 = Voff1 + np.random.rand(nCubes) * 5 - 2.5

    logN1 = 13 + 1.5 * np.random.rand(nCubes)
    logN2 = 13 + 1.5 * np.random.rand(nCubes)

    Width1NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))
    Width2NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))

    Width1 = np.sqrt(Width1NT + 0.08**2)
    Width2 = np.sqrt(Width2NT + 0.08**2)

    #print np.where(np.abs(Voff1-Voff2)<np.max(np.column_stack((Width1, Width2)), axis=1))[0]

    if remove_low_sep:
        # Find where centroids are too close
        too_close = np.where(np.abs(Voff1-Voff2)<np.max(np.column_stack((Width1, Width2)), axis=1))
        # Move the centroids farther apart by the length of largest line width 
        min_Voff = np.min(np.column_stack((Voff2[too_close],Voff1[too_close])), axis=1)
        max_Voff = np.max(np.column_stack((Voff2[too_close],Voff1[too_close])), axis=1)
        Voff1[too_close]=min_Voff-np.max(np.column_stack((Width1[too_close], Width2[too_close])), axis=1)/2.
        Voff2[too_close]=max_Voff+np.max(np.column_stack((Width1[too_close], Width2[too_close])), axis=1)/2.

    #print Voff1[too_close]
    #print Voff2[too_close]

    #print len(np.where(np.abs(Voff1-Voff2)<np.max(np.column_stack((Width1, Width2)), axis=1))[0])

    scale = np.array([[0.2, 0.1, 0.1, 0.01]])
    gradX1 = np.random.randn(nCubes, 4) * scale
    gradY1 = np.random.randn(nCubes, 4) * scale
    gradX2 = np.random.randn(nCubes, 4) * scale
    gradY2 = np.random.randn(nCubes, 4) * scale

    params1 = [{'ntot':14,
                'width':1,
                'xoff_v':0.0}] * nCubes
    params2 = [{'ntot':14,
                'width':1,
                'xoff_v':0.0}] * nCubes

    hdrkwds = {'BUNIT': 'K',
               'INSTRUME': 'KFPA    ',
               'BMAJ': 0.008554169991270138,
               'BMIN': 0.008554169991270138,
               'TELESCOP': 'GBT',
               'WCSAXES': 3,
               'CRPIX1': 2,
               'CRPIX2': 2,
               'CRPIX3': 500,
               'CDELT1': -0.008554169991270138,
               'CDELT2': 0.008554169991270138,
               'CDELT3': 5720.0,
               'CUNIT1': 'deg',
               'CUNIT2': 'deg',
               'CUNIT3': 'Hz',
               'CTYPE1': 'RA---TAN',
               'CTYPE2': 'DEC--TAN',
               'CTYPE3': 'FREQ',
               'CRVAL1': 0.0,
               'CRVAL2': 0.0,
               'LONPOLE': 180.0,
               'LATPOLE': 0.0,
               'EQUINOX': 2000.0,
               'SPECSYS': 'LSRK',
               'RADESYS': 'FK5',
               'SSYSOBS': 'TOPOCENT'}
    truekwds = ['NCOMP', 'LOGN1', 'LOGN2', 'VLSR1', 'VLSR2',
                'SIG1', 'SIG2', 'TKIN1', 'TKIN2']

    for i in ProgressBar(range(nCubes)):
        xmat, ymat = np.indices((2 * nBorder + 1, 2 * nBorder + 1))
        cube11 = np.zeros((xarr11.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))
        #cube22 = np.zeros((xarr22.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))

        for xx, yy in zip(xmat.flatten(), ymat.flatten()):
            T1 = Temp1[i] * (1 + gradX1[i, 0] * (xx - 1)
                             + gradY1[i, 0] * (yy - 1)) + 5
            T2 = Temp2[i] * (1 + gradX2[i, 0] * (xx - 1)
                             + gradY2[i, 0] * (yy - 1)) + 5
	    if T1<2.74:
		T1=2.74
	    if T2<2.74:
		T2=2.74
            W1 = np.abs(Width1[i] * (1 + gradX1[i, 1] * (xx - 1)
                                     + gradY1[i, 1] * (yy - 1)))
            W2 = np.abs(Width2[i] * (1 + gradX2[i, 1] * (xx - 1)
                                     + gradY2[i, 1] * (yy - 1)))
            V1 = Voff1[i] + (gradX1[i, 2] * (xx - 1) + gradY1[i, 2] * (yy - 1))
            V2 = Voff2[i] + (gradX2[i, 2] * (xx - 1) + gradY2[i, 2] * (yy - 1))
            N1 = logN1[i] * (1 + gradX1[i, 3] * (xx - 1)
                             + gradY1[i, 3] * (yy - 1))
            N2 = logN2[i] * (1 + gradX2[i, 3] * (xx - 1)
                             + gradY2[i, 3] * (yy - 1))
            if nComps[i] == 1:
                spec11 = ammonia.cold_ammonia(xarr11, T1,
                                              ntot=N1,
                                              width=W1,
                                              xoff_v=V1)
                #spec22 = ammonia.cold_ammonia(xarr22, T1,
                #                              ntot=N1,
                #                              width=W1,
                #                              xoff_v=V1)
                if (xx == nBorder) and (yy == nBorder):
                    Tmax11a = np.max(spec11)
		    Tmax11b = 0
                    Tmax11 = np.max(spec11)
            if nComps[i] == 2:
                spec11a = ammonia.cold_ammonia(xarr11, T1,
                                               ntot=N1,
                                               width=W1,
                                               xoff_v=V1)
                spec11b = ammonia.cold_ammonia(xarr11, T2,
                                                 ntot=N2,
                                                 width=W2,
                                                 xoff_v=V2)
                spec11 = spec11a + spec11b

                if (xx == nBorder) and (yy == nBorder):
                    Tmax11a = np.max(spec11a)
                    Tmax11b = np.max(spec11b)
                    Tmax11 = np.max(spec11)
            if nComps[i]==0:
	        cube11[:, yy, xx] = np.zeros(1000)
	    else:
		cube11[:, yy, xx] = spec11
            #cube22[:, yy, xx] = spec22
        cube11 += np.random.randn(*cube11.shape) * noise_rms
        #cube22 += np.random.randn(*cube22.shape) * noise_rms
	
	loc11 = cube11[:,1,1].reshape(1000,1)
	loc11 = loc11/np.max(loc11)
	glob11 = np.mean(cube11.reshape(1000,9),axis=1)
	glob11 = glob11/np.max(glob11)

	loc11 = cube11[:,1,1].reshape(1000,1)
	T1 = Tmax11a/np.max(loc11)
	T2 = Tmax11b/np.max(loc11)
	loc11 = loc11/np.max(loc11)
	glob11 = np.mean(cube11.reshape(1000,9),axis=1)
	glob11 = glob11/np.max(glob11)
	z = np.column_stack((loc11,glob11))
	out_arr.append(z)
	V1 = Voff1[i]
	V2 = Voff2[i]
	W1 = Width1[i]
	W2 = Width2[i]
	
	out_vel = [min([V1,V2]), max([V1,V2]), [W1,W2][np.argmin([V1,V2])], [W1,W2][np.argmax([V1,V2])], [T1,T2][np.argmin([V1,V2])], [T1,T2][np.argmax([V1,V2])]]
        out_vels.append(out_vel)
	#if nComps[i]==1:
		#plt.plot(xax, loc11)
		#plt.plot(xax, glob11+1)
		#plt.errorbar([Voff1[i], Voff2[i]], [0,0.1], xerr=[Width1[i],Width2[i]], color='orange', fmt='none')
		#plt.title(i)
		#plt.show()

        #hdu11 = fits.PrimaryHDU(cube11)
        #for kk in hdrkwds:
        #    hdu11.header[kk] = hdrkwds[kk]
        #    for kk, vv in zip(truekwds, [nComps[i], logN1[i], logN2[i],
        #                                 Voff1[i], Voff2[i], Width1[i], Width2[i],
        #                                 Temp1[i], Temp2[i]]):
        #        hdu11.header[kk] = vv
        #hdu11.header['CRVAL3'] = 23694495500.0
        #hdu11.header['RESTFRQ'] = 23694495500.0
        #hdu11.writeto(output_dir + '/random_cube_NH3_11_'
        #              + '{0}'.format(i).zfill(nDigits)
        #              + '.fits',
        #              overwrite=True)
        #hdu22 = fits.PrimaryHDU(cube22)
        #for kk in hdrkwds:
        #    hdu22.header[kk] = hdrkwds[kk]
        #    for kk, vv in zip(truekwds, [nComps[i], logN1[i], logN2[i],
        #                                 Voff1[i], Voff2[i], Width1[i], Width2[i],
        #                                 Temp1[i], Temp2[i]]):
        #        hdu22.header[kk] = vv
        #hdu22.header['CRVAL3'] = 23722633600.0
        #hdu22.header['RESTFRQ'] = 23722633600.0
        #hdu22.writeto(output_dir + '/random_cube_NH3_22_'
        #              + '{0}'.format(i).zfill(nDigits) + '.fits',
        #              overwrite=True)
    with h5py.File('nh3_three_class_' + set_name + '.h5', 'w') as hf:
	hf.create_dataset('data', data=np.array(out_arr))
	hf.close()
    with h5py.File('labels_nh3_three_class_' + set_name + '.h5', 'w') as hf:
	hf.create_dataset('data', data=np.column_stack((out_y1, out_y2, out_y3)))
	hf.close()
    with h5py.File('params_nh3_three_class_' + set_name + '.h5', 'w') as hf:
	hf.create_dataset('data', data=np.array(out_vels))
	hf.close()
    #print out_y, np.column_stack((out_y1, out_y2, out_y3))

#if __name__ == '__main__':
#    print(sys.argv)
#    if len(sys.argv) > 1:
#        generate_cubes(nCubes=int(sys.argv[1]))
#    else:
#        generate_cubes()

generate_cubes(nCubes=300000, set_name='GAS_reg_train', regression=True)
generate_cubes(nCubes=90000, set_name='GAS_reg_val', regression=True)
generate_cubes(nCubes=30000, set_name='GAS_reg_test', regression=True)
