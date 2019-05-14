# astroclover
Convnet Line-fitting Of Velocities in Emission-line Regions (CLOVER)

CLOVER is a convolutional neural network (ConvNet) trained to identify spectra with two velocity components along the line of sight and predict their kinematics.  It works with Gaussian emission lines (e.g., CO) and lines with hyperfine structure (e.g., NH3).  This repository holds both the scripts used to train the ConvNet and those used to test the trained ConvNet's performance or predict on real data cubes. 

CLOVER has two prediction steps:

1) Classification - CLOVER first segments the pixels in an input data cube into one of three classes:
 - Noise (i.e., no emission)
 - One-component (emission line with single velocity component)
 - Two-component (emission line with two velocity components)

2) Parameter Prediction - For the pixels identified as two-components in step 1, a second regression ConvNet is used to predict the following parameters for each velocity component:
 - Centroid Velocity
 - Velocity Dispersion
 - Peak Intensity

Usage:

First, run the download_models.py script.  This will download the trained CNNs from a remote directory.

Next, prepare your data cube by clipping the spectral axis to 500 channels for a non-hyperfine emission line or 1000 channels for a NH3 (1,1) cube.

To run CLOVER on the data cube, simply use the predict(f=your_cube_name.fits) function in the clover.py script. If your cube is NH3 (1,1), add nh3=True in the call to predict() (e.g., predict(f=your_nh3_cube.fits, nh3=False)).

CLOVER will then output the classification map and parmeter predictions as individual FITS files.
