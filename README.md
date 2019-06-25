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

First, run the `download_models.py` script.  This will download the trained ConvNets from a remote directory.  This will take some time since the 14 files are ~ 9 GB in total.

The easiest way to install all the required packages is to install [anaconda](https://www.anaconda.com/distribution/) and create a new virtual environment. Anaconda version 4.6.11 or later is recommended.  Once anaconda is installed, you can run the following commands to setup a new environment and install the remaining packages required for CLOVER:

```
conda create -n clover_env python=2.7 anaconda
conda activate clover_env
pip install tensorflow==1.8.0 keras==2.2.0 pprocess spectral_cube
```

To run CLOVER on your data cube, simply use the `predict(f=your_cube_name.fits)` function in the `clover.py` script. If your cube is NH3 (1,1), add `nh3=True` in the call to `predict()` (e.g., `predict(f=your_nh3_cube.fits, nh3=True)`).

CLOVER's predictions require 500 spectral channels for Gaussian emission lines and 1000 channels for NH3 (1,1).  If the cube you input into the predict function is smaller those sizes, CLOVER will add random noise channels to each end of the spectral axis up to the required size.  If the input cube is smaller than the required input size, CLOVER will clip channels from each end of the spectral axis until the required size is obtained.  

It is recommended that the centroid of the emission lines in your cube be located within the central ~275 channels for Gaussian emission lines and the central ~140 channels for NH3 (1,1).  These bounds are set by the range of possible centroids used to train CLOVER.  If your cube has large centroid velocity gradients, then you may need to split the cube into sub-cubes so that the emission is within the aforementioned bounds.

CLOVER will output its classification map and parameter predictions as individual FITS files.  In total, up to eight files are generated:
1. input_name + '_clover.fits' - cube after the spectral axis has been corrected (not generated if input cube already has proper spectral length)
2. input_name + '_class.fits' - predicted class of each pixel (2=two-component, 1=noise, 0=one-component)
3. input_name + '_vlsr1.fits' - predicted centroid velocity of component with lowest centroid
4. input_name + '_vlsr2.fits' - predicted centroid velocity of component with highest centroid
5. input_name + '_sig1.fits' - predicted velocity dispersion of component with lowest centroid
6. input_name + '_sig2.fits' - predicted velocity dispersion of component with highest centroid
7. input_name + '_tpeak1.fits' - predicted peak intensity of component with lowest centroid
8. input_name + '_tpeak2.fits' - predicted peak intensity of component with highest centroid

The classification step uses an ensemble of six independently trained ConvNets to make the final class prediction.  These six predictions can be done in parallel by specifying the number of desired parallel processes.  For example, to run all six predictions at once, use `predict(f=your_cube_name.fits, nproc=6)`.  
