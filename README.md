# astroclover
Convnet Line-fitting Of Velocities in Emission-line Regions (CLOVER)

CLOVER is a convolutional neural network (ConvNet) trained to identify spectra with two velocity components along the line of sight and predict their kinematics.  It works with Gaussian emission lines (e.g., CO) and lines with hyperfine structure (e.g., NH3).  This repository holds both the scripts used to train the ConvNet and those used to test the trained ConvNet's performance or predict on real data cubes. 

CLOVER has two prediction steps:

1) Classification - CLOVER first segments the pixels in an input data cube into one of three classes:
 - Noise (i.e., no emission)
 - One-component (emission line with single velocity component)
 - Two-component (emission line with two velocity components)

2) Velocity Prediction - For the pixels identified as two-components in step 1, a second regression ConvNet is used to predict the following parameters for each velocity component:
 - Centroid Velocity
 - Velocity Dispersion
 - Peak Intensity
