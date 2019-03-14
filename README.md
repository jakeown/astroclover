# astroclover
Convnet Line-fitting Of Velocities in Emission-line Regions (CLOVER)

CLOVER is a convolutional neural network (CNN) method for identifying emission line spectra with two velocity components along the line of sight and predicting their kinematics.  It works with Gaussian emission lines (e.g., CO) and lines with hyperfine structure (e.g., NH3).  This repository holds both the scripts used to train the neural network and those used to test the trained network's performance or predict on real data cubes. 

CLOVER has two predictions steps:

1) Classification 
CLOVER first segments the pixels in an input data cube into one of three classes:
 - Noise (i.e., no emission)
 - One-component (emission line with single velocity component)
 - Two-component (emission line with two velocity components)

2) Velocity Prediction
For the pixels identified as two-components in step 1, a second regression CNN is used to predict the following parameters for each velocity component:
 - Centroid Velocity
 - Velocity Dispersion
 - Peak Intensity
