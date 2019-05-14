from test_cnn_3class import test_data as predict_class
from train_cnn_tpeak import test_data as predict_gauss_reg
from train_cnn_reg_nh3 import test_data as predict_nh3_reg

def predict(f='Oph2_13CO_conv_test_smooth_clip.fits', nh3=False):
	if nh3:
		type_name = 'nh3'
	else:
		type_name = 'branch'
	
	# Segment cube into one-comp, two-comp, and noise
	print 'Predicting Class of Each Pixel...'
	predict_class(f=f, type_name=type_name)

	# Predict kinematics of two-comp pixels
	print 'Predicting Kinematics of Two-Component Pixels...'
	if nh3:
		predict_nh3_reg(f=f)
	else:
		predict_gauss_reg(f=f)

#predict()
#predict(f='M17_NH3_11_ml_test.fits', nh3=True)
