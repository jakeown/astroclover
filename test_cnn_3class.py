import skimage
from astropy.io import fits
from keras.models import load_model
import numpy
import matplotlib.pyplot as plt 
from keras.models import Model, Input
from keras.layers import Average
from keras import layers
from sklearn.metrics import confusion_matrix, roc_curve, auc
import h5py
import itertools
import numpy as np
import time
from sklearn.metrics import accuracy_score

# Order of class names in training set
class_names = ['one', 'noise', 'two']

def get_train_multi(type_name='train'):
	# Quicker data loading method
	# Data is stored and loaded in two h5 files
	# One with training data, other with training labels
	if type_name=='test2':
		t = '.h5'
	else:
		t = '1.h5'

	print 'Loading Training Data...'
	with h5py.File('three_class_gauss_'+type_name+t, 'r') as hf:
		X = hf['data'][:]
	hf.close()
	with h5py.File('labels_three_class_gauss_'+type_name+t, 'r') as hf:
		y = hf['data'][:]
	hf.close()
	return X, y

def get_meta(type_name='test'):
	# Quicker data loading method
	# Data is stored and loaded in two h5 files
	# One with training data, other with training labels
	if type_name=='test2':
		t = '.h5'
	else:
		t = '1.h5'

	print 'Loading Training Data...'
	with h5py.File('params_three_class_gauss_'+type_name+t, 'r') as hf:
		X = hf['data'][:]
	hf.close()
	return X

def get_train_multi_nh3(type_name='test'):
	# Quicker data loading method
	# Data is stored and loaded in two h5 files
	# One with training data, other with training labels

	print 'Loading Training Data...'
	with h5py.File('nh3_three_class_GAS_'+type_name+'.h5', 'r') as hf:
		X = hf['data'][:]
	hf.close()
	with h5py.File('labels_nh3_three_class_GAS_'+type_name+'.h5', 'r') as hf:
		y = hf['data'][:]
	hf.close()
	return X, y

def get_nh3_meta(type_name='test'):
	# Quicker data loading method
	# Data is stored and loaded in two h5 files
	# One with training data, other with training labels

	print 'Loading Training Data...'
	with h5py.File('params_nh3_three_class_GAS_'+type_name+'.h5', 'r') as hf:
		X = hf['data'][:]
	hf.close()
	return X

def grab_model(type_name, X_val_new):
	if type_name=='branch':
		# branch model
		#model = load_model("model_cnn_3class.h5")
		model = load_model("model_cnn_3class0_gauss_3000_2conv_GAS.h5")
		X_val_new = [X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)]
	elif type_name=='global':
		# global-only model
		model = load_model("model_cnn_3class_gauss_3000_2conv_global.h5") 
		X_val_new = X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)
	elif type_name=='local':
		model = load_model("model_cnn_3class_gauss_3000_2conv_local.h5") 
		X_val_new = X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)
	elif type_name=='mergedspec':
		model = load_model("model_cnn_3class_mergedspec.h5")
		X_val_new = numpy.concatenate([X_val_new[:,:,0], X_val_new[:,:,1]], axis=1).reshape(X_val_new.shape[0], X_val_new.shape[1]*2, 1)
	elif type_name=='nh3':
		# branch model
		model = load_model("model_cnn_3class_nh3_sep_short_valacc_GAS.h5")
		X_val_new = [X_val_new[:,:,0].reshape(X_val_new.shape[0], X_val_new.shape[1], 1), X_val_new[:,:,1].reshape(X_val_new.shape[0], X_val_new.shape[1], 1)]
	return model, X_val_new

def test_CNN():
	X_val_new, y_val_new = get_train_multi(type_name='test')
	# Voff1, Voff2, Width1, Width2, Temp1, Temp2, logN1, logN2
	meta = get_meta(type_name='test')
	#print meta[:,2]
	#print meta[:,3]
	fig = plt.figure()
	def plt_preds(predictions, type_name='branch'):

		meta2 = meta[np.argmax(y_val_new, axis=1)==2]
		predictions2 = predictions[np.argmax(y_val_new, axis=1)==2]
		X_val_new2 = X_val_new[np.argmax(y_val_new, axis=1)==2]
		y_val_new2 = np.argmax(y_val_new[np.argmax(y_val_new, axis=1)==2], axis=1)
	
		bins = numpy.arange(0.2, 3.4, 0.2)
		bin_ids = numpy.digitize(numpy.absolute(meta2[:,0]-meta2[:,1]), bins)

		lengths = []
		scores = []
		for i in numpy.arange(max(bin_ids))+1:
			scores.append(accuracy_score(y_val_new2[bin_ids==i],predictions2[bin_ids==i]))
			lengths.append(len(y_val_new2[bin_ids==i]))
		center = (bins[:-1] + bins[1:]) / 2
		center = numpy.concatenate((center, [max(bins+0.2)]))
		if type_name!='CNN Local+Global':
			plt.plot(center, scores, label=type_name, marker='o')
		else:
			sc = plt.scatter(center, scores, c=lengths, cmap=plt.cm.get_cmap('RdYlBu'), label=type_name, edgecolors='black', zorder=20)
			cbar = plt.colorbar(sc)
			cbar.set_label('Number of Samples', rotation=270, labelpad=10)

	model, X_v = grab_model(type_name='branch', X_val_new=X_val_new)
	predictions = np.argmax(model.predict(X_v), axis=1)
	plt_preds(predictions=predictions, type_name='CNN Local+Global')
	x2l = numpy.load('chi_predictions_3class_local_test_bic_snr4.npy')
	plt_preds(predictions=x2l,type_name='Chi Local')
	x2g = numpy.load('chi_predictions_3class_global_test_bic_snr4.npy')
	plt_preds(predictions=x2g,type_name='Chi Global')
	plt.xlabel('$\Delta V_{LSR}$ (km s$^{-1}$)', size=18)
	plt.ylabel(r'Classification Accuracy', size=18)
	plt.title('Two-Comp Predictions')
	plt.legend()
	fig.savefig('CNN_acc_vlsr.pdf', bbox_inches='tight')
	plt.show()

def test_CNN_snr():
	X_val_new, y_val_new = get_train_multi(type_name='test')
	specs = X_val_new[:,:,0]
	rms = numpy.std(numpy.column_stack((specs[:,0:50], specs[:,-50:])), axis=1)
	peak=numpy.max(specs, axis=1)
	snrs = peak/rms
	# Voff1, Voff2, Width1, Width2, Temp1, Temp2, logN1, logN2
	meta = get_meta(type_name='test')
	#print meta[:,2]
	#print meta[:,3]
	fig = plt.figure()
	def plt_preds(predictions, type_name='branch'):

		snrs2 = snrs[np.argmax(y_val_new, axis=1)==2]
		predictions2 = predictions[np.argmax(y_val_new, axis=1)==2]
		X_val_new2 = X_val_new[np.argmax(y_val_new, axis=1)==2]
		y_val_new2 = np.argmax(y_val_new[np.argmax(y_val_new, axis=1)==2], axis=1)
	
		bins = numpy.arange(2, 20, 2)
		bin_ids = numpy.digitize(snrs2, bins)

		lengths = []
		scores = []
		for i in numpy.arange(max(bin_ids))+1:
			scores.append(accuracy_score(y_val_new2[bin_ids==i],predictions2[bin_ids==i]))
			lengths.append(len(y_val_new2[bin_ids==i]))
		print lengths
		center = (bins[:-1] + bins[1:]) / 2
		center = numpy.concatenate((center, [max(bins+2)]))
		if type_name!='CNN Local+Global':
			plt.plot(center, scores, label=type_name, marker='o')
		else:
			sc = plt.scatter(center, scores, c=lengths, cmap=plt.cm.get_cmap('RdYlBu'), label=type_name, edgecolors='black', zorder=20)
			cbar = plt.colorbar(sc)
			cbar.set_label('Number of Samples', rotation=270, labelpad=10)

	model, X_v = grab_model(type_name='branch', X_val_new=X_val_new)
	predictions = np.argmax(model.predict(X_v), axis=1)
	plt_preds(predictions=predictions, type_name='CNN Local+Global')
	x2l = numpy.load('chi_predictions_3class_local_test_bic_snr4.npy')
	plt_preds(predictions=x2l,type_name='Chi Local')
	x2g = numpy.load('chi_predictions_3class_global_test_bic_snr4.npy')
	plt_preds(predictions=x2g,type_name='Chi Global')
	plt.xlabel('Local Spectrum SNR', size=18)
	plt.ylabel(r'Classification Accuracy', size=18)
	plt.title('Two-Comp Predictions')
	plt.legend()
	fig.savefig('CNN_acc_snr.pdf', bbox_inches='tight')
	plt.show()

def test_GAS():
	X_val_new, y_val_new = get_train_multi_nh3(type_name='test')
	# Voff1, Voff2, Width1, Width2, Temp1, Temp2, logN1, logN2
	meta = get_nh3_meta(type_name='test')
	#print meta[:,2]
	#print meta[:,3]
	model, X_v = grab_model(type_name='nh3', X_val_new=X_val_new)
	predictions = np.argmax(model.predict(X_v), axis=1)

	meta2 = meta[np.argmax(y_val_new, axis=1)==2]
	predictions2 = predictions[np.argmax(y_val_new, axis=1)==2]
	X_val_new2 = X_val_new[np.argmax(y_val_new, axis=1)==2]
	y_val_new2 = np.argmax(y_val_new[np.argmax(y_val_new, axis=1)==2], axis=1)
	
	bins = numpy.arange(0.2, 2.8, 0.2)
	bin_ids = numpy.digitize(numpy.absolute(meta2[:,0]-meta2[:,1]), bins)

	lengths = []
	scores = []
	for i in numpy.arange(max(bin_ids))+1:
		scores.append(accuracy_score(y_val_new2[bin_ids==i],predictions2[bin_ids==i]))
		lengths.append(len(y_val_new2[bin_ids==i]))
	print lengths
	print scores
	center = (bins[:-1] + bins[1:]) / 2
	center = numpy.concatenate((center, [max(bins+0.2)]))
	fig = plt.figure()
	sc = plt.scatter(center, scores, c=lengths, cmap=plt.cm.get_cmap('RdYlBu'), label='Two-Comp')
	cbar = plt.colorbar(sc)
	cbar.set_label('Number of Samples', rotation=270, labelpad=10)
	plt.xlabel('$\Delta V_{LSR}$ Bin Center (km s$^{-1}$)', size=18)
	plt.ylabel(r'Classification Accuracy', size=18)
	plt.title('Two-Comp CNN Predictions')
	fig.savefig('GAS_CNN_acc_vlsr.pdf', bbox_inches='tight')
	plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, stds=None, use_stds=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	cm[numpy.where(numpy.isnan(cm))]=0
        print("")
    else:
        print('')

    print(cm)

    if cm.shape[0]==2:
	cl = [classes[0], classes[2]]
    else:
	cl = classes
    plt.rcParams.update({'font.size': 18})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cl))
    plt.xticks(tick_marks, cl, rotation=45)
    plt.yticks(tick_marks, cl)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	if use_stds:
        	plt.text(j, i, str(int(round(cm[i, j])))+ '$\pm$' + str(int(round(stds[i, j]))),
                 horizontalalignment="center", fontsize=14,
                 color="white" if cm[i, j] > thresh else "black")
	else:
		plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def ensembleModels(testX):
	models = []
	for i in range(8):
		models.append(load_model("model_cnn_3class"+str(i)+".h5"))
	# make predictions
	yhats = [model.predict(testX) for model in models]
	yhats = numpy.array(yhats)
	# sum across ensembles
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	#outcomes = numpy.argmax(summed, axis=1)
    	return summed

def get_low_snr(X_val, local=True, snr_cut=4, less_than=True):
	if local:
		specs = X_val[:,:,0]
	else:
		specs = X_val[:,:,1]
	rms = numpy.std(numpy.column_stack((specs[:,0:50], specs[:,-50:])), axis=1)
	peak=numpy.max(specs, axis=1)
	snrs_single = peak/rms

	if less_than:
		out = numpy.where((snrs_single<snr_cut))
	else:
		out = numpy.where((snrs_single>snr_cut))
	return out

def save_cm(y_val_new, predictions, save_name='nh3', class_names=class_names, ttl='NH3'):
	cnf_matrix = confusion_matrix(np.argmax(y_val_new,axis=1), predictions)
	np.set_printoptions(precision=3)

	# Plot non-
	fig = plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='')
	plt.title(ttl)
	fig.savefig(save_name+'_cm_3class.pdf', bbox_inches='tight')

	# Plot 
	fig = plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='')
	plt.title(ttl)
	fig.savefig(save_name+'_norm_cm_3class.pdf', bbox_inches='tight')

	#if 'chi' in save_name:
	#	scores_chi = predictions-np.argmax(y_val_new, axis=1)
	#	scores = len(numpy.where(scores_chi==0)[0])/float(len(scores_chi))
	#	print("Accuracy: %.2f%%" % (scores*100))
	#else:
	#	scores = accuracy_score(np.argmax(y_val_new,axis=1), predictions)
	#	print("Accuracy: %.2f%%" % (scores*100))
	scores = accuracy_score(np.argmax(y_val_new,axis=1), predictions)
	print("Accuracy: %.2f%%" % (scores*100))

def get_cm_err():
	# Load chi-local predictions
	loc_preds = numpy.load('chi_predictions_3class_local_test_bic_snr4_full.npy')
	loc_s = loc_preds[0:100000]
	loc_n = loc_preds[100000:200000]
	loc_m = loc_preds[200000:]
	# Load chi-global predictions
	glob_preds = numpy.load('chi_predictions_3class_global_test_bic_snr4_full.npy')
	glob_s = glob_preds[0:100000]
	glob_n = glob_preds[100000:200000]
	glob_m = glob_preds[200000:]
	# Load dat for CNN predictions
	X_val_new, y_val_new = get_train_multi(type_name='test2')

	model, X_v = grab_model(type_name='branch', X_val_new=X_val_new)
	model_local, X_v = grab_model(type_name='local', X_val_new=X_val_new)
	model_global, X_v = grab_model(type_name='global', X_val_new=X_val_new)
	print numpy.shape(X_v)
	print X_val_new.shape

	# Now predict for low snrs (local)
	low_snrs = get_low_snr(X_val_new)
	print len(low_snrs[0])
	model2, X_v2 = grab_model(type_name='branch', X_val_new=X_val_new[low_snrs])
	predictions = np.argmax(model.predict(X_v2), axis=1)
	save_cm(y_val_new[low_snrs], predictions, save_name='low_snr', class_names=class_names)

	save_cm(y_val_new[low_snrs], loc_preds[low_snrs], save_name='chi_local_low_snr', class_names=class_names)

	save_cm(y_val_new[low_snrs], glob_preds[low_snrs], save_name='chi_global_low_snr', class_names=class_names)

	# Now predict for low snrs (global)
	low_snrs = get_low_snr(X_val_new, local=True, snr_cut=5.0, less_than=False)
	print len(low_snrs[0])
	model2, X_v2 = grab_model(type_name='branch', X_val_new=X_val_new[low_snrs])
	predictions = np.argmax(model.predict(X_v2), axis=1)
	save_cm(y_val_new[low_snrs], predictions, save_name='low_snr2', class_names=class_names)

	save_cm(y_val_new[low_snrs], loc_preds[low_snrs], save_name='chi_local_low_snr2', class_names=class_names)

	save_cm(y_val_new[low_snrs], glob_preds[low_snrs], save_name='chi_global_low_snr2', class_names=class_names)

	single = X_val_new[0:100000]
	noise = X_val_new[100000:200000]
	multi = X_val_new[200000:]
	ys = y_val_new[0:100000]
	yn = y_val_new[100000:200000]
	ym = y_val_new[200000:]
	print numpy.shape(multi)
	inds = numpy.arange(0,110000,10000)
	mats = []
	mats_loc = []
	mats_glob = []
	cnn_loc = []
	cnn_glob = []
	for i in range(10):
		X_val = numpy.vstack((single[inds[i]:inds[i+1]], noise[inds[i]:inds[i+1]], multi[inds[i]:inds[i+1]]))
		X_val = [X_val[:,:,0].reshape(X_val.shape[0], X_val.shape[1], 1), X_val[:,:,1].reshape(X_val.shape[0], X_val.shape[1], 1)]
		y_val = numpy.vstack((ys[inds[i]:inds[i+1]], yn[inds[i]:inds[i+1]], ym[inds[i]:inds[i+1]]))
		predictions = np.argmax(model.predict(X_val), axis=1)
		cnf_matrix = confusion_matrix(np.argmax(y_val,axis=1), predictions)
		print cnf_matrix
		mats.append(cnf_matrix)

		print np.shape(X_val)
		predictions = np.argmax(model_local.predict(X_val[0]), axis=1)
		cnf_matrix = confusion_matrix(np.argmax(y_val,axis=1), predictions)
		print cnf_matrix
		cnn_loc.append(cnf_matrix)

		predictions = np.argmax(model_global.predict(X_val[1]), axis=1)
		cnf_matrix = confusion_matrix(np.argmax(y_val,axis=1), predictions)
		print cnf_matrix
		cnn_glob.append(cnf_matrix)

		cm_loc = confusion_matrix(np.argmax(y_val,axis=1), numpy.concatenate((loc_s[inds[i]:inds[i+1]], loc_n[inds[i]:inds[i+1]], loc_m[inds[i]:inds[i+1]])))
		print cm_loc
		mats_loc.append(cm_loc)
		cm_glob = confusion_matrix(np.argmax(y_val,axis=1), numpy.concatenate((glob_s[inds[i]:inds[i+1]], glob_n[inds[i]:inds[i+1]], glob_m[inds[i]:inds[i+1]])))
		print cm_glob
		mats_glob.append(cm_glob)

	meds = numpy.mean(mats, axis=0)
	stds = numpy.std(mats, axis=0)
	fig = plt.figure()
	plot_confusion_matrix(meds, classes=class_names,
                      title='CNN Local+Global', stds=stds, use_stds=True)
	fig.savefig('cm_3class_test10.pdf', bbox_inches='tight')

	meds = numpy.mean(cnn_loc, axis=0)
	stds = numpy.std(cnn_loc, axis=0)
	fig = plt.figure()
	plot_confusion_matrix(meds, classes=class_names,
                      title='CNN Local', stds=stds, use_stds=True)
	fig.savefig('cnn_local_cm_3class_test10.pdf', bbox_inches='tight')

	meds = numpy.mean(cnn_glob, axis=0)
	stds = numpy.std(cnn_glob, axis=0)
	fig = plt.figure()
	plot_confusion_matrix(meds, classes=class_names,
                      title='CNN Global', stds=stds, use_stds=True)
	fig.savefig('cnn_global_cm_3class_test10.pdf', bbox_inches='tight')

	meds = numpy.mean(mats_loc, axis=0)
	stds = numpy.std(mats_loc, axis=0)
	fig = plt.figure()
	plot_confusion_matrix(meds, classes=class_names,
                      title='$\chi^{2}$ Local', stds=stds, use_stds=True)
	fig.savefig('chi_local_cm_3class_test10.pdf', bbox_inches='tight')

	meds = numpy.mean(mats_glob, axis=0)
	stds = numpy.std(mats_glob, axis=0)
	fig = plt.figure()
	plot_confusion_matrix(meds, classes=class_names,
                      title='$\chi^{2}$ Global', stds=stds, use_stds=True)
	fig.savefig('chi_global_cm_3class_test10.pdf', bbox_inches='tight')

def ftest_thresh(x2l_nh3, threshold=3):
	x2l_nh3[(x2l_nh3<threshold) & (x2l_nh3!=1)] = 0
	x2l_nh3[(x2l_nh3>threshold) & (x2l_nh3!=1)] = 2
	return x2l_nh3


def get_confusion_matrix(type_name='branch'):
	#x2 = numpy.load('chi_predictions_local.npy')
	#x2[x2==2] = 1
	x2g = numpy.load('chi_predictions_3class_global_test_bic_snr4.npy')
	x2l = numpy.load('chi_predictions_3class_local_test_bic_snr4.npy')
	x2l_nh3 = numpy.load('chi_predictions_nh3_local_snr4.npy')
	x2g_nh3 = numpy.load('chi_predictions_nh3_global_snr4.npy')
	#x2l_nh3 = ftest_thresh(x2l_nh3, threshold=3)
	#x2g_nh3 = ftest_thresh(x2g_nh3, threshold=3)
	#x2g[x2g==2] = 1
	#class_names = ['signal', 'noise']
	class_names = ['one', 'noise', 'two']
	X_test, y_val_new2 = get_train_multi(type_name='test')
	X_val_new, y_val_new = get_train_multi_nh3(type_name='test_small')

	low_snrs = get_low_snr(X_val_new, snr_cut=7)
	print len(low_snrs[0])
	model3, X_v2 = grab_model(type_name=type_name, X_val_new=X_val_new[low_snrs])
	predictions = np.argmax(model3.predict(X_v2), axis=1)
	cnf_matrix = confusion_matrix(np.argmax(y_val_new[low_snrs],axis=1), predictions)
	print cnf_matrix
	fig = plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='')
	fig.savefig('low_snr_nh3_norm_cm_3class.pdf', bbox_inches='tight')

	low_snrs = get_low_snr(X_val_new, snr_cut=7, less_than=False)
	print len(low_snrs[0])
	model3, X_v2 = grab_model(type_name=type_name, X_val_new=X_val_new[low_snrs])
	predictions = np.argmax(model3.predict(X_v2), axis=1)
	cnf_matrix = confusion_matrix(np.argmax(y_val_new[low_snrs],axis=1), predictions)
	print cnf_matrix
	fig = plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='')
	fig.savefig('high_snr_nh3_norm_cm_3class.pdf', bbox_inches='tight')

	model, X_val_new = grab_model(type_name=type_name, X_val_new=X_val_new)
	model2, X_val_new2 = grab_model(type_name='branch', X_val_new=X_test)
	model3, X_val_new3 = grab_model(type_name='local', X_val_new=X_test)
	model4, X_val_new4 = grab_model(type_name='global', X_val_new=X_test)
	#X_val_new = X_val_new.transpose(0,2,1)
	predictions = np.argmax(model.predict(X_val_new), axis=1)
	scores = accuracy_score(np.argmax(y_val_new,axis=1), predictions)
	predictions2 = np.argmax(model2.predict(X_val_new2), axis=1)
	scores2 = accuracy_score(np.argmax(y_val_new2,axis=1), predictions2)
	predictions3 = np.argmax(model3.predict(X_val_new3), axis=1)
	scores3 = accuracy_score(np.argmax(y_val_new2,axis=1), predictions3)
	predictions4 = np.argmax(model4.predict(X_val_new4), axis=1)
	scores4 = accuracy_score(np.argmax(y_val_new2,axis=1), predictions4)

	# Compute confusion matrix
	print 'CNN NH3:'
	save_cm(y_val_new, predictions, save_name='nh3', class_names=class_names, ttl='CNN NH$_3$')

	print 'Chi-global NH3:'
	save_cm(y_val_new, x2g_nh3, save_name='chi_nh3_global', class_names=class_names, ttl='Chi-global NH3')

	print 'Chi-local NH3:'
	save_cm(y_val_new, x2l_nh3, save_name='chi_nh3_local', class_names=class_names, ttl='Chi-local NH3')

	print 'CNN Gauss:'
	save_cm(y_val_new2, predictions2, save_name='branch', class_names=class_names, ttl='CNN Local+Global')

	print 'CNN Gauss Local:'
	save_cm(y_val_new2, predictions3, save_name='local', class_names=class_names, ttl='CNN Local')

	print 'CNN Gauss Global:'
	save_cm(y_val_new2, predictions4, save_name='global', class_names=class_names, ttl='CNN Global')

	print 'Chi-global Gauss:'
	save_cm(y_val_new2, x2g, save_name='chi_gauss_global', class_names=class_names, ttl='Chi Global')

	print 'Chi-local Gauss:'
	save_cm(y_val_new2, x2l, save_name='chi_gauss_local', class_names=class_names, ttl='Chi Local')

def test_data(f='CygX_N_13CO_conv_test_smooth_clip.fits', type_name='branch'):
	# Predicts the class of each pixel in real FITS cube
	# If Gaussian cube, use type_name='branch'
	# If NH3 cube, use type_name='nh3'
	tic = time.time()
	data = fits.getdata(f)
	header = fits.getheader(f)
	print data.shape
	# Create a 2D array to place ouput predictions
	out_arr = data[0].copy()
	out_arr[:]=numpy.nan
	out_arr2 = out_arr.copy()
	
	window_shape = [data.shape[0],3,3]
	X_val_new = []
	X_val_full = []
	indices = []
	if type_name=='nh3':
		length=1000
	else:
		length=500
	for index, x in numpy.ndenumerate(data[0]):
		z = data[:, index[0]-1:index[0]+2, index[1]-1:index[1]+2]
		if z.shape==(length, 3,3):
			indices.append(index)
			local0 = z[:,1,1].reshape(length,1) # central pixel
			local0 = local0/numpy.max(local0)
			local1 = numpy.mean(z[:,:,:].reshape(length,9), axis=1) #3x3 pixel average
			local1 = local1/numpy.max(local1)
			#if max(local1)/numpy.std(local0[0:50])>6.0:
			#	plt.plot(range(len(local1)), local1)
			#	plt.plot(range(len(glob1)), glob1, alpha=0.5)
			#	plt.show()
			z = numpy.column_stack((local0,local1))
			X_val_new.append(z)
	X_val_new = numpy.array(X_val_new)
	indices = numpy.array(indices)
	#windows = skimage.util.view_as_windows(data, window_shape, 1)
	print X_val_new.shape

	#count = 0
	#for i in X_val_new:
	#	X_val_new[count] = i*(1/numpy.max(i))

	# load model
	new_model, X_val_new = grab_model(type_name, X_val_new)
	
	print "Loaded model from disk"
	
	# Make prediction on each pixel and output as 2D fits image
	predictions = new_model.predict(X_val_new, verbose=0)
	#print predictions.shape
	#predictions = ensembleModels(X_val_new)
	#print predictions.shape
	# Reshape to get back 2D structure
	for i,j in zip(predictions,indices):
		ind = numpy.argmax(i)
		out_arr[j[0], j[1]] = ind
		out_arr2[j[0],j[1]] = -np.log10(i[ind])
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
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_3class_'+type_name+'.fits', data=out_arr, header=header, overwrite=True)
	fits.writeto(f.split('.fits')[0]+'_pred_cnn_3class_err'+type_name+'.fits', data=out_arr2, header=header, overwrite=True)
	print "\n %f s for computation." % (time.time() - tic)

#get_confusion_matrix(type_name='branch')
#get_confusion_matrix(type_name='local')
#get_confusion_matrix(type_name='global')
#get_confusion_matrix(type_name='mergedspec')
#get_confusion_matrix(type_name='nh3')

#get_cm_err()

#test_CNN_snr()
#test_CNN()
#test_GAS()

#test_data(f='B18_HC5N_conv_test_smooth_clip.fits', type_name='branch')
#test_data(f='CygX_N_C18O_conv_test_smooth_clip2.fits', type_name='branch')
#test_data(f='Oph2_13CO_conv_test_smooth_clip.fits', type_name='branch')
#test_data(f='Oph2_13CO_conv_test_smooth_clip.fits', type_name='global')
#test_data(f='CygX_N_C18O_conv_test_smooth_clip2.fits', type_name='global')
#test_data(f='B18_HC5N_conv_test_smooth_clip.fits', type_name='global')
#test_data(f='M17_NH3_11_ml_test.fits',type_name='nh3')
#test_data(f='MonR2_NH3_11_ml_test.fits',type_name='nh3')
#for i in ['branch', 'local', 'global', 'mergedspec']:
#	test_data(f='B18_HC5N_conv_test_smooth_clip.fits', type_name=i)
#	test_data(f='CygX_N_C18O_conv_test_smooth_clip2.fits', type_name=i)
#	test_data(f='Oph2_13CO_conv_test_smooth_clip.fits', type_name=i)
#test_data(f='W3Main_C18O_conv_test_smooth_clip.fits', c=0)
#test_data(f='Oph_13CO_conv_test_smooth_clip.fits', c=0)
#test_data(f='NGC7538_C18O_conv_test_smooth_clip.fits', c=0)
#test_data(f='CygX_N_13CO_conv_test_smooth_clip.fits', c=1)
