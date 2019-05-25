import urllib2
import tarfile
from tqdm import tqdm

filelist = [
'model_cnn_3class0_gauss_3000_2conv_GAS.tar.gz',
'model_cnn_3class1_gauss_3000_2conv_GAS.tar.gz',
'model_cnn_3class2_gauss_3000_2conv_GAS.tar.gz',
'model_cnn_3class3_gauss_3000_2conv_GAS.tar.gz',
'model_cnn_3class4_gauss_3000_2conv_GAS.tar.gz',
'model_cnn_3class5_gauss_3000_2conv_GAS.tar.gz',
'model_cnn_3class_nh3_sep_short_valloss_GAS_0.tar.gz',
'model_cnn_3class_nh3_sep_short_valloss_GAS_1.tar.gz',
'model_cnn_3class_nh3_sep_short_valloss_GAS_2.tar.gz',
'model_cnn_3class_nh3_sep_short_valloss_GAS_3.tar.gz',
'model_cnn_3class_nh3_sep_short_valloss_GAS_4.tar.gz',
'model_cnn_3class_nh3_sep_short_valloss_GAS.tar.gz',
'model_cnn_reg_nh3_1000_2conv.tar.gz',
'model_cnn_tpeak_gauss_3000_2conv.tar.gz',
]

counter=1
for i in tqdm(filelist):
	print 'Downloading Model ' + str(counter) + ' of ' +str(len(filelist))
	filedata = urllib2.urlopen('http://www.astro.uvic.ca/~jkeown/astroclover/'+i)  
	datatowrite = filedata.read()

	with open(i, 'wb') as f:  
    		f.write(datatowrite)

	###

	tar = tarfile.open(i)
	tar.extractall()
	tar.close()

	counter+=1
