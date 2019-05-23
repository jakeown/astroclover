import urllib2
import tarfile

print 'Downloading Models (9 GB)'

filedata = urllib2.urlopen('http://www.astro.uvic.ca/~jkeown/astroclover/models.tar.gz')  
datatowrite = filedata.read()

with open('models.tar.gz', 'wb') as f:  
    f.write(datatowrite)

print 'Finished Downloading Models'

###

print 'Expanding Models'

tar = tarfile.open("models.tar.gz")
tar.extractall()
tar.close()

print 'Finished Expanding Models'
