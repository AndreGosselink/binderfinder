__hgrev__ = 85
__version__ = "1.3 rev {}".format(__hgrev__+1)

print "starting binderfinder " + bf.__version__  + ' ' + branch
print 'started with pid', os.getpid()


from matrix import Matrix
from pca import PCA
