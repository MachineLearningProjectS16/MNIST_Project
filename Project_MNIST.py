import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np

import theano
#import lasagne
#from lasagne import layers
#from lasagne.updates import nesterov_momentum
#from nolearn.lasagne import NeuralNet
#from nolearn.lasagne import visualize
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix

#url = 'http://yann.lecun.com/exdb/mnist/'
url='C:\\Users\\sravika\\Downloads\\Course#589\\Final_Project\\';

#filename = 'train-images-idx3-ubyte.gz'
def download_get_images(filename):
        if not os.path.exists(filename):
           urlretrieve(url+filename, filename) 
        # Read the inputs in Yann LeCun's binary format.
        filename=url+filename;
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
          #  print("DAta shape:",data.shape)
            data = data.reshape(-1, 1, 28, 28)
            print("DAta shape:",data.shape)
            return data / np.float32(256)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
         # following the shape convention: (examples, channels, rows, columns)
         # The inputs come as bytes, we convert them to float32 in range [0,1].
         # (Actually to range [0, 255/256], for compatibility to the version
         # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
           

def download_get_labels(filename):   
       filename=url+filename;
       #print("In reading labels")
       with gzip.open(filename, 'rb') as f:
            data= np.frombuffer(f.read(), np.uint8, offset=8)
            #print("Data shape labels:",data.shape)
            data.reshape(1,data.shape[0])   
       return data 
    
X_train=download_get_images('train-images-idx3-ubyte.gz');
Y_train=download_get_labels('train-labels-idx1-ubyte.gz');
X_test=download_get_images('t10k-images-idx3-ubyte.gz');
Y_test=download_get_labels('t10k-labels-idx1-ubyte.gz');

#Last 10k training samples put for validation 
X_train, X_val = X_train[:-10000], X_train[-10000:]
Y_train, Y_val = Y_train[:-10000], Y_train[-10000:]
#X_train[0][0]--of the shape 28x28 and values in the range of 0.0 and 1.0


