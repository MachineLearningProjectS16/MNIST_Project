#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.cuda_convnet

import gzip
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import PatchExtractor
from scipy.stats.mstats import mode
from ImageAugmenter import ImageAugmenter
from scipy import misc
from lasagne.regularization import regularize_layer_params
from lasagne.regularization import regularize_layer_params_weighted, l2, l1

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
#    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    with gzip.open("./train-images-idx3-ubyte.gz", 'rb') as f:
         data = np.frombuffer(f.read(), np.uint8, offset=16)
         data = data.reshape(-1, 1, 28, 28)
         X_train= data / np.float32(256)
         x_train=X_train
    print ("X_train shape:",X_train.shape)
   # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def translate_images():
    with gzip.open("./train-images-idx3-ubyte.gz", 'rb') as f:
         data = np.frombuffer(f.read(), np.uint8, offset=16)
         data = data.reshape(-1, 1, 28, 28)
         unscaled_images=data[:10000]

    print(unscaled_images.dtype,unscaled_images[0].shape)
    image_augmenter=ImageAugmenter(28,28,translation_x_px=(-5,5))

    augmented_images=image_augmenter.augment_batch(unscaled_images[0])
    #Translate the first half of the images along x-axis
    for image in range(1,unscaled_images.shape[0]/2):
        augmented_images=np.vstack((augmented_images,image_augmenter.augment_batch(unscaled_images[image])))
  
   #Translate the second half of the images along y-axis
    image_augmenter=ImageAugmenter(28,28,translation_y_px=(-5,10))
    for image in range(unscaled_images.shape[0]/2,unscaled_images.shape[0]):
        augmented_images=np.vstack((augmented_images,image_augmenter.augment_batch(unscaled_images[image])))
  
    print("Translated:",augmented_images.shape)        
    return (augmented_images/np.float32(256)).reshape(-1,1,28,28) 

def rotate_images(degrees=45):
    with gzip.open("./train-images-idx3-ubyte.gz", 'rb') as f:
         data = np.frombuffer(f.read(), np.uint8, offset=16)
         data = data.reshape(-1, 1, 28, 28)
         unscaled_images=data[:10000]

    print(unscaled_images.dtype,unscaled_images[0].shape)
    #Rotate all the images between -45 to  45
    image_augmenter=ImageAugmenter(28,28,rotation_deg=(60))
    augmented_images=image_augmenter.augment_batch(unscaled_images[0])
    for image in range(1,unscaled_images.shape[0]):
        augmented_images=np.vstack((augmented_images,image_augmenter.augment_batch(unscaled_images[image])))
    print("Augmented",augmented_images.shape)        
    return (augmented_images/np.float32(256)).reshape(-1,1,28,28) 

def expand_images():
     print("Loading data...")
     X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
     #This deletes teh last column of each image
     X_train_right_shift=np.delete(X_train,27,axis=3)
     #Add element 0 to the first column of each image.
     X_train_right_shift=np.insert(X_train_right_shift,0,0,axis=3)
     print("X_train_right shape:",X_train_right_shift.shape)
     #Create left shift,up shift and down shift -->all of them hstack to the
     #original array X_train.
     #This deletes teh first column of each image
     X_train_left_shift=np.delete(X_train,0,axis=3)
     #Add element 0 to the last column of each image.
     X_train_leftt_shift=np.insert(X_train_left_shift,27,0,axis=3)
     print("X_train_left shape:",X_train_leftt_shift.shape)
   
     #This deletes teh first row  of each image
     X_train_up_shift=np.delete(X_train,0,axis=2)
     #Add element 0 to the last row  of each image.
     X_train_up_shift=np.insert(X_train_up_shift,27,0,axis=2)
     print("X_train_up shape:",X_train_up_shift.shape)
 
     #This deletes teh first row  of each image
     X_train_down_shift=np.delete(X_train,27,axis=2)
     #Add element 0 to the last row  of each image.
     X_train_down_shift=np.insert(X_train_down_shift,0,0,axis=2)
     print("X_train_up shape:",X_train_up_shift.shape)
     
     X_train_expanded=np.vstack((X_train_right_shift,X_train_leftt_shift,X_train_up_shift,X_train_down_shift)) 
     print("Shape of expanded:{4 times X_train} ",X_train_expanded.shape)    
     print("y_train shape:",y_train.shape)
  
     #Rotate all the training images between -45degrees and  + 45degrees
     start=time.time()
     rotated_images=rotate_images()
     end=time.time()
     print("Time for rotation:",end-start) 
   
     start=time.time()
     translated_images=translate_images()
     X_train_expanded=np.vstack((X_train_expanded,rotated_images,translated_images)) 
     end=time.time()
     print("Time for translation:",end-start) 
     print("Expanded size:",X_train_expanded.shape)
     return X_train_expanded,np.hstack((y_train,y_train,y_train,y_train,y_train[:10000],y_train[:10000]))

def compute_lables_per_cluster(labels,n_clusters):
    labels_cluster=[]
    for c in range(n_clusters):
        n_labels=np.sum(labels==c)
        labels_cluster.append((c,n_labels))
    print("[cluster,num_labels]:",labels_cluster)

def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.2)
   # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def build_cuda_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    network = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))

    network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train_pca_cnn(num_epochs):
     print("Loading data...")
     X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

     X_train,y_train=expand_images()
     X_train=X_train.reshape(-1,28*28)
     X_test=X_test.reshape(-1,28*28)
     X_val=X_val.reshape(-1,28*28)
     start=time.time()
     pca=PCA(whiten=True)
     X_train=pca.fit_transform(X_train)   
     X_val=pca.transform(X_val)   
     X_test=pca.transform(X_test)   
     print("After PCA new X_train shape:",X_train.shape)
     X_train=X_train.reshape(-1,1,np.sqrt(X_train.shape[1]),np.sqrt(X_train.shape[1]))
     X_val=X_val.reshape(-1,1,np.sqrt(X_val.shape[1]),np.sqrt(X_val.shape[1]))
     X_test=X_test.reshape(-1,1,np.sqrt(X_test.shape[1]),np.sqrt(X_test.shape[1]))
     print("After PCA new X_train shape:",X_train.shape[2])
     print("Time for pca conversion:",time.time() -start)

     input_var = T.tensor4('inputs')
     target_var = T.ivector('targets')
    
     network = lasagne.layers.InputLayer(shape=(None, 1,X_train.shape[2],X_train.shape[2]),input_var=input_var)
     network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
 
     network = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
   
     network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
 
     network = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
    
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
     
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
     prediction = lasagne.layers.get_output(network)
    
     loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
     loss = loss.mean()
    
     params = lasagne.layers.get_all_params(network, trainable=True)
     updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

     test_prediction = lasagne.layers.get_output(network, deterministic=True)
     test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
     test_loss = test_loss.mean()
     test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

     train_fn = theano.function([input_var, target_var], loss, updates=updates)

     val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
     print("Starting training...")
    # We iterate over epochs:
     print(" X_train shape:",X_train.shape[2])
     for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
         train_err = 0
         train_batches = 0
         start_time = time.time()
         for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
             inputs, targets = batch
             train_err += train_fn(inputs, targets)
             train_batches += 1
       # print ("***************************") 
        # And a full pass over the validation data:
         val_err = 0
         val_acc = 0
         val_batches = 0
         for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
             inputs, targets = batch
             err, acc = val_fn(inputs, targets)
             val_err += err
             val_acc += acc
             val_batches += 1
        #print("***************************") 
        # Then we print the results for this epoch:
         print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
             epoch + 1, num_epochs, time.time() - start_time))
         print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
         print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
         print("  validation accuracy:\t\t{:.2f} %".format(
             val_acc / val_batches * 100))
  
         if epoch%5==0:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                   test_acc / test_batches * 100))

     test_err = 0
     test_acc = 0
     test_batches = 0
     for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
         inputs, targets = batch
         err, acc = val_fn(inputs, targets)
         test_err += err
         test_acc += acc
         test_batches += 1
     print("Final results:")
     print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
     print("  test accuracy:\t\t{:.2f} %".format(
         test_acc / test_batches * 100))


def train_pca_kmeans_mlp(num_epochs):
     print("Loading data...")
     x_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
     x_train,y_train=expand_images()
  
     with gzip.open("./train-images-idx3-ubyte.gz", 'rb') as f:
         data = np.frombuffer(f.read(), np.uint8, offset=16)
         data = data.reshape(-1, 28* 28)
         X_train= data / np.float32(256)

     start=time.time()
     pca=PCA(n_components=256,whiten=True)
     X_train=pca.fit_transform(X_train)   
     print("After PCA new X_train shape:",X_train.shape)
     X_train=X_train.reshape(-1,np.sqrt(X_train.shape[1]),np.sqrt(X_train.shape[1]))
     print("After PCA new X_train shape:",X_train.shape)
     print("Time for pca conversion:",time.time() -start)

     ps=(4,4)
     patch_size=ps[0]
     extractor=PatchExtractor(patch_size=ps)
     #Randomly pick the images to learn features from.
     
     indices = np.arange(X_train.shape[0])
     np.random.shuffle(indices)
     patches=extractor.transform(X_train[indices[:200]]).reshape(-1,patch_size*patch_size)
     print("Patches extracted:",np.array(patches,dtype=theano.config.floatX).shape)     
     nc=512

     start=time.time()
     kmeans_clf=KMeans(n_clusters=nc,init='k-means++',n_init=5)
     kmeans_clf.fit(np.array(patches,dtype=theano.config.floatX))
     print("Cluster centers{Dictionary}:",kmeans_clf.cluster_centers_)
     print("Time for clustering:",time.time()-start)
       
     compute_lables_per_cluster(kmeans_clf.labels_,nc)  
    # plot_dictionary_kmeans() 
     centroids = np.array(kmeans_clf.cluster_centers_,dtype=theano.config.floatX)
     print("centroids shape:",centroids.shape)
     w_shared=lasagne.utils.create_param(centroids.reshape(nc,1,4,4),(nc,1,4,4),'w_shared')
     w_shared2=lasagne.utils.create_param(centroids.reshape(1,nc,4,4),(1,nc,4,4),'w_shared')
    
    
     input_var = T.tensor4('inputs')
     target_var = T.ivector('targets')
    
     network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),input_var=input_var)
     network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=nc, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=w_shared)
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
     network1 = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))
     print("Output of max_pool layer-1 shape:",lasagne.layers.get_output_shape(network1))  

     get_reduced_dataset = lasagne.layers.get_output(network1)
     
     extract_features_fn = theano.function([input_var], get_reduced_dataset,on_unused_input='warn')
     print("x_train shape:",x_train[:1].shape)   
     
     reduced_data_set=extract_features_fn(x_train[:2])
     print("reduced data set size:",reduced_data_set.shape)
    
     #Build MLP that should be trained.

     print("Buld 2 layer deep,fully connected  MLP")
     network =lasagne.layers.InputLayer(shape=(None,nc,reduced_data_set.shape[2],reduced_data_set.shape[3]),input_var=input_var)
   
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
     
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    
     prediction = lasagne.layers.get_output(network)
    
     loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
     loss = loss.mean()
    
     params = lasagne.layers.get_all_params(network, trainable=True)
     updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

     test_prediction = lasagne.layers.get_output(network, deterministic=True)
     test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
     test_loss = test_loss.mean()
     test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

     train_fn = theano.function([input_var, target_var], loss, updates=updates)

     val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
     print("Starting training...")
    # We iterate over epochs:
     for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
         train_err = 0
         train_batches = 0
         start_time = time.time()
         for batch in iterate_minibatches(x_train, y_train, 500, shuffle=True):
             inputs, targets = batch
             inputs=extract_features_fn(inputs)
             train_err += train_fn(inputs, targets)
             train_batches += 1
       # print ("***************************") 
        # And a full pass over the validation data:
         val_err = 0
         val_acc = 0
         val_batches = 0
         for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
             inputs, targets = batch
             inputs=extract_features_fn(inputs)
             err, acc = val_fn(inputs, targets)
             val_err += err
             val_acc += acc
             val_batches += 1
        #print("***************************") 
        # Then we print the results for this epoch:
         print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
             epoch + 1, num_epochs, time.time() - start_time))
         print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
         print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
         print("  validation accuracy:\t\t{:.2f} %".format(
             val_acc / val_batches * 100))
  
         if epoch%5==0:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                inputs=extract_features_fn(inputs)
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                   test_acc / test_batches * 100))

     test_err = 0
     test_acc = 0
     test_batches = 0
     for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
         inputs, targets = batch
         inputs=extract_features_fn(inputs)
         err, acc = val_fn(inputs, targets)
         test_err += err
         test_acc += acc
         test_batches += 1
     print("Final results:")
     print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
     print("  test accuracy:\t\t{:.2f} %".format(
         test_acc / test_batches * 100))


def train_shuffled_kmeans_mlp(num_epochs):
#1. Whiten the data samples --if already normalized.
#2.Extract patches of different sizes--start with 4x4
#3.Run k-means clustering on the patches extracted--learn dictionary with k centroids
#4.Use them to extract features and pool perform pooling. 
#5. Build a single layer mlp/SVM that classifies data based on the above training.
     print("Loading data...")
     x_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
     
     with gzip.open("./train-images-idx3-ubyte.gz", 'rb') as f:
         data = np.frombuffer(f.read(), np.uint8, offset=16)
         data = data.reshape(-1, 28, 28)
         X_train= data / np.float32(256)
     
     ps=(4,4)
     patch_size=ps[0]
     extractor=PatchExtractor(patch_size=ps)
     #Randomly pick the images to learn features from.
     
     indices = np.arange(X_train.shape[0])
     np.random.shuffle(indices)
     patches=extractor.transform(X_train[indices[:2000]]).reshape(-1,patch_size*patch_size)
     print("Patches extracted:",np.array(patches,dtype=theano.config.floatX).shape)     
     nc=512
     kmeans_clf=KMeans(n_clusters=nc,init='k-means++',n_init=2)
     kmeans_clf.fit(np.array(patches,dtype=theano.config.floatX))
     print("Cluster centers{Dictionary}:",kmeans_clf.cluster_centers_)
     compute_lables_per_cluster(kmeans_clf.labels_,nc)  
    # plot_dictionary_kmeans() 
     centroids = np.array(kmeans_clf.cluster_centers_,dtype=theano.config.floatX)
     print("centroids shape:",centroids.shape)
     w_shared=lasagne.utils.create_param(centroids.reshape(nc,1,4,4),(nc,1,4,4),'w_shared')
     w_shared2=lasagne.utils.create_param(centroids.reshape(1,nc,4,4),(1,nc,4,4),'w_shared')
    
    
     input_var = T.tensor4('inputs')
     target_var = T.ivector('targets')
    
     network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),input_var=input_var)
     network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=nc, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=w_shared)
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
     network = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))
     print("Output of max_pool layer-1 shape:",lasagne.layers.get_output_shape(network))  
     network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=nc, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify)
     print("Output of conv layer-3 shape:",lasagne.layers.get_output_shape(network))  
     network1 = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))

     print("Output of max_pool layer-2 shape:",lasagne.layers.get_output_shape(network1))  

     get_reduced_dataset = lasagne.layers.get_output(network1)
     
     extract_features_fn = theano.function([input_var], get_reduced_dataset,on_unused_input='warn')
     print("x_train shape:",x_train[:1].shape)   
     
     reduced_data_set=extract_features_fn(x_train[:2])
     print("reduced data set size:",reduced_data_set.shape)
    
     #Build MLP that should be trained.

     print("Buld 2 layer deep,fully connected  MLP")
     network =lasagne.layers.InputLayer(shape=(None,nc,reduced_data_set.shape[2],reduced_data_set.shape[3]),input_var=input_var)
   
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
     
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    
     prediction = lasagne.layers.get_output(network)
    
     loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
     loss = loss.mean()
    
     params = lasagne.layers.get_all_params(network, trainable=True)
     updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

     test_prediction = lasagne.layers.get_output(network, deterministic=True)
     test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
     test_loss = test_loss.mean()
     test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

     train_fn = theano.function([input_var, target_var], loss, updates=updates)

     val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
     print("Starting training...")
     x_train,y_train=expand_images()
    # We iterate over epochs:
     for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
         train_err = 0
         train_batches = 0
         start_time = time.time()
         for batch in iterate_minibatches(x_train, y_train, 500, shuffle=True):
             inputs, targets = batch
             inputs=extract_features_fn(inputs)
             train_err += train_fn(inputs, targets)
             train_batches += 1
       # print ("***************************") 
        # And a full pass over the validation data:
         val_err = 0
         val_acc = 0
         val_batches = 0
         for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
             inputs, targets = batch
             inputs=extract_features_fn(inputs)
             err, acc = val_fn(inputs, targets)
             val_err += err
             val_acc += acc
             val_batches += 1
        #print("***************************") 
        # Then we print the results for this epoch:
         print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
             epoch + 1, num_epochs, time.time() - start_time))
         print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
         print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
         print("  validation accuracy:\t\t{:.2f} %".format(
             val_acc / val_batches * 100))
  
         if epoch%5==0:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                inputs=extract_features_fn(inputs)
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                   test_acc / test_batches * 100))

     test_err = 0
     test_acc = 0
     test_batches = 0
     for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
         inputs, targets = batch
         inputs=extract_features_fn(inputs)
         err, acc = val_fn(inputs, targets)
         test_err += err
         test_acc += acc
         test_batches += 1
     print("Final results:")
     print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
     print("  test accuracy:\t\t{:.2f} %".format(
         test_acc / test_batches * 100))
"""  
"""

def train_kmeans_mlp(num_epochs):
#1. Whiten the data samples --if already normalized.
#2.Extract patches of different sizes--start with 4x4
#3.Run k-means clustering on the patches extracted--learn dictionary with k centroids
#4.Use them to extract features and pool perform pooling. 
#5. Build a single layer mlp/SVM that classifies data based on the above training.
     print("Loading data...")
     x_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
     
     with gzip.open("./train-images-idx3-ubyte.gz", 'rb') as f:
         data = np.frombuffer(f.read(), np.uint8, offset=16)
         data = data.reshape(-1, 28, 28)
         X_train= data / np.float32(256)
 
     ps=(4,4)
     patch_size=ps[0]
     extractor=PatchExtractor(patch_size=ps)
     patches=extractor.transform(X_train[:1000]).reshape(-1,patch_size*patch_size)
     print("Patches extracted:",np.array(patches,dtype=theano.config.floatX).shape)     
  #   x_train_whitened=PCA(whiten=True).fit_transform(X_train) #all componenets kept if n_comp not mentioned.
  #  print("X_train_shape:",x_train_whitened.shape)
     nc=64
     kmeans_clf=KMeans(n_clusters=nc,init='k-means++',n_init=2)
     kmeans_clf.fit(np.array(patches,dtype=theano.config.floatX))
     print("Cluster centers{Dictionary}:",kmeans_clf.cluster_centers_)
     compute_lables_per_cluster(kmeans_clf.labels_,nc)  
    # plot_dictionary_kmeans() 
     centroids = np.array(kmeans_clf.cluster_centers_,dtype=theano.config.floatX)
     print("centroids shape:",centroids.shape)
     w_shared=lasagne.utils.create_param(centroids.reshape(nc,1,4,4),(nc,1,4,4),'w_shared')
    
    
     input_var = T.tensor4('inputs')
     target_var = T.ivector('targets')
    
     network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),input_var=input_var)
     network = lasagne.layers.cuda_convnet.Conv2DCCLayer(
            network, num_filters=nc, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=w_shared)
     print("Output of conv layer-1 shape:",lasagne.layers.get_output_shape(network))  
   
     network1 = lasagne.layers.cuda_convnet.MaxPool2DCCLayer(network, pool_size=(2, 2))
     
     print("Output of max_pool layer-1 shape:",lasagne.layers.get_output_shape(network1))  
     get_reduced_dataset = lasagne.layers.get_output(network1)
     
     extract_features_fn = theano.function([input_var], get_reduced_dataset,on_unused_input='warn')
     print("x_train shape:",x_train[:1].shape)   
     
     reduced_data_set=extract_features_fn(x_train[:1])
     print("reduced data set type:",type(reduced_data_set))
     print("reduced data set size:",reduced_data_set.shape)

     print("Buld 2 layer deep,fully connected  MLP")
     network =lasagne.layers.InputLayer(shape=(None,nc,reduced_data_set.shape[2],reduced_data_set.shape[3]),input_var=input_var)
     
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
     network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    
     prediction = lasagne.layers.get_output(network)
    
     loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
     loss = loss.mean()
    
     params = lasagne.layers.get_all_params(network, trainable=True)
     updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

     test_prediction = lasagne.layers.get_output(network, deterministic=True)
     test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
     test_loss = test_loss.mean()
     test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

     train_fn = theano.function([input_var, target_var], loss, updates=updates)

     val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
     print("Starting training...")
     x_train,y_train=expand_images()
    # We iterate over epochs:
     for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
         train_err = 0
         train_batches = 0
         start_time = time.time()
         for batch in iterate_minibatches(x_train, y_train, 500, shuffle=True):
             inputs, targets = batch
             inputs=extract_features_fn(inputs)
             train_err += train_fn(inputs, targets)
             train_batches += 1
       # print ("***************************") 
        # And a full pass over the validation data:
         val_err = 0
         val_acc = 0
         val_batches = 0
         for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
             inputs, targets = batch
             inputs=extract_features_fn(inputs)
             err, acc = val_fn(inputs, targets)
             val_err += err
             val_acc += acc
             val_batches += 1
        #print("***************************") 
        # Then we print the results for this epoch:
         print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
             epoch + 1, num_epochs, time.time() - start_time))
         print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
         print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
         print("  validation accuracy:\t\t{:.2f} %".format(
             val_acc / val_batches * 100))
         if epoch%5==0:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                inputs=extract_features_fn(inputs)
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                   test_acc / test_batches * 100))

     test_err = 0
     test_acc = 0
     test_batches = 0
     for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
         inputs, targets = batch
         inputs=extract_features_fn(inputs)
         err, acc = val_fn(inputs, targets)
         test_err += err
         test_acc += acc
         test_batches += 1
     print("Final results:")
     print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
     print("  test accuracy:\t\t{:.2f} %".format(
         test_acc / test_batches * 100))

def train_weighted_ensemble_cnn(num_ensembles,num_epochs):
    
    #One list for the predictions of each network.

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train,y_train=expand_images()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    # Create neural network model (depending on first command line parameter)
    print("Building mlp and compiling functions...")
    cnn_network=[]
    
    for network_num in range(num_ensembles):
        cnn_network.append(build_cuda_cnn(input_var))
   
    for epoch in range(num_epochs):
        test_prediction_ensemble=[[] for _ in range(num_ensembles)]
        print("test_prediction_ensemble: ",test_prediction_ensemble)
        for network in cnn_network:
            prediction = lasagne.layers.get_output(network)
            #l1_penalty = regularize_layer_params(network, l1) * 1e-4
            loss = lasagne.objectives.categorical_crossentropy(prediction,
                    target_var)
            #+l1_penalty
            loss = loss.mean()
    
            params = lasagne.layers.get_all_params(network, trainable=True)
            updates = lasagne.updates.nesterov_momentum(
             loss, params, learning_rate=0.01, momentum=0.9)

            test_prediction = lasagne.layers.get_output(network, deterministic=True)
            test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
            test_loss = test_loss.mean()
            test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

            train_fn = theano.function([input_var, target_var], loss, updates=updates)

            val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

            print("Training netowrk:",network)
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                 inputs, targets = batch
                 err, acc = val_fn(inputs, targets)
                 val_err += err
                 val_acc += acc
                 val_batches += 1
            print("One Epoch done.") 
        # Then we print the results for this epoch:
            print("For Nework:",network)
            print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
               epoch + 1, num_epochs, time.time() - start_time))
            print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        test_accuracy_network=[] 
        if epoch%2==0:
           for network,network_id in zip(cnn_network,range(num_ensembles)):
               test_err = 0
               test_acc= 0
               test_batches = 0
               test_prediction = lasagne.layers.get_output(network, deterministic=True)
               test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                    target_var)
               test_loss = test_loss.mean()
               test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)
               test_prediction_max=T.argmax(test_prediction, axis=1)
               val_fn = theano.function([input_var, target_var], [test_loss,
                test_acc,test_prediction_max])
               for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                  inputs, targets = batch
                  err, acc,test_pred= val_fn(inputs, targets)
                  test_err += err
                  test_acc += acc
                  test_batches += 1
                  test_prediction_ensemble[network_id]=test_prediction_ensemble[network_id]+list(test_pred)
               test_accuracy_network.append(np.mean(test_prediction_ensemble[network_id]==np.array([y_test]))*100)

           print("Test accuracy network:",test_accuracy_network)
           weights=np.asarray(np.trunc(np.array([test_accuracy_network]).transpose()),dtype=np.uint8)
           print("Weights:",weights)
           final_pred=np.asarray(np.trunc(np.sum(weights*np.array(test_prediction_ensemble),axis=0)/np.sum(weights)),dtype=np.uint8)
          
           print("Final pred:",final_pred[:100])
           print("y_test:",y_test[:100])
           print("Final shape{0} y_test{1}".format(final_pred.shape,np.array([y_test]).shape))
           print("  test\
                accuracy:\t\t{:.6f}".format(np.mean(np.array([final_pred])==np.array([y_test]))*100))
               
    print("***************************") 
    test_accuracy_network=[] 
    test_prediction_ensemble=[[] for _ in range(num_ensembles)]
    print("test_prediction_ensemble: ",test_prediction_ensemble)
    for network,network_id in zip(cnn_network,range(num_ensembles)):
        test_err = 0
        test_acc = 0
        test_batches = 0
        #setting deterministic=true removes the drop out functionality at mlp
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                    target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)
        test_prediction_max=T.argmax(test_prediction, axis=1)
        val_fn = theano.function([input_var, target_var], [test_loss,
            test_acc,test_prediction_max])
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs,targets = batch
            err, acc,test_pred = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
            test_prediction_ensemble[network_id]=test_prediction_ensemble[network_id]+list(test_pred)
        test_accuracy_network.append(np.mean(test_prediction_ensemble[network_id]==np.array([y_test]))*100)

    print("Test accuracy network:",test_accuracy_network)
    weights=np.asarray(np.trunc(np.array([test_accuracy_network]).transpose()),dtype=np.uint8)
    print("Weights:",weights)
    final_pred=np.asarray(np.trunc(np.sum(weights*np.array(test_prediction_ensemble),axis=0)/np.sum(weights)),dtype=np.uint8)
          
    print("Final pred:",final_pred[:100])
    print("y_test:",y_test[:100])
    print("Final shape{0} y_test{1}".format(final_pred.shape,np.array([y_test]).shape))
    print("  test\
                accuracy:\t\t{:.6f}".format(np.mean(np.array([final_pred])==np.array([y_test]))*100))
               
def train_ensemble_cnn(num_ensembles,num_epochs):
    
    #One list for the predictions of each network.
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train,y_train=expand_images()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    # Create neural network model (depending on first command line parameter)
    print("Building mlp and compiling functions...")
    cnn_network=[]
    
    for network_num in range(num_ensembles):
        cnn_network.append(build_cuda_cnn(input_var))
   
    for epoch in range(num_epochs):
        test_prediction_ensemble=[[] for _ in range(num_ensembles)]
        print("test_prediction_ensemble: ",test_prediction_ensemble)
        for network in cnn_network:
            prediction = lasagne.layers.get_output(network)
            #l1_penalty = regularize_layer_params(network, l1) * 1e-4
            loss = lasagne.objectives.categorical_crossentropy(prediction,
                    target_var)
            #+l1_penalty
            loss = loss.mean()
    
            params = lasagne.layers.get_all_params(network, trainable=True)
            updates = lasagne.updates.nesterov_momentum(
             loss, params, learning_rate=0.01, momentum=0.9)

            test_prediction = lasagne.layers.get_output(network, deterministic=True)
            test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
            test_loss = test_loss.mean()
            test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

            train_fn = theano.function([input_var, target_var], loss, updates=updates)

            val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

            print("Training netowrk:",network)
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                 inputs, targets = batch
                 err, acc = val_fn(inputs, targets)
                 val_err += err
                 val_acc += acc
                 val_batches += 1
            print("One Epoch done.") 
        # Then we print the results for this epoch:
            print("For Nework:",network)
            print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
               epoch + 1, num_epochs, time.time() - start_time))
            print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        if epoch%2==0:
           for network,network_id in zip(cnn_network,range(num_ensembles)):
               test_err = 0
               test_acc = 0
               test_batches = 0
               test_prediction = lasagne.layers.get_output(network, deterministic=True)
               test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                    target_var)
               test_loss = test_loss.mean()
               test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)
               test_prediction_max=T.argmax(test_prediction, axis=1)
               val_fn = theano.function([input_var, target_var], [test_loss,
                  test_acc,test_prediction_max])
               for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                  inputs, targets = batch
                  err, acc,test_pred= val_fn(inputs, targets)
                  test_err += err
                  test_acc += acc
                  test_batches += 1
                  test_prediction_ensemble[network_id]=test_prediction_ensemble[network_id]+list(test_pred)
           final_pred,count=mode(np.array([test_prediction_ensemble]),axis=1)
           print("Final pred:",final_pred)
           print("y_test:",y_test)
           print("Final shape{0} y_test{1}".format(final_pred[0].shape,np.array([y_test]).shape))
           print("  test\
                accuracy:\t\t{:.6f}".format(np.mean(final_pred[0]==np.array([y_test]))*100))
               
    print("***************************") 
    test_prediction_ensemble=[[] for _ in range(num_ensembles)]
    print("test_prediction_ensemble: ",test_prediction_ensemble)
    for network,network_id in zip(cnn_network,range(num_ensembles)):
        test_err = 0
        test_acc = 0
        test_batches = 0
        #setting deterministic=true removes the drop out functionality at mlp
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                    target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)
        test_prediction_max=T.argmax(test_prediction, axis=1)
        val_fn = theano.function([input_var, target_var], [test_loss,
            test_acc,test_prediction_max])
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc,test_pred = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
            test_prediction_ensemble[network_id]=test_prediction_ensemble[network_id]+list(test_pred)
        print("test_pred_ensemble:",np.array([test_prediction_ensemble]).shape)
    final_pred,count=mode(np.array([test_prediction_ensemble]),axis=1)
    print("final_pred:",final_pred)
    print("  test\
                accuracy:\t\t\t{:.6f}".format(np.mean(final_pred[0]==np.array([y_test]))*100))

def train_cnn(num_epochs,input_var=None):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train,y_train=expand_images()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building mlp and compiling functions...")
    network=build_cuda_cnn(input_var)
   
    prediction = lasagne.layers.get_output(network)
    #l1_penalty = regularize_layer_params(network, l1) * 1e-4
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #+l1_penalty
    loss = loss.mean()
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
       # print ("***************************") 
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        #print("***************************") 
        # Then we print the results for this epoch:
        print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if epoch%10==0:
           test_err = 0
           test_acc = 0
           test_batches = 0
           for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
               inputs, targets = batch
               err, acc = val_fn(inputs, targets)
               test_err += err
               test_acc += acc
               test_batches += 1
           print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
           print("  test accuracy:\t\t{:.2f} %".format(
             test_acc / test_batches * 100))
           
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

def train_mlp(num_epochs,input_var=None):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train,y_train=expand_images()
    print(X_train.shape[0],y_train)
    print(X_test.shape[0],y_test)
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building mlp and compiling functions...")
    network=build_mlp(input_var)
   
    prediction = lasagne.layers.get_output(network)
    #l1_penalty = regularize_layer_params(network, l1) * 1e-4
    loss = lasagne.objectives.categorical_crossentropy(prediction,
            target_var)
    #+l1_penalty
    loss = loss.mean()
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
       # print ("***************************") 
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        #print("***************************") 
        # Then we print the results for this epoch:
        print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if epoch%10==0:
           test_err = 0
           test_acc = 0
           test_batches = 0
           for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
               inputs, targets = batch
               err, acc = val_fn(inputs, targets)
               test_err += err
               test_acc += acc
               test_batches += 1
           print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
           print("  test accuracy:\t\t{:.2f} %".format(
              test_acc / test_batches * 100))
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

def train_custom_nn(num_epochs,input_var=None):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train,y_train=expand_images()
    print(X_train.shape[0],y_train)
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building mlp and compiling functions...")
 
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_in, num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.2)
   # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
  
    network=l_out
   
    prediction = lasagne.layers.get_output(network)
    #l1_penalty = regularize_layer_params(network, l1) * 1e-4
    loss = lasagne.objectives.categorical_crossentropy(prediction,
            target_var)
    #+l1_penalty
    loss = loss.mean()
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
       # print ("***************************") 
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        #print("***************************") 
        # Then we print the results for this epoch:
        print("Epoch of complete batch learning {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training  error:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation error:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if epoch%10==0:
           test_err = 0
           test_acc = 0
           test_batches = 0
           for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
               inputs, targets = batch
               err, acc = val_fn(inputs, targets)
               test_err += err
               test_acc += acc
               test_batches += 1
           print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
           print("  test accuracy:\t\t{:.2f} %".format(
              test_acc / test_batches * 100))
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

def DecisionTree(selector='pca'):
    dt_clf=DecisionTreeClassifier(criterion='entropy',max_depth=20,min_samples_leaf=5)
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
   # X_train,y_train=expand_images()
    X_train=np.vstack((X_train,X_val))
    y_train=np.hstack((y_train,y_val))
    print("X_train.shape=",X_train.shape)
    X_train=np.asarray(X_train.reshape(-1,X_train.shape[2]*X_train.shape[3]),dtype=theano.config.floatX) 
    X_test=np.asarray(X_test.reshape(-1,X_test.shape[2]*X_test.shape[3]),dtype=theano.config.floatX) 
    print("X_train.shape after transformation =",X_train.shape)
    print("X_test.shape after transformation =",X_test.shape)
    x_train=X_train
    x_test=X_test

    #Use RFE and sVM for feature selection  --PCA(poor performance for <100)
    for n_features in [10,20,25,30,35,40,50,60,70,80,90,100,120,150,200,350,784]:
        print("Number of features:",n_features)
        print("selector:",selector)
        if selector =='rfe':
           start=time.time() 
           rfe_selector=RFE(dt_clf,n_features_to_select=n_features)
           X_train=rfe_selector.fit_transform(x_train,y_train)
           X_test=rfe_selector.transform(x_test)
           print("New shape of x_train:",X_train.shape)
           print("Time to feature select:",time.time()-start)
        elif selector=='pca':
           start=time.time() 
           pca_selector=PCA(n_components=n_features)
           X_train=pca_selector.fit_transform(x_train,y_train)
           X_test=pca_selector.transform(x_test)
           print("New shape of x_train:",X_train.shape)
           print("Time to feature select:",time.time()-start)
        elif selector=='sfm':
           start=time.time() 
           sfm_selector=SelectFromModel(dt_clf)
           X_train=sfm_selector.fit_transform(x_train,y_train)
           X_test=sfm_selector.transform(x_test)
           print("New shape of x_train:",X_train.shape)
           print("Time to feature select:",time.time()-start)
    
        start=time.time() 
        dt_clf.fit(X_train,y_train)
        print("Time to train:",time.time()-start)

        start=time.time() 
        pred=dt_clf.predict(X_test)
        pred=np.array(pred)
        print("First few elements of pred:",pred[:20])
        y_test=y_test.transpose()
        print("First few elements of y_test:",y_test[:20],np.mean(pred[:5]==y_test[:5]))
        print("Test Accuracy:",np.mean(pred==y_test)*100)
        print("Time to test:",time.time()-start)

def Linear_SVM(selector='pca'):
    svm_clf=LinearSVC(penalty='l2', loss='hinge', tol=0.0001, C=0.2, multi_class='ovr', fit_intercept=True)
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
   # X_train,y_train=expand_images()
    X_train=np.vstack((X_train,X_val))
    y_train=np.hstack((y_train,y_val))
    print("X_train.shape=",X_train.shape)
    X_train=np.asarray(X_train.reshape(-1,X_train.shape[2]*X_train.shape[3]),dtype=theano.config.floatX) 
    X_test=np.asarray(X_test.reshape(-1,X_test.shape[2]*X_test.shape[3]),dtype=theano.config.floatX) 
    print("X_train.shape after transformation =",X_train.shape)
    print("X_test.shape after transformation =",X_test.shape)
    x_train=X_train
    x_test=X_test

    #Use RFE and sVM for feature selection  --PCA(poor performance for <100)
    for n_features in [100,150,200,300,350,400,450,500]:
        print("Number of features:",n_features)
        print("selector:",selector)
        if selector =='rfe':
           start=time.time() 
           rfe_selector=RFE(svm_clf,n_features_to_select=n_features)
           X_train=rfe_selector.fit_transform(x_train,y_train)
           X_test=rfe_selector.transform(x_test)
           print("New shape of x_train:",X_train.shape)
           print("Time to feature select:",time.time()-start)
        elif selector=='pca':
           start=time.time() 
           pca_selector=PCA(n_components=n_features)
           X_train=pca_selector.fit_transform(x_train,y_train)
           X_test=pca_selector.transform(x_test)
           print("New shape of x_train:",X_train.shape)
           print("Time to feature select:",time.time()-start)
        elif selector=='sfm':
           start=time.time() 
           sfm_selector=SelectFromModel(svm_clf,threshold=0.3)
           X_train=sfm_selector.fit_transform(x_train,y_train)
           X_test=sfm_selector.transform(x_test)
           print("New shape of x_train:",X_train.shape)
           print("Time to feature select:",time.time()-start)
    
        start=time.time() 
        svm_clf.fit(X_train,y_train)
        print("Time to train:",time.time()-start)

        start=time.time() 
        pred=svm_clf.predict(X_test)
        pred=np.array(pred)
        print("First few elements of pred:",pred[:20])
        y_test=y_test.transpose()
        print("First few elements of y_test:",y_test[:20],np.mean(pred[:5]==y_test[:5]))
        print("Test Accuracy:",np.mean(pred==y_test)*100)
        print("Time to test:",time.time()-start)

def main(model='mlp', num_epochs=500,n_ensembles=5):
    if model == 'mlp':
        train_mlp(num_epochs)
    elif model=='nn':
         train_custom_nn(num_epochs)
    elif model == 'cnn':
         train_cnn(num_epochs)
    elif model == 'cnn-pca':
         train_pca_cnn(num_epochs)
    elif model == 'k-means':
         train_kmeans_mlp(num_epochs)
    elif model == 'k-means-random':
         train_shuffled_kmeans_mlp(num_epochs)
    elif model =='ensemble-cnn':
         train_ensemble_cnn(n_ensembles,num_epochs)
    elif model =='ensemble-cnn-weighted':
         train_weighted_ensemble_cnn(n_ensembles,num_epochs)
    elif model=='k-means-pca':
         train_pca_kmeans_mlp(num_epochs)
    elif model=='svm':
         Linear_SVM()
    elif model=='dt':
         DecisionTree()
    else:
        print("Unrecognized model type %r." % model)
        return


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS][NUM_ENSEMBLES]]" % sys.argv[0])
        print("Model: svm dt k-means k-means-pca k-means-random cnn mlp\
                ensemble-cnn ensemble-cnn-weighted nn cnn-pca")
        print()
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        if len(sys.argv) > 3 :
           kwargs['n_ensembles']=int(sys.argv[3])
     
        main(**kwargs)
