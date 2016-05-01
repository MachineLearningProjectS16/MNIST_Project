import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from skimage.feature import hog

def plotBarGraph(labels, values, title, xLabel, yLabel, filename):
    #Create values and labels for bar chart
    inds   =np.arange(len(values))

    #Plot a bar chart
    plt.figure(1, figsize=(20,10))  #6x4 is the aspect ratio for the plot
    plt.bar(inds, values, align='center') #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel(yLabel) #Y-axis label
    plt.xlabel(xLabel) #X-axis label
    plt.title(title) #Plot title
    plt.xlim(-1,len(values)) #set x axis range
    plt.xticks(range(1, len(labels)), labels, rotation='vertical')
    minvalue = min(values)
    maxvalue = max(values)
    plt.ylim(minvalue-minvalue, minvalue+maxvalue) #Set yaxis range

    #Set the bar labels
    plt.gca().set_xticks(inds) #label locations
    plt.gca().set_xticklabels(labels) #label values
    #Save the chart
    filename = "./Figures/"+filename+".pdf"
    plt.savefig(filename)

def getPrincipleComponents(xtr, xte, n_components=50):
    train = np.array(xtr)
    test = np.array(xte)
    pca = RandomizedPCA(n_components=n_components).fit(train)
    xtrain = pca.transform(train)
    xtest = pca.transform(test)
    return xtrain, xtest

def getHogFeatures(xtr, xte, per_cell = 7):
    train = [hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(per_cell, per_cell), cells_per_block=(1, 1), visualise=False) for feature in xtr]
    test = [hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(per_cell, per_cell), cells_per_block=(1, 1), visualise=False) for feature in xte]
    return np.array(train, 'float64'), np.array(test, 'float64')
