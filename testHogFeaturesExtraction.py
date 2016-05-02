from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import util
import matplotlib.pyplot as plt


dataset = datasets.fetch_mldata("MNIST Original")
(xtr, xte, ytr, yte) = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)

xtrain, xtest = util.getHogFeatures(xtr, xte, 4)

clf = KNeighborsClassifier()
clf = clf.fit(xtrain, ytr)
y_pred = clf.predict(xtest)
print "KNN score ", clf.score(xtest,yte)


clf = DecisionTreeClassifier()
clf = clf.fit(xtrain, ytr)
y_pred = clf.predict(xtest)
print "DT score ", clf.score(xtest,yte)

clf = GaussianNB()
clf = clf.fit(xtrain, ytr)
y_pred = clf.predict(xtest)
print "GaussianNB score ", clf.score(xtest,yte)

clf = BernoulliNB()
clf = clf.fit(xtrain, ytr)
y_pred = clf.predict(xtest)
print "BernoulliNB score ", clf.score(xtest,yte)

clf = SVC(kernel = 'rbf', C=1000, gamma=0.001)
clf = clf.fit(xtrain, ytr)
y_pred = clf.predict(xtest)
print "SVC score ", clf.score(xtest,yte)


"""
(14,14)
KNN score  0.917662337662
DT score  0.789393939394
GaussianNB score  0.842510822511
BernoulliNB score  0.619264069264
SVC score  0.879956709957

(7,7)
KNN score  0.914415584416
DT score  0.844848484848
GaussianNB score  0.750432900433
BernoulliNB score  0.838181818182
SVC score  0.94194805194

(4,4)
KNN score  0.949220779221
DT score  0.866666666667
GaussianNB score  0.740346320346
BernoulliNB score  0.901774891775
SVC score  0.96329004329
"""
