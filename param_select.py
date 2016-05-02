import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import datasets
from sklearn.grid_search import GridSearchCV




dataset = datasets.fetch_mldata("MNIST Original")


X_train, X_test, y_train, y_test = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)

svc_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

knn_parameters = [{'n_neighbors': [1,5,10,20]}] 

print("Starting knn")
clf1 = GridSearchCV(KNeighborsClassifier(), knn_parameters, cv=5)
score1 = clf1.fit(X_train,y_train).score(X_test,y_test)
best_parameters = clf1.best_estimator_.get_params()
print(best_parameters)
print("Score1: ",score1)
print("Starting SVC")
clf2 = GridSearchCV(SVC(C=1), svc_parameters, cv=5)
score2 = clf2.fit(X_train,y_train).score(X_test,y_test)
best_parameters = clf2.best_estimator_.get_params()
print(best_parameters)
print("Score2: ",score2)
print("Starting DT")
clf3 = DecisionTreeClassifier()
score3 = clf3.fit(X_train,y_train).score(X_test,y_test)
print("Score3: ",score3)
print("Starting gaussian naive bayes")
clf4 = GaussianNB()
score4 = clf4.fit(X_train,y_train).score(X_test,y_test)
print("Score4: ",score4)
print("Starting bernoulli naive bayes")
clf5 = BernoulliNB()
score5 = clf5.fit(X_train,y_train).score(X_test,y_test)
print("Score5: ",score5)

y = (score1,score2,score3,score4.score5)
N = len(y)
x = range(N)

ind = np.arange(N)    
width = 0.35 
plt.bar(x, y, width, color="blue")

plt.ylabel('Scores')
plt.title('Scores by classifier')
plt.xticks(ind, ('knn','svc','dt','gaussian','bernoulli'))

plt.show()