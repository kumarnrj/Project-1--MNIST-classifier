# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:44:17 2019

@author: NR
"""
#importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets form sklearn
from sklearn import datasets
minist = datasets.load_digits()
print(minist)
X= minist.data
y= minist.target
#spilitng the data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#fiting the model
from sklearn.neighbors import KNeighborsClassifier
kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier

for k in range(1, 30, 2):
          # train the k-Nearest Neighbor classifier with the current value of `k`
          model = KNeighborsClassifier(n_neighbors=k)
          model.fit(X_train, y_train)
          # evaluate the model and update the accuracies list
          score = model.score(X_test, y_test)
          print("k=%d, accuracy=%.2f%%" % (k, score * 100))
          accuracies.append(score)
y_pred = model.predict(X_test)

#confusion matrxi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

for i in np.random.randint(0, high=len(y_test), size=(5,)):
         # grab the image and classify it
         image = X_test[i]
         prediction = model.predict([image])[0]
         
         # show the prediction
         
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((8,8))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print(" digit is : {}".format(prediction))
         plt.show()


#fitting the logestic Regression
         
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#score 
classifier.score(X_train,y_train)
classifier.score(X_test,y_test)

#visulization
for i in np.random.randint(0, high=len(y_test), size=(5,)):
         # grab the image and classify it
         image = X_test[i]
         prediction = model.predict([image])[0]
         
         # show the prediction
         
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((8,8))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print(" digit is : {}".format(prediction))
         plt.show()


#fittin svm
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

classifier.score(X_train,y_train)
classifier.score(X_test,y_test)
#visulization
for i in np.random.randint(0, high=len(y_test), size=(5,)):
         # grab the image and classify it
         image = X_test[i]
         prediction = model.predict([image])[0]
         
         # show the prediction
         
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((8,8))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print(" digit is : {}".format(prediction))
         plt.show()

