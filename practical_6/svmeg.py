__author__ = 'user'
# http://pythonprogramming.net/support-vector-machine-svm-example-tutorial-scikit-learn-python/

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

digits = datasets.load_digits()
x, y = digits.data[:-1], digits.target[:-1]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)
# classifier = svm.SVC(gamma=0.01, C=100,kernel='linear')
# classifier.fit(train_x, train_y)
# y_predict = classifier.predict(test_x)
# accuracy = metrics.accuracy_score(test_y, y_predict)
# print(accuracy)

# plt.figure(figsize=(18,8))
# c=[1,10,20,30,40,50,60,70,80,90,100]
# gamma=[0.001,0.002,0.003,0.004,0.005,0.01,0.02,0.03,0.04,0.05,0.06]
kernel=['linear', 'poly', 'rbf']

acc=[]
for i in range(len(kernel)):
    classifier = svm.SVC(gamma=0.01, C=200, kernel=kernel[i])
    classifier.fit(train_x, train_y)
    y_predict=classifier.predict(test_x)
    accuracy=metrics.accuracy_score(test_y,y_predict)
    print(i)
    print(accuracy)
    acc.append(accuracy)

# y_predict=classifier.predict(test_x)
# accuracy=metrics.accuracy_score(test_y,y_predict)
# print(accuracy)
print(acc)
plt.plot(range(3),acc)
plt.xlabel('kernel')
plt.ylabel('acc')
# plt.savefig('c.png',dpi=100,bbox_inches='tight')
# # plt.savefig('gamma.png',dpi=100,bbox_inches='tight')
plt.savefig('kernel.png',dpi=100,bbox_inches='tight')
plt.show()

#
#
# print('Prediction:', classifier.predict(digits.data[-1].reshape(1,-1)))
#
# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
