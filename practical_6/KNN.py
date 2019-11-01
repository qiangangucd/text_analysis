import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv('iris.csv', header=None)
split = 0.6
k = [2, 6, 11, 16, 19]
# k = 6
# X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:, :-1], iris.iloc[:, -1], test_size=1 - split,
#                                                     random_state=10)


acc = []
for i in k:
    neigh = KNeighborsClassifier(n_neighbors=i)
    acc.append(np.mean(cross_val_score(neigh, iris.iloc[:, :-1], iris.iloc[:, -1], cv=5)))
print(acc)
plt.plot(k,acc)
plt.xticks(k)
plt.yticks(acc)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.savefig('cvl.png',dpi=100,bbox_inches='tight')
plt.show()



# acc = []
# for i in k:
#     neigh = KNeighborsClassifier(n_neighbors=i)
#     neigh.fit(X_train, y_train)
#     acc.append(neigh.score(X_test, y_test))
# print(acc)
# plt.plot(k,acc)
# plt.xticks(k)
# plt.yticks(acc)
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.savefig('knn.png',dpi=100,bbox_inches='tight')
# plt.show()
