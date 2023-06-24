from sklearn.mixture import GaussianMixture
import numpy as np
from MILframe import MIL
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
instances = MIL.MIL('../MILframe/data/benchmark/musk1.mat').ins
train_data = instances[:80]
test_data = instances[81:]
# print(train_data[0].reshape(1, -1))
gmm = GaussianMixture(n_components=2)
gmm.fit(train_data)
print(test_data[0].reshape(1, -1))
# print(gmm.predict_proba(test_data[0].reshape(1, -1)))



