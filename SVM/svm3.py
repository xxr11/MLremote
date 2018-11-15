# 线性不可分向量机
from time import time
import logging
import matplotlib as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print(lfw_people.keys())
print('oooooooooooo')
# n_samples, h, w = lfw_people.image.shape
