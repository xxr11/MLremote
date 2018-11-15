from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
# 花的萼长，宽，花瓣的长，宽
# 字典
iris = datasets.load_iris()

print(iris)

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[6.2, 3.4, 5.4, 2.3]])
print(predictedLabel)
