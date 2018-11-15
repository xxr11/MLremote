from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
import numpy as np

allElectronicsData = open(r'buy_compuetr.csv', 'r', encoding='UTF-8')
reader = csv.reader(allElectronicsData)
headers = ['ID', 'age', 'Income', 'student', 'credit_rating']
print(reader)
featureList = []
labelList = []
# 对离散分类数据进行类似one-hot编码
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        # 将需要转换的数据变成字典列表，一个字典就是一条数据记录
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

# 转换属性行为one-hot编码
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print(dummyX)
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(dummyY)
clf = tree.DecisionTreeClassifier(criterion='entropy')
# 创建分类器
clf.fit(dummyX, dummyY)
# 画出结果

with open("allElectronicInformationGainOri.dot", 'w')as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

    # 自定义测试
oneRowX = dummyX[0, :]
print(oneRowX)
newRowX = oneRowX
print(type(newRowX))
newRowX[0] = 1
newRowX[2] = 0
l1 = np.array([[1, 2], [3, 4], [5, 6]])
print('-----------------------')
print(str(l1.reshape(2, 3)))
predictedY = clf.predict([newRowX])
print(str(predictedY))
