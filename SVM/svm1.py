from sklearn import svm

X = [[2, 0], [3, 0], [1, 1], [2, 3]]
y = [0, 0, 0, 1]
# get support vectors
clf = svm.SVC(kernel='linear')
# get indices of support vectors
clf.fit(X, y)
# 查看支持向量
print(clf.support_vectors_)
# 支持向量索引
print('---------------------')
print(clf.support_)
print('---------------------')
# 找到的支持向量索引
# get number of support vectors for each class
print(clf.n_support_)
