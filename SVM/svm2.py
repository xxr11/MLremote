import numpy as np
import pylab as pl
from sklearn import svm

# 每次运行结果不变
np.random.seed(0)
# 20行，2维的矩阵，+，-表示在上边和下边
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# W=(w1,w2,....)
# w0*x1+w1*x2+w2=y
w = clf.coef_[0]
# 二维
a =- w[0] / w[1]
print(a)
# 从-5到5产生连续的值
xx = np.linspace(-5, 5)
# clf.intercept_[0]取得是截距bias
yy = a * xx - (clf.intercept_[0] / w[1])

b = clf.support_vectors_[0]
# 下方的线
yy_dom = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
# 上方的线
yy_up = a * xx + (b[1] - a * b[0])

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_dom, 'k--')
pl.plot(xx, yy_up, 'k--')
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, edgecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
