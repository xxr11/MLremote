import csv
import random
import math
import operator


# 读取文件数据--csv
# 随机加进trainset和testset
def loadDataset(filename, split, trainSet=[], testSet=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        for x in range(len(dataSet) - 1):
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
            if random.random() < split:
                trainSet.append(dataSet[x])
            else:
                testSet.append(dataSet[x])


# 多维计算距离///，length可以表示维度
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    # 获取未知点的维度
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        # 数据集里到目标点的距离
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    # 取前k个距离
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# python3消除了iteritems，只能自定义返回字典列表
def dict2list(dic):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes.keys():
            classVotes[response] += 1
        else:
            classVotes[response] = 1
            sortedVotes = sorted(classVotes.items(), key=lambda classVotes: classVotes[1], reverse=False)
    return sortedVotes[-1]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    trainSet = []
    testSet = []
    split = 0.67
    loadDataset(r'iris.data.txt', split, trainSet, testSet)
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result[0])
        print('>predicted=' + repr(result) + ',actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy: " + repr(accuracy) + '%')


main()
