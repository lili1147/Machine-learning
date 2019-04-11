# -*- coding: utf-8 -*-
# @Author: lili
# @Date:   2019-04-08 14:24:54
# @Last Modified by:   lili1147
# @Last Modified time: 2019-04-08 21:40:06


'''
从文本文件中读取数据并进行聚类
'''
from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def LodaData(filename):
    DataMatrix = []
    f = open(filename)
    for line in f:
        curLine = line.strip().split('\t')
        resLine = map(float, curLine)  # map(func,iterable) func函数对迭代器中的每个元素进行处理。
        DataMatrix.append(list(resLine))
    return array(DataMatrix)


def CalDist(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataset, k):
    '''
        随机生成k个质心，并确保质心在数据范围内，采用一列的最大值减去最小值乘以0到1的随机值
        '''
    n = shape(dataset)[1]
    dataset = mat(dataset)
    centroids = mat(zeros((k, n)))  # centroids 质心
    for j in range(n):
        minJ = min(dataset[:, j])
        rangeJ = float(max(dataset[:, j]) - minJ)
        print(rangeJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def draw_scatter(dataset):
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1])
    plt.savefig('Kmeans.jpg')


def KMeans(dataset, k, distMeas=CalDist, creatCent=randCent):
    '''
    计算质心——分配——重新计算
    用来计算距离和创建质心的函数都是可以变动的
    '''
    m = shape(dataset)[0]  # 数据总数
    clusterAssement = mat(zeros((m, 2)))
    centroids = creatCent(dataset, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distJ = distMeas(centroids[j, :], dataset[i, :])
                if distJ < minDist:
                    minDist = distJ
                    minIndex = j
            if clusterAssement[i, 0] != minIndex:
                clusterChanged = True
            clusterAssement[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            # print('----')
            # print(dataset[nonzero(clusterAssement[:, 0].A == 1)[0]])
            # print('----')
            ptsInClust = dataset[nonzero(clusterAssement[:, 0].A == cent)[0]]  # 找出dataset中对应的clusterAssement簇
            # print(ptsInClust)
            # print('----')
            centroids[cent, :] = mean(ptsInClust, axis=0)
        return centroids, clusterAssement


def biKmeans(dataSet, k, distMeas=CalDist):
    m = shape(dataSet)[0]  # 数据集样本数目
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # 初始化质心矩阵
    for j in range(m):  # 计算数据集与初始质心之间的SSE
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf  # 初始SSE设为无穷大
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  # 通过数组过滤获得对应簇i的数据集样本
            centroidMat, splitClustAss = KMeans(ptsInCurrCluster, 2, distMeas)  # K -均值算法会生成两个质心(簇），同时给出每个簇的误差值
            sseSplit = sum(splitClustAss[:, 1])  # 当前数据集的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 未划分数据集的SSE
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  # 更新最小误差值
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将k-means得到的0、1结果簇重新编号，修改为划分簇及新加簇的编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 将原来的质心点替换为具有更小误差对应的质心点
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # 重新构建簇矩阵
    return mat(centList), clusterAssment

# 聚类结果显示


def showCluster(dataSet, k, centroids, clusterAssment):
    m = shape(dataSet)[0]  # 行数
    n = shape(dataSet)[1]  # 列数
    if n != 2:
        print("sorry, can't draw because your data dimension is 2")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("sorry, your K is too large")
        return 1
    # 绘制所有点对颜色
    for i in range(m):
        # print(clusterAssment[i, 0])
        markIndex = int(clusterAssment[i, 0])  # 为样本指定颜色
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制簇中心点对颜色
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


if __name__ == '__main__':
    res = LodaData('F:/机器学习实战/源代码及数据集/machinelearninginaction/Ch10/testSet2.txt')
    # randCent(res, 4)
    # draw_scatter(res)
    # centroids, clusterAssement = KMeans(res, 4)
    centroids, clusterAssement = biKmeans(res, 3)
    # print(centroids)
    # showCluster(res, 3, centroids, clusterAssement)
    for i in range(3):
        PtsInClust = res[np.nonzero(clusterAssement[:, 0].A == i)[0]]
        if i == 0:
            plt.scatter(PtsInClust[:, 0].tolist(), PtsInClust[:, 1].tolist(),
                        marker='x', color='r', label='0', s=12)
            plt.scatter(centroids[i, 0], centroids[i, 1], color='r', marker='+', s=180)
        if i == 1:
            plt.scatter(PtsInClust[:, 0].tolist(), PtsInClust[:, 1].tolist(),
                        marker='s', color='g', label='1', s=12)
            plt.scatter(centroids[i, 0], centroids[i, 1], color='g', marker='+', s=180)
        if i == 2:
            plt.scatter(PtsInClust[:, 0].tolist(), PtsInClust[:, 1].tolist(),
                        marker='p', color='b', label='2', s=12)
            plt.scatter(centroids[i, 0], centroids[i, 1], color='b', marker='+', s=180)

    plt.show()
