from numpy import *


def loadDataSet(fileName):#文件读取函数，读取txt
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat


def distEculd(vecA,vecB):# 计算欧几里得距离
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):# 随机抽取k个中心
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    dataSet=mat(dataSet)
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j])-minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k,1)
    return centroids


def kMeans(dataSet, k, distMeas = distEculd, createCent = randCent): #标准kMeans算法
    m = shape(dataSet)[0]
    dataSet=mat(dataSet)
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)   #初始中心
    clusterChanged = True   # 判定变量，如果分类发生变化为True，分类不变为False跳出迭代
    while clusterChanged:
        clusterChanged =False
        for i in range(m):  # 对每个数据点计算到k个中心的距离，选出最小的
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if(clusterAssment[i,0] != minIndex): clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k): # 计算k个类的重心作为下一轮迭代的中心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # ?
            centroids[cent,:] = mean(ptsInClust, axis = 0)
    return(centroids, clusterAssment) #输出结果为centroids：中心，clusterAssment：每个点的分类和到中心的距离


def biKmeans(dataSet, k, distMeas = distEculd): #二分聚类
    m = shape(dataSet)[0]
    dataSet=mat(dataSet)
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while(len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A!=i)[0],1]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notsplit:",sseSplit,sseNotSplit)
            if((sseSplit + sseNotSplit) < lowestSSE):
                bestCentToSplit = i
                bestNewCent = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is:', bestCentToSplit)
        print('the len of bestClustToSplit', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCent[0, :]
        centList.append(bestNewCent[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return(mat(centList), clusterAssment)
