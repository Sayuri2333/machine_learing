import numpy as np
import random
# import warnings
# warnings.filterwarnings("ignore")

"""
函数说明:sigmoid函数

Parameters:
    inX - 数据
Returns:
    sigmoid函数

"""


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明:改进的随机梯度上升算法

Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)

"""
def stocGradAscent1(dataMatIn, classLabels, numIter=150):
    dataMatrix = np.array(dataMatIn)
    m, n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化
    weights_array = np.array([])                                               #储存每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        # 作者脑子应该吃了屎 以下代码的意义相当于打乱了dataMatrix的顺序而已对于每一个元素都要做迭代
        # 不仅如此 在每一个元素循环的时候都要更新weight权重 会不会使得权重值更难稳定
        for i in range(m):
            # 降低alpha的大小从而防止后期overshooting
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            # 使用randIndex来随机引用样本
            randIndex = int(random.uniform(0, len(dataIndex)))                #随机选取样本
            theChosenOne = dataIndex[randIndex]
            # 更新回归系数的时候只用到一个样本
            h = sigmoid(sum(dataMatrix[theChosenOne]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[theChosenOne] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[theChosenOne]       #更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)         #添加回归系数到数组中
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    weights_array = weights_array.reshape(numIter * m, n)
    return weights


"""
函数说明:使用Python写的Logistic分类器做预测

Parameters:
    无
Returns:
    无

"""

def colicTest():
    frTrain = open('horseColicTraining.txt')                                        #打开训练集
    frTest = open('horseColicTest.txt')                                                #打开测试集
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():  # 读入数据集中的每一行
        currLine = line.strip().split('\t')  # 对于每一行将其分割为几个值
        lineArr = []  # 存储每一行的特征值
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))  # 将此行的所有特征值存储到lineArr中
        trainingSet.append(lineArr)  # 将此行的特征值存储到训练集中
        trainingLabels.append(float(currLine[-1]))  # 最后一个值作为此行的判断值y
    trainWeights = gradAscent(trainingSet, trainingLabels)        # 使用梯度上升(原始)
    errorCount = 0  # 错误计数
    numTestVec = 0.0  # 测试集的总样本量
    for line in frTest.readlines():  # 读入测试集
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), np.mat(trainWeights)))!= int(currLine[-1]):  # 训练的结果与实际结果不一致的话
            errorCount += 1  # 错误计数加1
    errorRate = (float(errorCount)/numTestVec) * 100                                 #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)


"""
函数说明:分类函数

Parameters:
    inX - 特征向量
    weights - 回归系数
Returns:
    分类结果

"""

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


"""
函数说明:梯度上升算法

Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置(m行1列)
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()  # 将矩阵转换为数组，并返回

colicTest()