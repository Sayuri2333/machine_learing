import matplotlib.pyplot as plt
import numpy as np
import random
import types

"""
函数说明:读取数据

Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签
"""


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return dataMat, labelMat


"""
函数说明:数据可视化

Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
Returns:
    无
"""


def showDataSet(dataMat, labelMat):
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    # 下面矩阵需要转置的原因是scatter(x,y)接受所有点的x坐标的数组和所有点的y坐标的数组
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])  # 负样本散点图
    plt.show()


"""
函数说明:随机选择alpha

Parameters:
    i - alpha
    m - alpha参数个数
Returns:
    j -
"""


def selectJrand(i, m):
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
        return j


"""
函数说明:修剪alpha

Parameters:
    aj - alpha值
    H - alpha上限
    L - alpha下限
Returns:
    aj - alpha值
"""


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
函数说明:简化版SMO算法

Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
Returns:
    无
"""


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 将给定的数据集和标签集合转化为mat矩阵
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    # 初始化b的值
    b = 0
    # 获得数据集的大小
    m, n = np.shape(dataMatrix)
    # 初始化alpha的值为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 在迭代次数的范围内
    while (iter_num < maxIter):
        # 转换了多少对alpha值
        alphaPairsChanged = 0
        # 在本次迭代中也要遍历整个数据集
        for i in range(m):
            # 步骤1: 计算当前Ei和f(xi)
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 优化alpha, 并设置一定的容错率
            # 只有当i的分类误差大于阈值，且alphas[i]在0到c之间时才对alphas[i]进行更新
            # 也就是当前选择的点不符合KKT条件才进行更新
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择与i配套的另一个j
                j = selectJrand(i, m)
                # 步骤1: 计算对应的Ej和f(xj)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 深拷贝当前的old的ai和aj
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 步骤2: 计算对应的L(下界)和H(上界)
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print('L==H'); continue
                # 步骤3: 计算eta(学习速率)
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                # 步骤4: 更新aj
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 步骤5: 修剪aj
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 如果变化幅度太小就抛弃此次修改, abs表示取绝对值
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("too little change!")
                    continue
                # 步骤6: 更新ai
                alphas[i] += labelMat[i] * labelMat[j] * (alphaJold - alphas[j])
                # 步骤7: 更新b1和b2
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 步骤8: 根据b1和b2更新b
                if (0 < alphas[i]) and (C > alphas[i]):  # 0 < ai < c 意味着对应的i为支持向量,可以用来更新
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):  # 0 < aj < c 意味着对应的j为支持向量, 可以用来更新
                    b = b2
                else:  # 都不是支持向量 就取中间值更新
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))
        # 更新迭代次数
        if(alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alphas


"""
函数说明:分类结果可视化

Parameters:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线解决
Returns:
    无
"""


def showClassifier(dataMat, labelMat, w, b):
    # 绘制样本点
    data_plus = []  # 存储正样本
    data_minus = []  # 存储负样本
    for i in range(len(dataMat)):  # 遍历数据集
        # 分类添加正负样本
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 将正负样本数据集转换为numpy数组
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 画出样本分布图 (numpy数组的转置需要用到np.transpose方法)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=.7)
    # 绘制直线
    # 获得左右端点的x值
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    # 获得w
    a1, a2 = w
    # 获得b
    b = float(b)

    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点 enumerate方法可以将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            # alpha > 0的值对应的点就是支持向量点
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=.7, linewidths=1.5, edgecolors='red')
    plt.show()


"""
函数说明:计算w

Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
    alphas - alphas值
Returns:
    无
"""


def get_w(dataMat, labelMat, alphas):
    # 获得alpha, dataMat, labelMat的数据
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    # 计算w的值, 其中np.tile(A, reps)方法是用来扩充维度的，A指待输入数组，reps则决定A重复的次数; reshape方法(1, -1)为将整个矩阵转换为一行的意思(不管有多少列)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()

dataMat, labelMat = loadDataSet('testSet.txt')
b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
w = get_w(dataMat, labelMat, alphas)
showClassifier(dataMat, labelMat, w, b)