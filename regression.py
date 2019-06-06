import matplotlib.pyplot as plt
import numpy as np


def loadDataset(filename):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        xArr - x数据集
        yArr - y数据集
    """

    numFeat = len(open(filename).readline().split('\t')) - 1  # 获得当前样本中的特征数量
    xArr = []  # 存储当前样本中的所有特征
    yArr = []  # 存储当前样本中的所有y值
    fr = open(filename)  # 文件对象
    for line in fr.readlines():
        lineArr = []  # 存储当前文件中每一行的特征值
        curLine = line.strip().split('\t')  # 使用tab间隔符分割当前每一行
        for i in range(numFeat):  # 遍历所有特征
            lineArr.append(float(curLine[i]))  # 将特征存储到lineArr中
        xArr.append(lineArr)  # x数组增添当前特征
        yArr.append(float(curLine[-1]))  # y数组增添预测值
    return xArr, yArr  # 返回两个数组


def plotDataset():
    """
    函数说明:绘制数据集
    Parameters:
        无
    Returns:
        无
    """

    xArr, yArr = loadDataset('ex0.txt')
    n = len(xArr)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('Dataset')
    plt.xlabel('x')
    plt.show()


def standRegres(xArr, yArr):
    """
    函数说明:计算回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    """

    xMat = np.mat(xArr)  # 将x数组转化为矩阵
    yMat = np.mat(yArr).T  # 将y数组的转置转化为矩阵
    xTx = xMat.T * xMat  # 计算xTx
    if np.linalg.det(xTx) == 0.0:  # 检测是否可逆
        print("can not do it")
        return
    ws = xTx.I * (xMat.T * yMat)  # 可逆的话进行计算
    return ws  # 返回值


def plotRegression():
    """
    函数说明:绘制回归曲线和数据点
    Parameters:
        无
    Returns:
        无
    """

    xArr, yArr = loadDataset('ex0.txt')  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = np.mat(xArr)  # 创建x矩阵
    yMat = np.mat(yArr)  # 创建y矩阵
    xCopy = xMat.copy()  # 深拷贝x矩阵
    xCopy.sort(0)  # 排序x矩阵
    yHat = xCopy * ws  # 计算根据回归系数得出的预测y矩阵
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:, 1], yHat, c='red')  # 绘制预测直线
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('dataset')
    plt.xlabel('x')
    plt.show()


def lwlr(testpoint, xArr, yArr, k=1.0):
    """
    函数说明:使用局部加权线性回归计算回归系数w
    Parameters:
        testPoint - 测试样本点
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
    """

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testpoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

# xArr, yArr = loadDataset('ex0.txt')
# ws = standRegres(xArr, yArr)
# xMat = np.mat(xArr)
# yMat = np.mat(yArr)
# yHat = xMat * ws
# print(np.corrcoef(yHat.T, yMat))  # 计算yHat和yMat的相关性
