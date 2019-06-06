import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量

"""


def img2vector(filename):
    # 创建1 * 1024的0向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按照每行读取
    for i in range(32):
        linestr = fr.readline()
        # 每一行的前32个元素按顺序添加到returnVect中
        for j in range(32):
            # reutrnVect(行数, 列数)
            returnVect[0, 32 * i + j] = int(linestr[j])
        return returnVect


"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无

"""


def handWritingClassTest():
    # 测试集的labels
    hwlabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的名字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwlabels.append(classNumber)
        # 将每一个文件的1 * 1024数据存储到trainingMat中
        trainingMat[i, :] = img2vector(f'trainingDigits/{fileNameStr}')
    # 构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型, trainingMat为训练矩阵 hwlabels为对应的标签
    neigh.fit(trainingMat, hwlabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1024向量用来训练
        vectorUnderTest = img2vector(f'testDigits/{fileNameStr}')
        # 获得预测结果
        classifyResult = neigh.predict(vectorUnderTest)
        print(f'分类结果: {classifyResult} 真实结果: {classNumber}')
        if classNumber != classifyResult:
            errorCount += 1
    print(f'错误率为: {errorCount / float(mTest) * 100}%')


handWritingClassTest()
