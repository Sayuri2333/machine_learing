import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import operator
"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量

"""

def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件的所有内容
    arrayLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayLines)
    # 新建空的numpy矩阵 有numberOfLines行, 3列
    returnMat = np.zeros((numberOfLines, 3))
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    for line in arrayLines:
        # string.strip() 删除字符串左右的空白字符
        line = line.strip()
        # s.split(str='')方法根据str进行字符串分片
        listFromlines = line.split('\t')
        # 将数据的前3列提取出来存放到numpy矩阵中作为数据集
        returnMat[index, :] = listFromlines[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromlines[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromlines[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromlines[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


"""
函数说明:可视化数据

Parameters:
    datingDataMat - 特征矩阵
    datingLabels - 分类Label
Returns:
    无

"""

def showdatas(datingDtaMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r'C:\Users\HASEE\Desktop\machinelearninginaction\simsun.ttc', size=14)
    # 把fig画布分割成一行一列 不共享x轴和y轴 fig画布的大小为(13, 8)
    # 当nrow=2,nclos=2时, 代表fig画布被分割为4个区域 axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    labelsColors = []
    for i in datingLabels:
        if i == 1:
            labelsColors.append('black')
        elif i == 2:
            labelsColors.append('orange')
        elif i == 3:
            labelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDtaMat[:, 0], y=datingDtaMat[:, 1], color=labelsColors, s=15, alpha=.5)
    # 设置标题和x轴label y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDtaMat[:, 0], y=datingDtaMat[:, 2], color=labelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDtaMat[:, 1], y=datingDtaMat[:, 2], color=labelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值


"""


def autoNorm(dataset):
    # 获得数据的最小值
    # numpy.min()方法返回最小值 无参数返回矩阵最小值 0返回列最小值 1返回行最小值
    minVal = dataset.min(0)
    maxVal = dataset.max(0)
    # 获得最大值和最小值的范围
    ranges = maxVal - minVal
    # 新建矩阵 shape(dataset)返回dataset的行列数
    normDataMat = np.zeros(np.shape(dataset))
    # 返回dataset的行数 用来建立最小值构成的矩阵
    m = dataset.shape[0]
    # 原始值减去最小值
    normDataMat = dataset - np.tile(minVal, (m, 1))
    # 除以范围
    normDataMat = normDataMat / np.tile(ranges, (m, 1))
    return normDataMat, ranges, minVal


"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果

"""


def classfiy0(inX, dataset, labels, k):
    datasetSize = dataset.shape[0]
    inXMat = np.tile(inX, (datasetSize, 1))
    diffMat = dataset - inXMat
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistanceIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabels = labels[sortedDistanceIndices[i]]
        classCount[voteIlabels] = classCount.get(voteIlabels, 0) + 1
        # 统计完成后对字典进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
函数说明:分类器测试函数

Parameters:
    无
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

"""


def datingClassTest():
    # 打开的文件名
    filename = 'peibangdatingdata.txt'
    # 处理文件导出数据集矩阵以及标签
    datingDataMat, datinglabels = file2matrix(filename)
    # 取得所有数据的10%
    hoRatio = 0.10
    # 数据归一化, 返回归一化的矩阵, 数据范围和最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        # 注意对应的标签也需要切分块
        classifierResult = classfiy0(normMat[i, :], normMat[numTestVecs:m, :], datinglabels[numTestVecs:m], 4)
        print(f'分类结果: {classifierResult} 真实类别: {datinglabels[i]}')
        if classifierResult != datinglabels[i]:
            errorCount += 1
    print(f'错误率: {errorCount / float(numTestVecs) * 100}%')


"""
函数说明:通过输入一个人的三维特征,进行分类输出

Parameters:
    无
Returns:
    无

"""


def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input('玩视频游戏所耗的时间的百分比: '))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input('每周消耗的冰激凌的公升数: '))
    filename = 'peibangdatingdata.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minvals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, precentTats, iceCream])
    normArr = (inArr - minvals) / ranges
    classifyResult = classfiy0(normArr, normMat, datingLabels, 4)
    print(f'你可能{resultList[classifyResult - 1]}这个人')


classifyPerson()