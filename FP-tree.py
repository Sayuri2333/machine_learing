def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):  # 创建指针列表
    retDict = {}  # 初始化指针列表
    for trans in dataSet:  # 遍历数据集中的项
        fset = frozenset(trans)  # 把项中的元素提取并转换为forzenset
        retDict.setdefault(fset, 0)  # 指针列表以字典形式存储元素以及对应的出现次数
        retDict[fset] += 1
    return retDict


class treeNode:  # 定义树节点类
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 名字(元素名)
        self.count = numOccur  # 出现次数
        self.nodeLink = None  # 同元素指针(指向下一个相同元素)
        self.parent = parentNode  # 父节点
        self.children = {}  # 子节点(以字典形式存储)

    def inc(self, numOccur):  # increase,增加出现次数
        self.count += numOccur

    # def disp(self, ind=1):  # ind=>缩进
    #     print('   ' * ind, self.name, ' ', self.count)
    #     for child in self.children.values():
    #         child.disp(ind + 1)  # 递归调用,对于子节点缩进增加一个位置


def createTree(dataSet, minSup=1):  # dataSet实际上为字典,有形如frozenset({'j', 'p', 'z', 'r', 'h'}): 1的键值对
    headerTable = {}
    # 此一次遍历数据集， 记录每个数据项的支持度(这里的支持度是以出现次数为单位的)
    for trans in dataSet:  # 默认取出键(frozenset)进行迭代
        for item in trans:  # 对于frozenset中的每一个元素
            headerTable[item] = headerTable.get(item, 0) + 1  # headerTable统计所有元素的出现次数并做成字典如'j': 1, 'p': 2
    # 根据最小支持度过滤
    # filter(函数, 序列) 将序列中的每个元素带入函数,根据返回的true或false判断是否丢弃
    lessThanMinsup = list(filter(lambda k: headerTable[k] < minSup, headerTable.keys()))
    for k in lessThanMinsup:
        del(headerTable[k])

    freqItemSet = set(headerTable.keys())  # 生成一项的频繁项集
    # 如果所有数据都不满足最小支持度，返回None, None
    if len(freqItemSet) == 0:
        return None, None

    for k in headerTable:  # 字典值的扩充,由headerTable[k] => headerTable[k], None
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('φ', 1, None)  # 构建根节点
    # 第二次遍历数据集，构建fp-tree
    for tranSet, count in dataSet.items():  # transet=>frozenset, count=>项的出现次数
        # 根据最小支持度处理一条训练样本，key:样本中的一个样例，value:该样例的的全局支持度
        localD = {}  # localD为存储一条训练样本中包含在一项频繁集中的(元素: 支持度)的字典
        for item in tranSet:  # 对于项中的每一个元素
            if item in freqItemSet:  # 如果元素在一项的频繁项集中
                localD[item] = headerTable[item][0]  # 将元素的支持度作为value添加进localD中
        if len(localD) > 0:  # 如果localD中有东西(这个样本中有频繁元素)
            # 根据全局频繁项对每个事务中的数据进行排序,等价于 order by p[1] desc, p[0] desc
            # 在Python中可以使用sorted函数对list进行排序，但是如果排序的对象是a list of tuples，sorted函数会使用tuple的第一个元素
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: (p[1], p[0]), reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)  # count是项的出现次数,没有重复的项的话应该为1
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    # items是已经排好序的list of items in transaction
    if items[0] in inTree.children:  # 判断是更新计数还是插入
        inTree.children[items[0]].inc(count)  # 如果第一个元素已经出现过,则增加其计数(tree以字典存储孩子节点)
    else:  # 没有出现过,把items[0]作为节点添加
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:  # 需要插入时,判断同元素指针的连接情况
            headerTable[items[0]][1] = inTree.children[items[0]]  # 设置位于指针列表的同元素指针
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:  # call updateTree() with remaining ordered items
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)  # 对items列表递归调用


def updateHeader(nodeToTest, targetNode):  # 判断targetNode应该接在哪个节点的nodeLink上
    while (nodeToTest.nodeLink != None):  # 如果当前节点的同元素指针不为空
        nodeToTest = nodeToTest.nodeLink  # 移动至下一个同元素
    nodeToTest.nodeLink = targetNode  # 插入到最后一个同元素的nodeLink上

# simpDat = loadSimpDat()
# dictDat = createInitSet(simpDat)
# myFPTree, myheader = createTree(dictDat, 3)
# myFPTree.disp()


def ascendTree(leafNode, prefixPath):  # 迭代从下部回溯整棵树
    if leafNode.parent != None:  # 有父节点
        prefixPath.append(leafNode.name)  # 将节点添加进路径
        ascendTree(leafNode.parent, prefixPath)  # 递归调用将整个路径上的节点添加进去


def findPrefixPath(basePat, headTable):  # 找到某个节点的条件模式基(1或多条路径组成)
    condPats = {}  # 使用字典存储路径以及路径计数
    treeNode = headTable[basePat][1]  # 根据headTable的同元素指针找到下一个同元素
    while treeNode != None:  # 不是空的话
        prefixPath = []  # 初始化路径
        ascendTree(treeNode, prefixPath)  # 获得路径
        if len(prefixPath) > 1:  # 如果路径长度大于1(路径前缀不为空)
            condPats[frozenset(prefixPath[1:])] = treeNode.count  # 将(路径, 计数)存储至condPats(计数是本节点的计数)
        treeNode = treeNode.nodeLink  # 移动到下一个同元素节点
    return condPats


def mineTree(inTree, headerTable, minSup=1, preFix=set([]), freqItemList=[]):
    # order by minSup asc, value asc
    # 按照支持度升序排列的元素列表(出现在headerTable中的)
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: (p[1][0], p[0]))]
    for basePat in bigL:  # 对于支持度升序的元素列表中的每一个元素
        newFreqSet = preFix.copy()  # 由前缀初始化新频繁项
        newFreqSet.add(basePat)  # 将当前元素添加进新频繁项
        freqItemList.append(newFreqSet)  # 把新频繁项添加进频繁项集
        # 通过条件模式基找到的频繁项集
        condPattBases = findPrefixPath(basePat, headerTable)  # 找到此节点的条件模式基
        myCondTree, myHead = createTree(condPattBases, minSup)  # 根据条件模式基生成新树和新的指针列表
        if myHead != None:  # 如果新树不为空
            # print('condPattBases: ', basePat, condPattBases)  # 输出条件模式基
            myCondTree.disp()  # 树可视化
            # print('*' * 30)

            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


# simpDat = loadSimpDat()
# dictDat = createInitSet(simpDat)
# myFPTree, myheader = createTree(dictDat, 3)
# myFPTree.disp()
# condPats = findPrefixPath('z', myheader)
# print('z', condPats)
# condPats = findPrefixPath('x', myheader)
# print('x', condPats)
# condPats = findPrefixPath('y', myheader)
# print('y', condPats)
# condPats = findPrefixPath('t', myheader)
# print('t', condPats)
# condPats = findPrefixPath('s', myheader)
# print('s', condPats)
# condPats = findPrefixPath('r', myheader)
# print('r', condPats)

# frq = []
# mineTree(myFPTree, myheader, 2, freqItemList=frq)
# print(frq)