from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 加载文件
with open('lenses.txt', 'r') as fr:
    # 处理文件每一行 将每行数据存储到一个数组里
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# 提取每一组的类别(眼镜类型) 保存在列表里
lenses_target = []

for each in lenses:
    lenses_target.append(each[-1])
# 特征标签
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# 保存lenses某一列数据的临时列表
lenses_list = []
# 保存lenses数据的字典
lenses_dict = {}
# 提取信息并生成字典
for each_label in lensesLabels:
    # 对于lenses中的每一行 提取相应位置的数据到lenses_list中
    for each in lenses:
        lenses_list.append(each[lensesLabels.index(each_label)])
    lenses_dict[each_label] = lenses_list
    lenses_list = []
print(lenses_dict)
# 根据字典生成dataFrame 字典的key成为column值 values成为数据内容
lenses_dataframe = pd.DataFrame(lenses_dict)
print(lenses_dataframe)

# 创建LabelEncoder对象 用来序列化(给每个字符串编码 将字符串转为数值用于sklearn)
le = LabelEncoder()
# .columns方法返回dataframe的column数组
for col in lenses_dataframe.columns:
    lenses_dataframe[col] = le.fit_transform(lenses_dataframe[col])
print(lenses_dataframe)

# 创建决策树对象
clf = tree.DecisionTreeClassifier()
# 拟合数据以及分类目标
clf = clf.fit(lenses_dataframe.values.tolist(), lenses_target)
print(clf.predict([[1, 1, 1, 0]]))