from numpy import *

# 创建数组
a = arange(15).reshape(3, 5)
b = array([2, 3, 4])
c = array([(1.5, 2, 3), (4, 5, 6)])
d = zeros((2, 3))
e = ones((3, 4))
f = arange(0, 15, 1).reshape(5, 3)

# 打印数组
print(a)
print(b)
print(f)

# 基本运算 numpy中的运算时按照元素的
a = array([20, 30, 40, 50])
b = arange(4)
c = a - b
print(c)
d = b ** 2
print(d)
e = a < 35  # 对于其中的每一个元素作比较
print(e)

# 数组的一些基本性质
print(a.dtype)  # data type
print(a.shape)  # shape of the array
print(a.size)  # number of the element
print(a.ndim)  # ndarray dimension

# numpy中的通用函数 sin() cos() exp() sqrt() add()等等
B = arange(3)
C = arange(3)
print(exp(B))
print(sqrt(B))
print(add(B, C))

# 索引切片和迭代

a = arange(10) ** 3
print(a[2])
print(a[2: 5])
print(a[0: 3: 1])
a[0: 3: 1] = 1000  # 使用数组切片进行迭代赋值
print(a[0: 3: 1])


def f(x, y):
    return 10 * x + y


b = fromfunction(f, (5, 4), dtype=int)
# fromfunction函数根据函数创建数组 (5, 4)元组指定传入的坐标数组(element值分别等于行数和列数的数组共两个)
print(b)
print(b[0: 5, 1])  # 使用逗号进行分割行列下标
print(b[-1])  # same as print(b[-1, :])
# 当少于轴数的索引被提供时，缺失的索引被认为是整个切片

# 多维数组时使用迭代器是针对第一个轴而言的
for row in b:
    print(row)

# 使用flat属性针对数组中的每一个元素进行运算(.flat产生一个针对元素的迭代器)
for element in b.flat:
    print(element)

# 数组的形状以及改变数组的形状
a = floor(10 * random.random((3, 4)))  # floor()方法针对每个元素取其整数值(舍去小数部分)
print(a.shape)

print(a.ravel())  # .ravel()方法将数组转换为行向量 它返回的是视图 会对原矩阵产生影响
# .flatten()方法也可以执行同样的操作 不过返回的是原矩阵的拷贝

print(a.reshape(2, 6))  # .reshape()方法有返回值 不会对原始数组进行修改
# .resize()方法没有返回值 会对原始数组进行修改

a = floor(10 * random.random((3, 4)))
b = floor(10 * random.random((3, 4)))
print(a)
print(b)
# 根据ab生成增广矩阵
print(vstack((a, b)))  # 沿着第一个轴组合
print(hstack((a, b)))  # 沿着第二个轴组合
# 根据自身增加维度
a = array([4., 2.])
print(a[:, newaxis])
# 分割数组
a = floor(10 * random.random((2, 12)))
print(a)
print(hsplit(a, 3))  # 沿着第一个轴进行切割 切割成3块
print(hsplit(a, (3, 4)))  # 指定沿着哪一个轴向分割
# vsplit方法支持沿着纵向的轴进行分割
# 数组的复制
# 不复制
print('-----不复制-----')
a = arange(12)
b = a
print(b is a) # 判断a与b是否是同一个对象
b.shape = 3, 4  # 修改b数组的形状
print(a.shape)  # 也会影响a数组的形状

# 浅拷贝 共用数据源 但是不共用数组
a = arange(12)
c = a.view()
print(c is a)
print(c.base is a)
c.shape = 3, 4
print(a.shape)
c[0, 3] = 1234  # 修改c数组数据源的值
print(a)  # 也会改变a数组数据源的值

# 深拷贝 不共用数据源和数组
d = a.copy()

# 线性代数
a = array([[1.0, 2.0], [3.0, 4.0]])
print(a)
print(a.transpose())

# 矩阵求你
print(linalg.inv(a))

# 矩阵求行列式的值
print(linalg.det(a))

# 矩阵乘法
print(dot(a, linalg.inv(a)))


# 使用numpy中的矩阵艾完成上述操作

A = matrix('1.0, 2.0; 3.0, 4.0')
print(A)

# 求A的转置
print(A.T)

# 求矩阵乘法
print(A * A.T)

# 求A的逆矩阵
print(A.I)

# 解决方程组问题
Y = matrix('5.0, 7.0')
print(Y)
print(linalg.solve(A, Y.T))

