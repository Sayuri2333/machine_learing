import math
import random

class GA(): # 遗传算法
    def __init__(self, length, count):
        # 染色体长度
        self.length = length
        # 种群中染色体的数量
        self.count = count
        # 随机生成初始种群
        self.population = self.gen_population(length, count)

    def evolve(self, retain_rate=.2, random_select_rate=.5, mutation_rate=.01):
        """
        进化
        对当前一代种群依次进行选择、交叉并生成新一代种群，然后对新一代种群进行变异
        """
        # 选择当前一代种群中的父母个体
        parents = self.selection(retain_rate, random_select_rate)
        # 父母个体之间随机交叉
        self.crossover(parents)
        # 产生变异
        self.mutation(mutation_rate)

    def gen_chromosome(self, length):
        """
        随机生成长度为length的染色体，每个基因的取值是0或1
        这里用一个bit表示一个基因
        """
        # 初始化染色体的值
        chromosome = 0
        # 随机产生染色体
        for i in range(length):
            chromosome |= (1 << i) * random.randint(0, 1)
        return chromosome

    def gen_population(self, length, count):
        """
        获取初始种群（一个含有count个长度为length的染色体的列表）
        """
        return [self.gen_chromosome(length) for i in range(count)]

    def fitness(self, chromosome):
        """
        计算适应度，将染色体解码为0~9之间数字，代入函数计算
        因为是求最大值，所以数值越大，适应度越高
        """
        # 解码染色体
        x = self.decode(chromosome)
        # 直接将带入计算得到的解作为返回值
        return x + 10 * math.sin(5 * x) + 7 * math.cos(4 * x)

    def selection(self, retain_rate, random_select_rate):
        """
        选择
        先对适应度从大到小排序，选出存活的染色体
        再进行随机选择，选出适应度虽然小，但是幸存下来的个体
        """
        # graded是a list of tuple, 元组由适应度和染色体的值构成
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        # 对整个数组进行降序排序, 并直接取第二个位置的值
        graded = [x[1] for x in sorted(graded, reverse=True)]
        # 根据留存率计算留存下来的个体的量
        retain_length = int(len(graded) * retain_rate)
        # 根据留存量截取具有较高的适应度的个体作为这一代的留存个体
        parents = graded[:retain_length]
        # 对于适应度不够高的个体, 根据随机选择率随机的将它们留存
        for chromosome in graded[retain_length:]:
            if random.random() < random_select_rate:
                parents.append(chromosome)
        # 返回这一代淘汰后的留存个体
        return parents

    def crossover(self, parents):
        """
        染色体的交叉、繁殖，生成新一代的种群
        """
        # 初始化孩子种群
        children = []
        # 计算需要繁殖的孩子的量
        target_count = len(self.population) - len(parents)
        # 根据需要的量进行繁殖
        while len(children) < target_count:
            # 随机选取孩子的父母
            male_index = random.randint(0, len(parents) - 1)
            female_index = random.randint(0, len(parents) - 1)
            if male_index != female_index:
                # 随机选取交叉点(染色体进行互换的点)
                cross_pos = random.randint(0, self.length)
                # 生成掩码, 方便位操作
                mask = 0
                # 进行染色体的交叉
                for i in range(cross_pos):
                    mask |= (1 << i)
                    male = parents[male_index]
                    female = parents[female_index]
                    child = ((male & mask) | (female & ~mask)) & ((1 << self.length) - 1)
                    # 将孩子添加到孩子种群中
                    children.append(child)
        # 将孩子种群添加到整个种群中
        self.population = parents + children

    def mutation(self, rate):
        """
        变异
        对种群中的所有个体，随机改变某个个体中的某个基因
        """
        for i in range(len(self.population)):
            # 根据变异率随机改变种群中的个体
            if random.random() < rate:
                # 随机选取需要改变的位置并将这个位置的值取反
                j = random.randint(0, self.length - 1)
                self.population[i] ^= 1 << j

    def decode(self, chromosome):
        """
        解码染色体，将二进制转化为属于[0, 9]的实数
        """
        return chromosome * 9.0 / (2 ** self.length - 1)

    def result(self):
        """
        获得当前代的最优值，这里取的是函数取最大值时x的值
        """
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [x[1] for x in sorted(graded, reverse=True)]
        return self.decode(graded[0])

ga = GA(17, 300)
for x in range(200):
    ga.evolve()
print(ga.result())