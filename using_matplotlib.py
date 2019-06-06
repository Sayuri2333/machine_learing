import matplotlib.pyplot as plt
# basic using plot
plt.plot([1, 2, 3], [5, 7, 4])
# draw two lines
# line 1
x = [1, 2, 3]
y = [5, 7, 4]
# line 2
x2 = [1, 2, 3]
y2 = [10, 14, 12]
# draw line with label
plt.plot(x, y, label='First Line')
plt.plot(x2, y2, label='Second Line')
# add label in axis
plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph\nCheck it out')
plt.legend() # 生成默认图例

# draw bar plot
# plt.bar(barnumber, barheight)
plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")
plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')
plt.title('Epic Graph\nAnother Line! Whoa')

