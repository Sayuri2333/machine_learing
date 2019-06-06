import numpy as np
f = np.array([[0,0,0,0,1],[1,0,1,0,0],[0,0,0,0,1],[1,0,1,0,0],[0,1,0,1,0]])
result1 = np.dot(f, f)
print(result1)
result2 = np.dot(result1, f)
print(result2)
result3 = np.dot(result2, f)
print(result3)
result4 = np.dot(result3, f)
print(result4)
result = result1 + result2 + result3 + result4 + f
print(result)