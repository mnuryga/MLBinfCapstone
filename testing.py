import numpy as np

b1 = np.zeros(40,)
print(f'{b1.shape = }')
b1 = np.tile(b1, (5, 1)).T
print(f'{b1.shape = }')

'''
out:
b1.shape = (40,)
b1.shape = (40, 5)
'''