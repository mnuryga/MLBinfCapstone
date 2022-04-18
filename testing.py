import numba
from numba import jit
import numpy as np
from tqdm import tqdm
from einops import rearrange, reduce, repeat
import sys

@jit(nopython=True)
def funct(x):
	for i in range(len(x)):
		for j in range(len(x)):
			for k in range(len(x)):
				i+j+k

def main():
	x = np.zeros((10, 20, 30))
	y = np.zeros((*x.shape))
	print(f'{y.shape = }')

if __name__ == '__main__':
	main()