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
	x = np.arange(int(sys.argv[1]))
	# funct(x)
	x = np.ones(10)
	y = np.ones(10)

	z = rearrange()

if __name__ == '__main__':
	main()