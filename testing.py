import numba
from numba import jit
import numpy as np
from tqdm import tqdm
import sys

@jit(nopython=True, parallel = True)
def funct(x):
	for i in range(len(x)):
		for j in range(len(x)):
			for k in range(len(x)):
				i+j+k

def main():
	x = np.arange(int(sys.argv[1]))
	funct(x)

if __name__ == '__main__':
	main()