import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import numpy as np
import sys
from tqdm import tqdm
from einops import rearrange

# CONSTANTS
batch_size = 1
learning_rate = 0.01
num_epochs = 100
stride = 128
m = 256 # crop size
progress_bar = True
save_to_file = True
load_from_file = True

class Evo_Dataset(IterableDataset):
	def __init__(self):
		super().__init__()
		pass

	def __iter__(self):
		pass

class Evoformer_Model(nn.Module):
	def __init__(self):
		super().__init__()
		pass
	
	def forward(self, x):
		pass

def main():
	pass

if __name__ == '__main__':
	main()