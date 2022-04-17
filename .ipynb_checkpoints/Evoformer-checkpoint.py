import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import sys
from tqdm import tqdm
from einops import rearrange

from Evo_Dataset import Evo_Dataset
from Models import Evoformer_Model

# CONSTANTS
batch_size = 1
learning_rate = 0.01
num_epochs = 100
N_res = 256
stride = 128
N_clust = 16
progress_bar = True
save_to_file = True
load_from_file = True

def main():
	# create datasets and dataloaders
	train_dataset = Evo_Dataset('train', stride)
	train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last = True)

	valid_dataset = Evo_Dataset('valid-10', stride)
	valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, drop_last = True)

	test_dataset = Evo_Dataset('test', stride)
	test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, drop_last = True)

if __name__ == '__main__':
	main()