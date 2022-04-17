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
from Models import Evoformer

# CONSTANTS
batch_size = 4
r = 256
c_m = 256
c_z = 128
c = 32
s = 16

num_epochs = 100
learning_rate = 0.01
stride = 128
progress_bar = True
save_to_file = True
load_from_file = True

def main():
	# get device
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# # create datasets and dataloaders
	# train_dataset = Evo_Dataset('train', stride)
	# train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last = True)

	# valid_dataset = Evo_Dataset('valid-10', stride)
	# valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, drop_last = True)

	# test_dataset = Evo_Dataset('test', stride)
	# test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, drop_last = True)

	evoformer = Evoformer(batch_size, c_m, c_z, c)
	evoformer.train()

	msa_rep = torch.rand((batch_size, s, r, c_m))
	prw_rep = torch.rand((batch_size, r, r, c_z))

	a, b = evoformer(prw_rep, msa_rep)
	print(f'{a.shape = }')
	print(f'{b.shape = }')

	# optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	# loss_func = nn.CrossEntropyLoss(reduction = 'none')

	# for epoch in range(num_epochs):
	# 	model.train()
	# 	sum_loss = 0
	# 	for t_batch_idx, (pwr_crops, msa_crops, dmats, dmat_masks) in enumerate(tqdm(train_loader, disable = True)):
	# 		# send batch to device
	# 		pwr_crops, msa_crops, dmats, dmat_masks = pwr_crops.to(device), msa_crops.to(device), dmats.to(device), dmat_masks.to(device)
	# 		optimizer.zero_grad()



if __name__ == '__main__':
	main()