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
from alphafold1 import Residual_Model

# CONSTANTS
batch_size = 4
r = 64
c_m = 128
c_z = 64
c = 8
s = 8

stride = 32
progress_bar = True

def main():
	# get device
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create and load alphafold1 model
	af1_model = Residual_Model(4).to(device)
	af1_model = af1_model.load_state_dict(torch.load('checkpoints/best_alphafold1.pth')['state_dict'])

	# create and load evoformer
	evo_model = Evoformer(batch_size, c_m, c_z, c, device = device).to(device)
	evo_model = evo_model.load_state_dict(torch.load('checkpoints/best_evoformer.pth')['state_dict'])

	# create test dataset that batches by sequence
	test_dataset = Evo_Dataset('test', stride, r, s, c_m, c_z, progress_bar, False, by_seq = True)
	test_loader = DataLoader(dataset = test_dataset, batch_size = 1, drop_last = True)

	with torch.no_grad():
		# each batch from the test_loader will contain crops from the same sequence
		# these crops do not have a randomized starting position
		for batch_idx, (prw_crops, msa_crops, dmats, dmat_masks) in enumerate(tqdm(test_loader, disable = True)):
			# send batch to device and squeeze the batch dim (always 1)
			prw_crops, msa_crops, dmats, dmat_masks = prw_crops.to(device), msa_crops.to(device), dmats.to(device), dmat_masks.to(device)
			prw_crops, msa_crops, dmats, dmat_masks = prw_crops.squeeze(), msa_crops.squeeze(), dmats.squeeze(), dmat_masks.squeeze()
			print(f'{prw_crops.shape = }')
			print(f'{msa_crops.shape = }')
			print(f'{dmats.shape = }')
			print(f'{dmat_masks.shape = }')
			sys.exit(0)

if __name__ == '__main__':
	main()