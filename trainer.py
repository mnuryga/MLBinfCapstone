'''
Training loop for the AlphaFold 2 model.

Author: Matthew Uryga
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import sys
from tqdm import tqdm
from einops import rearrange

from datasets import Evo_Dataset
from models import Alphafold2_Model
import os

# CONSTANTS
num_gpu = 1
batch_size = 4 * num_gpu
batch_size_gpu = batch_size // num_gpu
batch_size_valid = 4
r = 64
c_m = 128
c_z = 64
c = 16
s = 8

stride = 64
num_epochs = 3
learning_rate = 0.005
progress_bar = True
save_to_file = False
load_from_file = False
USE_DEBUG_DATA = True
save_dir = './debug'


def main():

	# get device
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create datasets and dataloaders
	train_dataset = Evo_Dataset('train', stride, batch_size, r, progress_bar, USE_DEBUG_DATA)
	train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last = True)

	valid_dataset = Evo_Dataset('valid-10', stride, batch_size, r, progress_bar, USE_DEBUG_DATA)
	valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size_valid, drop_last = True)

	model = nn.DataParallel(Alphafold2_Model(s, c_m, c_z, c)).to(device)
	model.train()

	# load state_dict from file if specified
	if load_from_file:
		model.load_state_dict(torch.load(f'{save_dir}/best.pth')['state_dict'])

	# initialize optimizer
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)

	# prev_loss is used to store validation losses -> training is stopped 
	# once validation loss is above a 5-epoch rolling mean
	prev_loss = []

	# TRAINING
	with torch.autograd.set_detect_anomaly(False):
		for epoch in range(num_epochs):
			model.train()
			sum_loss = 0
			for t_batch_idx, (seqs, evos, masks, angs, coords, bb_rs, bb_ts) in enumerate(tqdm(train_loader, disable = True)):
				# send batch to device
				seqs, evos, masks, angs, coords, bb_rs, bb_ts = seqs.to(device), evos.to(device), masks.to(device), angs.to(device), coords.to(device), bb_rs.to(device), bb_ts.to(device)
				optimizer.zero_grad()

				# run foward pass
				pred_coords, L_fape, L_aux = model(seqs, evos, angs, (bb_rs, bb_ts), coords, masks)

				# sum aux and fape as specified in paper
				loss = 0.5*L_fape + 0.5*L_aux

				# run backward pass and sum current loss
				loss.backward()
				sum_loss += loss.item()

				# check if loss is nan
				if torch.isnan(loss).any().item():
					print('loss is nan')
					sys.exit(1)
				
				# step optimizer
				optimizer.step()

				# save to file if specified
				if t_batch_idx % 1 == 0 and save_to_file:
					checkpoint = {
						'epoch': epoch,
						'loss': sum_loss,
						'state_dict': evoformer.state_dict(),
						'optimizer': optimizer.state_dict(),
					}
					if not os.path.exists(save_dir):
						os.makedirs(save_dir)
					torch.save(checkpoint, f'{save_dir}/best_{epoch}.pth')

			# load model for validation
			model_valid = nn.DataParallel(Alphafold2_Model(s, c_m, c_z, c), device_ids=[0]).to(device)
			model_valid.load_state_dict(torch.load(f'{save_dir}/best_{epoch}.pth')['state_dict'])
			model_valid.eval()

			# VALIDATION
			valid_loss = 0
			with torch.no_grad():
				for t_batch_idx, (seqs, evos, masks, angs, coords, bb_rs, bb_ts) in enumerate(tqdm(valid_loader, disable = True)):
					# send batch to device
					seqs, evos, masks, angs, coords, bb_rs, bb_ts = seqs.to(device), evos.to(device), masks.to(device), angs.to(device), coords.to(device), bb_rs.to(device), bb_ts.to(device)

					# run forward pass
					pred_coords, L_fape, L_aux = model_valid(seqs, evos, angs, (bb_rs, bb_ts), coords)

					# calculate loss
					loss = 0.5*L_fape + 0.5*L_aux
					valid_loss += loss.item()

			# append current loss to prev_loss list
			prev_loss.append(valid_loss)

			# print out epoch stats
			print(f'Epoch {epoch:02d}, {t_batch_idx*batch_size:07,d} crops:')
			print(f'\tTrain loss per batch = {sum_loss/t_batch_idx/batch_size:.6f}')
			print(f'\tValid loss per batch = {valid_loss/v_batch_idx/batch_size_valid:.6f}')

			# # if valid_loss exceedes the 5-epoch rolling sum, break from training
			# if valid_loss > np.mean(prev_loss[-5:]):
			# 	break


if __name__ == '__main__':
	main()