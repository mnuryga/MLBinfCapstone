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

from evo_dataset import Evo_Dataset
from models import Evo_Model
import os

# CONSTANTS
num_gpu = 4
batch_size = 64 * num_gpu
batch_size_gpu = batch_size // num_gpu
batch_size_valid = 64
r = 64
c_m = 128
c_z = 64
c = 8
s = 8

stride = 64
num_epochs = 6
learning_rate = 0.001
progress_bar = True
save_to_file = True
load_from_file = False
USE_DEBUG_DATA = False
save_dir = './ln_6e_fixed'

def main():
	# get device
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create datasets and dataloaders
	train_dataset = Evo_Dataset('train', stride, batch_size, r, progress_bar, USE_DEBUG_DATA)
	train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last = True)

	valid_dataset = Evo_Dataset('valid-10', stride, batch_size, r, progress_bar, USE_DEBUG_DATA)
	valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size_valid, drop_last = True)

	evoformer = nn.DataParallel(Evo_Model(r, s, c_m, c_z, c)).to(device)
	evoformer.train()

	# load state_dict from file if specified
	if load_from_file:
		evoformer.load_state_dict(torch.load('checkpoints/best.pth')['state_dict'])

	# initialize optimizer and loss function
	optimizer = optim.Adam(evoformer.parameters(), lr = learning_rate)
	loss_func = nn.CrossEntropyLoss(reduction = 'none')

	# prev_loss is used to store validation losses -> training is stopped 
	# once validation loss is above a 5-epoch rolling mean
	prev_loss = []

	# TRAINING
	for epoch in range(num_epochs):
		evoformer.train()
		sum_loss = 0
		for t_batch_idx, (seqs, evos, dmat, dmat_mask, angs) in enumerate(tqdm(train_loader, disable = True)):
			# send batch to device
			seqs, evos, dmat, dmat_mask, angs = seqs.to(device), evos.to(device), dmat.to(device), dmat_mask.to(device), angs.to(device)
			optimizer.zero_grad()
			# run forward pass and cross entropy loss - reduction is none, so
			# loss output is a batch*crop_size*crop_size tensor
			pred_dmat, pred_angs = evoformer(seqs, evos)
			dmat_loss = loss_func(pred_dmat, dmat.long()).mul(dmat_mask.long())
			angs_loss = loss_func(pred_angs, angs.long())

			# multiply loss output element-wise by mask and take the mean
			# loss = loss.mul(dmat_mask)
			loss = torch.mean(dmat_loss)+torch.mean(angs_loss)

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
				torch.save(checkpoint, f'{save_dir}/best.pth')
                
        # load model for validation
		evoformer_valid = nn.DataParallel(Evo_Model(r, s, c_m, c_z, c), device_ids=[0]).to(device)
		evoformer_valid.load_state_dict(torch.load(f'{save_dir}/best.pth')['state_dict'])
		evoformer_valid.eval()

		# VALIDATION
		valid_loss = 0
		with torch.no_grad():
			for v_batch_idx, (seqs, evos, dmat, dmat_mask, angs) in enumerate(tqdm(valid_loader, disable = True)):
				# send batch to device
				seqs, evos, dmat, dmat_mask, angs = seqs.to(device), evos.to(device), dmat.to(device), dmat_mask.to(device), angs.to(device)

				# run forward pass and cross entropy loss - reduction is none, so
				# loss output is a batch*crop_size*crop_size tensor
				pred_dmat, pred_angs = evoformer_valid(seqs, evos)
				dmat_loss = loss_func(pred_dmat, dmat.long()).mul(dmat_mask.long())
				angs_loss = loss_func(pred_angs, angs.long())
				# multiply loss output element-wise by mask and take the mean
				# loss = loss.mul(dmat_mask)
				loss = torch.mean(dmat_loss)+torch.mean(angs_loss)
				valid_loss += loss.item()

		# append current loss to prev_loss list
		prev_loss.append(valid_loss)

		# print out epoch stats
		print(f'Epoch {epoch:02d}, {t_batch_idx*batch_size:07,d} crops:')
		print(f'\tTrain loss per batch = {sum_loss/t_batch_idx/batch_size:.6f}')
		print(f'\tValid loss per batch = {valid_loss/v_batch_idx/batch_size_valid:.6f}')

		# if valid_loss exceedes the 5-epoch rolling sum, break from training
# 		if valid_loss > np.mean(prev_loss[-5:]):
# 			break


if __name__ == '__main__':
    main()