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
num_gpu = 4
batch_size = 64 * num_gpu
batch_size_gpu = batch_size // num_gpu
r = 64
c_m = 128
c_z = 64
c = 8
s = 8

stride = 64
num_epochs = 100
learning_rate = 0.01
progress_bar = True
save_to_file = True
load_from_file = False
USE_DEBUG_DATA = True

def main():
	# get device
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create datasets and dataloaders
	train_dataset = Evo_Dataset('train', stride, r, s, c_m, c_z, progress_bar, USE_DEBUG_DATA)
	train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last = True)

	valid_dataset = Evo_Dataset('valid-10', stride, r, s, c_m, c_z, progress_bar, USE_DEBUG_DATA)
	valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, drop_last = True)

	evoformer = nn.DataParallel(Evoformer(batch_size_gpu, c_m, c_z, c, device = device)).to(device)
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

# 	# TRAINING
	for epoch in range(num_epochs):
		evoformer.train()
		sum_loss = 0
		for t_batch_idx, (prw_crops, msa_crops, dmats, dmat_masks) in enumerate(tqdm(train_loader, disable = True)):
			# send batch to device
			prw_crops, msa_crops, dmats, dmat_masks = prw_crops.to(device), msa_crops.to(device), dmats.to(device), dmat_masks.to(device)
			optimizer.zero_grad()

			# run forward pass and cross entropy loss - reduction is none, so
			# loss output is a batch*crop_size*crop_size tensor
			prw_crops, msa_crops = evoformer(prw_crops, msa_crops)
			loss = loss_func(rearrange(prw_crops, 'b i j c -> b c i j'), dmats.long())

			# multiply loss output element-wise by mask and take the mean
			loss = loss.mul(dmat_masks)
			loss = torch.mean(loss)

			# run backward pass and sum current loss
			loss.backward(retain_graph=True)
			sum_loss += loss.item()
			
			# step optimizer
			optimizer.step()

			# save to file if specified
			if t_batch_idx % 25 == 0 and save_to_file:
				checkpoint = {
					'epoch': epoch,
					'loss': sum_loss,
					'state_dict': evoformer.state_dict(),
					'optimizer': optimizer.state_dict(),
				}
				torch.save(checkpoint, f'checkpoints/best.pth')

		# VALIDATION
		valid_count = 0
		valid_loss = 0
		evoformer.eval()
		with torch.no_grad():
			print('starting validation')
			for v_batch_idx, (prw_crops, msa_crops, dmats, dmat_masks) in enumerate(tqdm(valid_loader)):
				print('loopy validation')
				# send batch to device
				prw_crops, msa_crops, dmats, dmat_masks = prw_crops.to(device), msa_crops.to(device), dmats.to(device), dmat_masks.to(device)

				# run forward pass and cross entropy loss - reduction is none, so
				# loss output is a batch*crop_size*crop_size tensor
				prw_crops, msa_crops = evoformer(rearrange(prw_crops, 'b i j c -> b c i j'), msa_crops)
				loss = loss_func(prw_crops, dmats.long())

				# multiply loss output element-wise by mask and take the mean
				loss = loss.mul(dmat_masks)
				loss = torch.mean(loss)
				valid_loss += loss.item()
				valid_count += 1
				print('HELLO')
				print(valid_count)

		# append current loss to prev_loss list
		prev_loss.append(valid_loss)

		# print out epoch stats
		print(f'Epoch {epoch:02d}, {t_batch_idx*batch_size:06,d} crops:')
		print(f'\tTrain loss per batch = {sum_loss/t_batch_idx/batch_size:.6f}')
		print(f'valid count is {valid_count}')
		print(f'\tValid loss per batch = {valid_loss/valid_count/batch_size:.6f}')

		# if valid_loss exceedes the 5-epoch rolling sum, break from training
		if valid_loss > np.mean(prev_loss[-5:]):
			break


if __name__ == '__main__':
	main()