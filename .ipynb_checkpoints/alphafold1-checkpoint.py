import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

import sidechainnet as scn
import sys
import numpy as np
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import pandas as pd

# CONSTANTS
crop_size = 64
crop_padding = 64
batch_size = 20
learning_rate = 0.005
num_epochs = 99

threshold = False
plot_predictions = False
progress_bar = True
load_from_file = True
save_to_file = False

# what L/k values are being used for accuracy calculation
seq_ks = [1, 2, 5, 10, 20, 50, 100]

# dilation block as described in Alphafold paper
class Dilation_Block(nn.Module):
	def __init__(self, d):
		super().__init__()
		self.bn1 = nn.BatchNorm2d(128)
		self.proj_down = nn.Conv2d(128, 64, 1)
		self.bn2 = nn.BatchNorm2d(64)
		self.d = nn.Conv2d(64, 64, 3, dilation = d, padding = d)
		self.bn3 = nn.BatchNorm2d(64)
		self.proj_up = nn.Conv2d(64, 128, 1)

	def forward(self, x_in):
		x = self.bn1(x_in)
		x = F.elu(x)
		x = self.proj_down(x)
		x = self.bn2(x)
		x = F.elu(x)
		x = self.d(x)
		x = self.bn3(x)
		x = F.elu(x)
		x = self.proj_up(x)
		return x + x_in # add residuals

# full model that links together many dilation blocks
class Residual_Model(nn.Module):
	def __init__(self, num_blocks):
		super().__init__()
		# project 124 features to 128
		self.proj_in = nn.Conv2d(124, 128, 1)

		# each set of blocks consists of dilation blocks with 1, 2, 4, and 8 dilation
		blocks = []
		for i in range(num_blocks):
			blocks.append(Dilation_Block(1))
			blocks.append(Dilation_Block(2))
			blocks.append(Dilation_Block(4))
			blocks.append(Dilation_Block(8))
		self.dilation_blocks = nn.ModuleList(blocks)
		# project down to 64 features -> needed for output and loss calc
		self.proj_out = nn.Conv2d(128, 64, 1)
	
	def forward(self, x):
		# project to 128 features and then sequentially run through each block
		x = self.proj_in(x)
		for block in self.dilation_blocks:
			x = block(x)

		# project out to 64 features
		x = self.proj_out(x)
		return x

class Crop_Dataset(IterableDataset):
	'''
	IterableDataset used to batch data as crops and send to model
	Outputs one crop and its associated mask, labels, etc at a time
	the Dataloader will batch these together to specified batch_size (currently 20)
	When specified, it will output sequence data as well - this is needed
	for evaluation.  The individual crops do not carry enough data to be
	able to be used in evaluation of sequence accuracy.
	'''
	def __init__(self, key, crop_stride = 64, randomize_start = True, yield_seq_data = False):
		super().__init__()
		# load data from scn
		self.data = scn.load(casp_version = 7, with_pytorch="dataloaders", 
										seq_as_onehot=True, aggregate_model_input=False,
										batch_size=16, num_workers = 0)

		# padding function for use in padding for edge crops
		self.pad = nn.ZeroPad2d((0, crop_padding, 0, crop_padding))

		# key is the subset of data that is being considered
		self.key = key

		# randomize start determines whether or not the initial i and j
		# values will be randomized or not.  When evaluating by sequence
		# we want to start at the very start of the feature matrix
		self.randomize_start = randomize_start
		self.crop_stride = crop_stride

		# flag to determine which data is yielded
		self.yield_seq_data = yield_seq_data

		# dmats and dmat masks are used in get_mat_and_mask() below
		self.dmats = []
		self.dmat_masks = []

	def __iter__(self):
		# seq_id used to indicate when sequence changes
		seq_id = 0

		# loop over each set of sequences of length L
		for batch_idx, batch in enumerate(tqdm(self.data[self.key], disable = not progress_bar)):
			seqs, evos, angs, masks, dmats, dmat_masks = get_seq_features(batch)

			# get dimensions for feature matrix
			L = seqs.shape[1]

			# create positive mask -> all m[i, j] such that j < i are zeroed out
			# because matrix is symmetrical, we do not want to consider anything below
			# the diagonal
			base_mask = torch.zeros((L+crop_padding, L+crop_padding))
			for i in range(L+crop_padding):
				base_mask[i, i:] = 1

			# loop over each sequence
			for seq_idx, (seq, evo, dmat, dmat_mask) in enumerate(zip(seqs, evos, dmats, dmat_masks)):
				# create feature matrix
				M = torch.zeros((L+crop_padding, L+crop_padding, 124))
				# create onehot sequence concat to evo information
				p = torch.zeros((L, 41))
				for i in range(L):
					p[i] = torch.cat((seq[i], evo[i]))
				p = torch.tile(p, (L, 1, 1))
				m = torch.cat((torch.transpose(p, 0, 1), p), dim = -1)

				# calculate difference and dot product between evo information
				diff = torch.abs(m[:, :, 20:41].sub(m[:, :, 61:82]))
				prod = m[:, :, 20:41].mul(m[:, :, 61:82])

				# append to feature matrix
				M[:L, :L, :] = torch.cat((m, diff, prod), dim = 2)

				# discreteize dmat in 64 bins
				dmat = torch.floor(torch.clamp(dmat, 2, 21.6875).sub(2).mul(3.2))
				self.dmats.append(dmat)
				# pad dmat
				dmat = self.pad(dmat)

				# multiply mask by positive mask from above, and pad
				dmat_mask = torch.logical_and(dmat_mask, base_mask[:-crop_padding, :-crop_padding])
				self.dmat_masks.append(dmat_mask)
				dmat_mask = self.pad(dmat_mask)

				# generate starting point
				# if sequence length is < 64, start at 0, 0
				i_start = 0 if L < 64 or not self.randomize_start else np.random.randint(0, 64)
				j_start = 0 if L < 64 or not self.randomize_start else np.random.randint(0, 64)

				# loop over each crop
				for i in range(i_start, L, self.crop_stride):
					for j in range(j_start, L, self.crop_stride):
						# if j < crop_size + i, the whole crop is below the middle of the matrix, continue
						if j + crop_size < i:
							continue
						# create the crop of the feature matrix, c
						c = M[i:i + crop_size, j:j + crop_size]

						# rearrange c such that the features are dim 0 (needed for model)
						crop = rearrange(c, 'i j c -> c i j')

						# crop distance and mask matrices
						distance = dmat[i:i + crop_size, j:j + crop_size]
						mask = dmat_mask[i:i + crop_size, j:j + crop_size]

						# yield appropriate data
						if self.yield_seq_data:
							yield crop, i, j, seq_id, L
						else:
							yield crop, distance, mask

				# next sequence, so increment seq_id
				seq_id += 1

	def get_mat_and_mask(self):
		'''
		Function to fetch dmat and its mask from the dataset.
		When creating crops in batches, sometimes one batch can span
		several sequences of varying length, so a way to get the current
		dmat and mask was needed.  Alternatively, this could have been 
		yielded alongside the crops, but most of the time, the batch would
		consist of many dmat and masks that are the same, which is inefficient.
		'''
		# self.dmats and self.dmat_masks act as queues
		x, y = self.dmats[0], self.dmat_masks[0]
		del self.dmats[0]
		del self.dmat_masks[0]
		return x, y


def get_seq_features(batch):
	'''
	Take a batch of sequence info and return the sequence (one-hot),
	evolutionary info and (phi, psi, omega) angles per position, 
	as well as position mask.
	Also return the distance matrix, and distance mask.
	'''
	str_seqs = batch.str_seqs # seq in str format
	seqs = batch.seqs # seq in one-hot format
	int_seqs = batch.int_seqs # seq in int format
	masks = batch.msks # which positions are valid
	lengths = batch.lengths # seq length
	evos = batch.evos # PSSM / evolutionary info
	angs = batch.angs[:,:,0:2] # torsion angles: phi, psi
	
	# use coords to create distance matrix from c-beta
	# except use c-alpha for G
	# coords[:, 4, :] is c-beta, and coords[:, 1, :] is c-alpha
	coords = batch.crds # seq coord info (all-atom)
	batch_xyz = []
	for i in range(coords.shape[0]):
		xyz = []
		xyz = [coords[i][cpos+4,:] 
				if masks[i][cpos//14] and str_seqs[i][cpos//14] != 'G'
				else coords[i][cpos+1,:]
				for cpos in range(0, coords[i].shape[0]-1, 14)]
		batch_xyz.append(torch.stack(xyz))
	batch_xyz = torch.stack(batch_xyz)
	# now create pairwise distance matrix
	dmats = torch.cdist(batch_xyz, batch_xyz)
	# create matrix mask (0 means i,j invalid)
	dmat_masks = torch.einsum('bi,bj->bij', masks, masks)
	
	return seqs, evos, angs, masks, dmats, dmat_masks

def main():
	# get device
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create datasets and dataloaders
	train_dataset = Crop_Dataset('train')
	train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last = True)

	valid_dataset = Crop_Dataset('valid-10')
	valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, drop_last = True)

	# we do not want to randomize data and we also need the extra sequence data
	# from the dataset
	test_dataset = Crop_Dataset('test', crop_stride = 32, randomize_start = False, yield_seq_data = True)
	test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, drop_last = True)

	# create model with 4 sets of dilation blocks (4*4 = 16 blocks total) and send to device
	model = Residual_Model(4)
	model = model.to(device)

	# initialize optimizer and loss function
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	loss_func = nn.CrossEntropyLoss(reduction = 'none')

	# load state_dict from file if specified
	if load_from_file:
		model.load_state_dict(torch.load('best.pth')['state_dict'])

	# prev_loss is used to store validation losses -> training is stopped 
	# once validation loss is above a 5-epoch rolling mean
	prev_loss = []

	# TRAINING
	for epoch in range(num_epochs):
		model.train()
		sum_loss = 0
		for t_batch_idx, (crops, distances, masks) in enumerate(tqdm(train_loader, disable = True)):
			# send batch to device
			crops, distances, masks = crops.to(device), distances.to(device), masks.to(device)
			optimizer.zero_grad()

			# run forward pass and cross entropy loss - reduction is none, so
			# loss output is a batch*crop_size*crop_size tensor
			preds = model(crops)
			loss = loss_func(preds, distances.long())

			# multiply loss output element-wise by mask and take the mean
			loss = loss.mul(masks)
			loss = torch.mean(loss)

			# run backward pass and sum current loss
			loss.backward()
			sum_loss += loss.item()
		
			# step optimizer
			optimizer.step()

			# save to file if specified
			if t_batch_idx % 100 == 0 and save_to_file:
				checkpoint = {
					'epoch': epoch,
					'loss': sum_loss,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
				}
				torch.save(checkpoint, f'best.pth')

		# VALIDATION
		valid_loss = 0
		model.eval()
		with torch.no_grad():
			for v_batch_idx, (crops, distances, masks) in enumerate(tqdm(train_loader, disable = True)):
				# send batch to device
				crops, distances, masks = crops.to(device), distances.to(device), masks.to(device)

				# run forward pass and cross entropy loss - reduction is none, so
				# loss output is a batch*crop_size*crop_size tensor
				preds = model(crops)
				loss = loss_func(preds, distances.long())

				# multiply loss output element-wise by mask and take the mean
				loss = loss.mul(masks)
				loss = torch.mean(loss)

				# sum loss
				valid_loss += loss.item()

		# append current loss to prev_loss list
		prev_loss.append(valid_loss)

		# print out epoch stats
		print(f'Epoch {epoch:02d}, {t_batch_idx*batch_size:06,d} crops:')
		print(f'\tTrain loss per batch = {sum_loss/t_batch_idx/batch_size:.6f}')
		print(f'\tValid loss per batch = {valid_loss/v_batch_idx/batch_size:.6f}')

		# if valid_loss exceedes the 5-epoch rolling sum, break from training
		if valid_loss > np.mean(prev_loss[-5:]):
			break

	# lists to store accuracy per sequence
	short_accs = []
	med_accs = []
	long_accs = []

	def get_n_largest(arr, n):
		'''
		Function to get the coordinates of the n largest element in arr
		This code was based on the stackoverflow post:
		https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
		'''
		flat = arr.flatten()
		indices = np.argpartition(flat, -n)[-n:]
		indices = indices[np.argsort(0-flat[indices])]
		return np.array(np.unravel_index(indices, arr.shape))
	
	def calc_mask_acc(contact_preds, mask, contact_labels, k, threshold = False):
		'''
		Function to calculate the accuracy of L/k predictions for a given sequence and mask.
		Mask input is either the short, medium, or long mask from the evaluation loop.
		'''
		seq_len = contact_preds.shape[0]

		# multiply the sequence predictions by the mask to zero out any predictions
		# not within the range we are considering
		contact_preds = contact_preds.mul(mask[:len(contact_preds), :len(contact_preds)])
		accs = []
		# for each k, calculate the accuracy of the top L/k predictions
		for i in k:
			num_preds = seq_len//i

			# get the indices of the model's most confident predictions
			largest_indices = get_n_largest(contact_preds.to('cpu').numpy(), num_preds)

			# create binary tensor - 1 is a predicted contact
			cp = torch.zeros_like(contact_preds)
			cp[largest_indices[0], largest_indices[1]] = 1

			# if specified, set all contact predictions that are not > .5 probability to 0
			if threshold:
				cp[contact_preds < 0.5] = 0

			# correct predictions found by doing element wise AND between the 
			# binary prediction and label tensors
			cp_correct = torch.logical_and(cp, contact_labels)

			# true positive count is the sum of the correct predictions
			TP = torch.sum(cp_correct)

			# false positive count is the number of total predictions made
			# get accuracy and append to list
			acc = (TP/min(num_preds, torch.sum(contact_labels).item())).item()
			accs.append(acc)

		# return list of accuracies
		return np.array(accs)

	# EVALUATION
	with torch.no_grad():
		# create masks for short/medium/long range contacts
		short_mask = torch.zeros((600, 600)).to(device)
		med_mask = torch.zeros((600, 600)).to(device)
		long_mask = torch.zeros((600, 600)).to(device)
		for i in range(600):
			short_mask[i, i+6:i+12] = 1
			med_mask[i, i+12:i+24] = 1
			long_mask[i, i+24:] = 1

		sum_loss = 0
		current_id = -1
		for batch_idx, (crops, ii, jj, seq_ids, Ls) in enumerate(tqdm(test_loader, disable = True)):
			# send crops to device
			crops = crops.to(device)

			# if on first sequence, initialize tensors
			if current_id == -1:
				seq_preds = torch.zeros((Ls[0]+crop_padding, Ls[0]+crop_padding, 64)).to(device)
				current_id = seq_ids[0]
				distances, mask = test_loader.dataset.get_mat_and_mask()
				distances, mask = distances.to(device), mask.to(device)

			# run forward pass
			preds = model(crops)
		
			# rearrange predictions such that the features are the last dimension
			preds = rearrange(preds, 'b c i j -> b i j c')

			# apply softmax to obtain probabilities
			preds = torch.softmax(preds, dim = 3)

			# loop over each prediction and construct the aggregate sequence
			# tensor by combining crops
			for pred, sid, L, i, j in zip(preds, seq_ids, Ls, ii, jj):
				# if next crop is from a new sequence, compute loss and accuracy on prior sequence
				if sid != current_id:
					# normalize and trim seq_preds
					seq_preds[32:-32, :, :] /= 2
					seq_preds[:, 32:-32, :] /= 2
					seq_preds = seq_preds[:-crop_padding, :-crop_padding, :]

					# calc loss over whole seq
					loss = loss_func(rearrange(seq_preds, 'i j c -> c i j').unsqueeze(0), distances.unsqueeze(0).long())

					# multiply loss by mask and calculate mean
					loss = torch.mean(loss.mul(mask))
					sum_loss += loss.item()

					# calculate predicted contacts
					contact_preds = torch.sum(seq_preds[:, :, :20], dim = 2)
					contact_preds[mask == 0] = 0

					# calculate contact labels
					contact_labels = torch.zeros_like(contact_preds).to(device)
					contact_labels[distances < 20] = 1
					contact_labels[mask == 0] = 0

					# calculate accuracy over short, medium, and long contacts
					short_accs.append(calc_mask_acc(contact_preds, short_mask, contact_labels, seq_ks, threshold = threshold))
					med_accs.append(calc_mask_acc(contact_preds, med_mask, contact_labels, seq_ks, threshold = threshold))
					long_accs.append(calc_mask_acc(contact_preds, long_mask, contact_labels, seq_ks, threshold = threshold))

					# plot predictions as binary map if specified
					# this was used for testing to visualize the predicted and expected contacts
					if plot_predictions:
						fig, axs = plt.subplots(2,3)
						axs[0,0].imshow(contact_labels.to('cpu').numpy(), cmap = 'Greys', interpolation = 'nearest')
						axs[0,1].imshow(contact_preds.to('cpu').numpy(), cmap = 'Greys', interpolation = 'nearest')
						axs[1,0].imshow(torch.logical_and(contact_labels, short_mask[:contact_preds.shape[0], :contact_labels.shape[1]]).to('cpu').numpy(), cmap = 'Greys', interpolation = 'nearest')
						axs[1,1].imshow(torch.logical_and(contact_labels, med_mask[:contact_preds.shape[0], :contact_labels.shape[1]]).to('cpu').numpy(), cmap = 'Greys', interpolation = 'nearest')
						axs[1,2].imshow(torch.logical_and(contact_labels, long_mask[:contact_preds.shape[0], :contact_labels.shape[1]]).to('cpu').numpy(), cmap = 'Greys', interpolation = 'nearest')
						plt.show()
						plt.clf()

					# reset variables for next sequence
					current_id = sid
					seq_preds = torch.zeros((L+crop_padding, L+crop_padding, 64)).to(device)
					distances, mask = test_loader.dataset.get_mat_and_mask()
					distances, mask = distances.to(device), mask.to(device)

				# otherwise, continue summing crops from same sequence together
				seq_preds[i:i+crop_size, j:j+crop_size, :] += pred

	# calculate mean for each L/k for each range
	short_accs = np.mean(short_accs, axis = 0)
	med_accs = np.mean(med_accs, axis = 0)
	long_accs = np.mean(long_accs, axis = 0)

	# put into dataframe for formatting
	accs = {'short' : short_accs, 'med' : med_accs, 'long' : long_accs}
	df = pd.DataFrame(data = accs, index = seq_ks)

	# print test loss and accuracy
	print(f'\nTest loss per crop: {sum_loss/batch_idx/batch_size:.6f}\n')
	print('---Accuracies for L/k sequences--')
	print(df)
		
if __name__ == '__main__':
	main()