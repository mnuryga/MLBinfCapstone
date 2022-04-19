'''
Evaluates the performance of the AlphaFold 2 model.

Author: Matthew Uryga
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import sys
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import pandas as pd

from evo_dataset import Evo_Dataset
from models import Evo_Model
from alphafold1 import Residual_Model

# CONSTANTS
batch_size = 16
r = 64
c_m = 128
c_z = 64
c = 8
s = 8
stride = 32
mask_size = 1000

progress_bar = True
threshold = False
USE_DEBUG_DATA = True

# what L/k values are being used for accuracy calculation
seq_ks = [1, 2, 5, 10, 20, 50, 100]

def main():
	# get device
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create and load alphafold1 model
	af1_model = Residual_Model(4).to(device)
	af1_model.eval()
	# af1_model = af1_model.load_state_dict(torch.load('checkpoints/best_alphafold1.pth')['state_dict'])

	# create and load evoformer
	# evo_model = Evo_Model(batch_size, r, s, c_m, c_z, c, device = device).to(device)
	evo_model = Evo_Model(1, r, s, c_m, c_z, c).to(device)
	evo_model.eval()
	# evo_model = evo_model.load_state_dict(torch.load('checkpoints/best.pth')['state_dict'])

	# create test dataset that batches by sequence
	test_dataset = Evo_Dataset('test', stride, batch_size, r, progress_bar, USE_DEBUG_DATA, by_seq = True)
	test_loader = DataLoader(dataset = test_dataset, batch_size = 1, drop_last = True)

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

	# lists to store accuracy per sequence
	short_accs = []
	med_accs = []
	long_accs = []

	# create masks for short/medium/long range contacts
	short_mask = torch.zeros((mask_size, mask_size)).to(device)
	med_mask = torch.zeros((mask_size, mask_size)).to(device)
	long_mask = torch.zeros((mask_size, mask_size)).to(device)

	for i in range(mask_size):
		short_mask[i, i+6:i+12] = 1
		med_mask[i, i+12:i+24] = 1
		long_mask[i, i+24:] = 1

	loss_func = nn.CrossEntropyLoss(reduction = 'none')

	num_crops = 0
	sum_loss = 0
	with torch.no_grad():
		# each batch from the test_loader will contain crops from the same sequence
		# these crops do not have a randomized starting position
		for batch_idx, (seqs, evos, dmat, dmat_mask, angs, full_dmat, full_mask) in enumerate(tqdm(test_loader, disable = True)):
			# send batch to device
			seqs, evos, dmat, dmat_mask, angs, full_dmat, full_mask = seqs.to(device), evos.to(device), dmat.to(device), dmat_mask.to(device), angs.to(device), full_dmat.to(device), full_mask.to(device)
			seqs, evos, dmat, dmat_mask, angs, full_dmat, full_mask = seqs.squeeze(), evos.squeeze(), dmat.squeeze(), dmat_mask.squeeze(), angs.squeeze(), full_dmat.squeeze(), full_mask.squeeze()

			B = seqs.shape[0]
			L = full_mask.shape[0]
			num_crops += B

			# run forward pass and cross entropy loss - reduction is none, so
			# loss output is a batch*crop_size*crop_size tensor
			# we implemented mhsa with batch_size as a parameter to drastically
			# improve performace, but this requires knowing the batch size at
			# initialization.  The dataloader returns a batch with all crops
			# for one sequence in it -> the batch size is variable from
			# sequence to sequence.  As such, we set batch_size to 1 for
			# mhsa and send each individual tensor to the model
			pred_dmat = torch.zeros((B, 64, r, r)).to(device)
			pred_angs = torch.zeros((B, 1296, r)).to(device)
			for i, seq, evo in zip(range(B), seqs, evos):
				pred_d, pred_a = evo_model(seq.unsqueeze(0), evo.unsqueeze(0))
				pred_dmat[i] = pred_d
				pred_angs[i] = pred_a

			dmat_loss = loss_func(pred_dmat, dmat.long()).mul(dmat_mask)
			angs_loss = loss_func(pred_angs, angs.long())

			# add loss to running total
			sum_loss += (torch.mean(dmat_loss)+torch.mean(angs_loss)).item()

			# apply softmax to obtain probabilities
			pred_dmat = rearrange(pred_dmat, 'b c i j -> b i j c')
			pred_dmat_probs = torch.softmax(pred_dmat, dim = -1)

			# create tensor to store aggregate probabilities
			pred_seq_probs = torch.zeros((L, L, 64)).to(device)

			# loop over crops and combine them into the total tensor
			for i, crop in zip(range(0, L-r, stride), pred_dmat_probs):
				# accout for overlap from when stride < r
				crop[:stride, :stride] /= 2
				crop[stride:, stride:] /= 2
				pred_seq_probs[i:i+r, i:i+r] += crop

			# the corners will be halved but they do not overlap with any other crops,
			# so multiply corners by 2
			pred_seq_probs[:stride, :stride] *= 2
			pred_seq_probs[-stride:, -stride:] *= 2

			# determine which pairs are within contact of eachother 
			contact_preds = torch.round(torch.sum(pred_seq_probs[:, :, :20], dim = -1))
			contact_preds[full_mask == 0] = 0

			# calculate contact labels
			contact_labels = torch.zeros_like(contact_preds).to(device)
			contact_labels[full_dmat < 20] = 1
			contact_labels[full_mask == 0] = 0

			# calculate accuracy over short, medium, and long contacts
			short_accs.append(calc_mask_acc(contact_preds, short_mask, contact_labels, seq_ks, threshold = threshold))
			med_accs.append(calc_mask_acc(contact_preds, med_mask, contact_labels, seq_ks, threshold = threshold))
			long_accs.append(calc_mask_acc(contact_preds, long_mask, contact_labels, seq_ks, threshold = threshold))

			# temp code to visualize the predicted contact map
			# plt.imshow(contact_preds.to('cpu').numpy(), cmap = 'Greys', interpolation = 'nearest')
			# plt.show()
			# sys.exit(0)

	# calculate mean for each L/k for each range
	short_accs = np.mean(short_accs, axis = 0)
	med_accs = np.mean(med_accs, axis = 0)
	long_accs = np.mean(long_accs, axis = 0)

	# put into dataframe for formatting
	accs = {'short' : short_accs, 'med' : med_accs, 'long' : long_accs}
	df = pd.DataFrame(data = accs, index = seq_ks)

	# print test loss and accuracy
	print(f'\nTest loss per crop: {sum_loss/num_crops:.6f}\n')
	print('---Accuracies for L/k sequences--')
	print(df)
		


if __name__ == '__main__':
	main()