import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

import sidechainnet as scn
import numpy as np
from tqdm import tqdm
import sys
from einops import rearrange, reduce, repeat
import numba
from numba import jit

from Models import PSSM_Projector
from Models import Input_Feature_Projector
from Models import Residue_Index_Projector

# CONSTANTS

N_res = 256
N_clust = 16
c_m = 256
c_2 = 128
stride = 128

randomize_start = True
USE_DEBUG_DATA = True
progress_bar = True


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
	if USE_DEBUG_DATA:
		data = scn.load('debug', with_pytorch="dataloaders", seq_as_onehot=True, aggregate_model_input=False, batch_size=8, num_workers = 0)
	else:
		data = scn.load(casp_version = 7, with_pytorch="dataloaders", seq_as_onehot=True, aggregate_model_input=False, batch_size=8, num_workers = 0)

	pssm_projector = PSSM_Projector(N_clust, c_m)
	input_feature_projector = Input_Feature_Projector(c_2)
	residue_index_projector = Residue_Index_Projector(c_2)

	train_loader = data['train']
	for batch_idx, batch in enumerate(tqdm(train_loader, disable = not progress_bar)):
		seqs, evos, angs, masks, dmats, dmat_masks = get_seq_features(batch)
		# get dimensions for feature matrix
		B = seqs.shape[0]
		L = seqs.shape[1]

		# discreteize dmat in 64 bins
		# dmats = torch.floor(torch.clamp(dmats, 2, 21.6875).sub(2).mul(3.2))

		# pad data
		seqs = F.pad(seqs, (0, 1, 0, N_res), 'constant', 0)
		evos = F.pad(evos, (0, 0, 0, N_res), 'constant', 0)
		# masks = F.pad(masks, (0, self.N_res), 'constant', 0)
		# dmats = F.pad(dmats, (0, self.N_res, 0, self.N_res), 'constant', 0)
		# dmat_masks = F.pad(dmat_masks, (0, self.N_res, 0, self.N_res), 'constant', 0)

		# get PSSM data projections
		msa_reps = pssm_projector(evos)

		# get residue index and target feat projections
		li, lj = input_feature_projector(seqs.float())

		# calculate outer sum
		li = repeat(li, 'b i c -> b rep i c', rep = L + N_res)
		lj = repeat(lj, 'b i c -> b rep i c', rep = L + N_res)
		lj = rearrange(lj, 'b i j c -> b j i c')
		outer_sum = torch.add(li, lj)

		# calculate relative positional encodings
		all_res = torch.arange(L+N_res)
		di = repeat(all_res, 'i -> rep i', rep = L + N_res)
		dj = repeat(-all_res, 'j -> rep j', rep = L + N_res)
		dj = rearrange(dj, 'i j -> j i')

		# clamp differences and encode as onehot
		dd = torch.add(torch.clamp(torch.add(di, dj), -32, 32), 32)
		dd = F.one_hot(dd)

		# pass through linear layer
		# relpos_encoding = residue_index_projector(dd.float())

		# create pairwise representation
		# pairwise_reps = torch.add(outer_sum, relpos_encoding)

		# for each sequence of length L
		for seq, evo, d in zip(seqs, evos, dd):
			# get crops of length N_res
			# generate starting position for window
			start_i = 0 if L < 64 or not randomize_start else np.random.randint(0, 64)
			for i in range(start_i, L, stride):
				msa_crop = pssm_projector(evo[i:i+N_res])
				print(f'{msa_crop.shape = }')
				sys.exit(0)

				prw_crop = 1
				prw_crop, msa_crop = pairwise_rep[i:i+N_res, i:i+N_res]


if __name__ == '__main__':
	main()