import torch
from torch.utils.data import IterableDataset

import sidechainnet as scn
import numpy as np
from tqdm import tqdm
import sys
from einops import rearrange

class Evo_Dataset(IterableDataset):
	def __init__(self, key, stride):
		super().__init__()
		self.key = key
		self.stride = stride
		self.data = scn.load(casp_version = 7, with_pytorch="dataloaders", seq_as_onehot=True,
													aggregate_model_input=False, batch_size=16, num_workers = 0)

	@classmethod
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

	def __iter__(self):
		for batch_idx, batch in enumerate(tqdm(self.data[self.key], disable = not progress_bar)):
			seqs, evos, angs, masks, dmats, dmat_masks = get_seq_features(batch)

			# get dimensions for feature matrix
			L = seqs.shape[1]