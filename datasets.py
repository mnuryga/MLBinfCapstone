'''
Parse data from sidechainnet into batches of crops.

Author: Matthew Uryga
'''
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

import sidechainnet as scn
import numpy as np
from tqdm import tqdm
import sys

class Evo_Dataset(IterableDataset):
	def __init__(self, key, stride, batch_size, r, progress_bar, USE_DEBUG_DATA, by_seq = False):
		super().__init__()
		self.by_seq = by_seq
		self.progress_bar = progress_bar
		self.key = key
		self.stride = stride
		self.r = r
		if USE_DEBUG_DATA:
			self.data = scn.load('debug', with_pytorch="dataloaders", seq_as_onehot=True,
				aggregate_model_input=False, batch_size=batch_size, num_workers = 0)
		else:
			self.data = scn.load(casp_version = 7, with_pytorch="dataloaders", seq_as_onehot=True,
				aggregate_model_input=False, batch_size=batch_size, num_workers = 0)

	@staticmethod
	# Taken from alphafold1 starter code
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
		# loop over sets of sequences with same length
		for batch_idx, batch in enumerate(tqdm(self.data[self.key], disable = not self.progress_bar)):
			seqs, evos, angs, masks, dmats, dmat_masks = self.get_seq_features(batch)

			# get dimensions for feature matrix
			B = seqs.shape[0]
			L = seqs.shape[1]

			# discreteize dmat in 64 bins
			dmats = torch.floor(torch.clamp(dmats, 2, 21.6875).sub(2).mul(3.2))
			desc_angs = torch.floor(torch.add(angs, np.pi).mul(5))
			angs = torch.zeros((angs.shape[0], angs.shape[1]))
			angs = 36 * desc_angs[:, :, 0] + desc_angs[:, :, 1]

			# pad data
			seqs = F.pad(seqs, (0, 1, 0, self.r), 'constant', 0)
			evos = F.pad(evos, (0, 0, 0, self.r), 'constant', 0)
			# masks = F.pad(masks, (0, self.r), 'constant', 0)
			dmats = F.pad(dmats, (0, self.r, 0, self.r), 'constant', 0)
			dmat_masks = F.pad(dmat_masks, (0, self.r, 0, self.r), 'constant', 0)
			angs = F.pad(angs, (0, self.r), 'constant', 0)

			# dmat is symmetrical, so we want to mask out all positions below the diagonal
			for i in range(L):
				dmat_masks[:, i, :i] = 0

			for seq, evo, dmat, dmat_mask, ang in zip(seqs, evos, dmats, dmat_masks, angs):
				# get crops of length r
				# generate starting position for window
				if not self.by_seq:
					start_i = 0 if L < 64 else np.random.randint(0, 64)
					for i in range(start_i, L, self.stride):
						yield seq[i:i+self.r], evo[i:i+self.r], dmat[i:i+self.r, i:i+self.r], dmat_mask[i:i+self.r, i:i+self.r], ang[i:i+self.r]
				else:
					num_seq = (L+self.r)//self.stride
					s = torch.zeros((num_seq, self.r, 21))
					e = torch.zeros((num_seq, self.r, 21))
					d = torch.zeros((num_seq, self.r, self.r))
					n = torch.zeros((num_seq, self.r, self.r))
					a = torch.zeros((num_seq, self.r))
					count = 0
					for i in range(0, L, self.stride):
						s[count] = seq[i:i+self.r]
						e[count] = evo[i:i+self.r]
						d[count] = dmat[i:i+self.r, i:i+self.r]
						n[count] = dmat_mask[i:i+self.r, i:i+self.r]
						a[count] = ang[i:i+self.r]
						count += 1
					yield s, e, d, n, a, dmat, dmat_mask


# main function for testing
def main():
	ds = Evo_Dataset('train', 128)
	dl = DataLoader(dataset = ds, batch_size = 5, num_workers = 0,  drop_last = True)
	for i, (pr, mr, dm, mm) in enumerate(dl):
		print(f'pr.shape={pr.shape}')
		print(f'mr.shape={mr.shape}')
		print(f'dm.shape={dm.shape}')
		print(f'mm.shape={mm.shape}')
		sys.exit(0)

if __name__ == '__main__':
	main()