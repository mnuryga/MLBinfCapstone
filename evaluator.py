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
from mpl_toolkits.mplot3d import Axes3D

from datasets import Evo_Dataset
from models import Alphafold2_Model

# CONSTANTS
batch_size = 1
r = 64
c_m = 128
c_z = 64
c = 16
s = 8
stride = 32

progress_bar = True
USE_DEBUG_DATA = False

def main():
	# get device
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create and load alphafold2 model from training
	model = nn.DataParallel(Alphafold2_Model(s, c_m, c_z, c)).to(device)
	model.eval()
	model.load_state_dict(torch.load('checkpoints/best_5.pth')['state_dict'])

	# create test dataset that batches by sequence
	test_dataset = Evo_Dataset('test', stride, batch_size, 0, progress_bar, USE_DEBUG_DATA, by_seq = True)
	test_loader = DataLoader(dataset = test_dataset, batch_size = 1, drop_last = True)

	best_loss = float('inf')

	losses = []
	s_len = []
	all_preds = np.zeros((93, 600, 3))
	all_coords = np.zeros((93, 600, 3))

	sum_loss = 0
	with torch.no_grad():
		# each batch from the test_loader will contain crops from the same sequence
		# these crops do not have a randomized starting position
		for t_batch_idx, (seqs, evos, masks, angs, coords, bb_rs, bb_ts) in enumerate(tqdm(test_loader, disable = True)):
			# send batch to device
			seqs, evos, masks, angs, coords, bb_rs, bb_ts = seqs.to(device), evos.to(device), masks.to(device), angs.to(device), coords.to(device), bb_rs.to(device), bb_ts.to(device)

			pred_coords, L_fape, L_aux = model(seqs, evos, angs, (bb_rs, bb_ts), coords, masks)

			loss = torch.mean((0.5*L_fape + 0.5*L_aux)).item()

			if loss < best_loss:
				best_preds = pred_coords
				best_coords = coords
			sum_loss += loss
			losses.append(loss)
			s_len.append(seqs.shape[1])
			all_preds[t_batch_idx, :seqs.shape[1]] = pred_coords.detach().cpu().numpy()
			all_coords[t_batch_idx, :seqs.shape[1]] = coords.detach().cpu().numpy()
			sys.exit(0)

	# write all data to files for use in visualization and loss processing
	np.save('visualization/losses.npy', np.array(losses))
	np.save('visualization/s_len.npy', np.array(s_len))
	np.save('visualization/all_preds.npy', np.array(all_preds))
	np.save('visualization/all_coords.npy', np.array(all_coords))
	print(f'Test loss per C_alpha: {sum_loss/t_batch_idx}')

if __name__ == '__main__':
	main()
