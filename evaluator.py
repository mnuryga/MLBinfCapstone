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
	model = nn.DataParallel(Alphafold2_Model(s, c_m, c_z, c), device_ids=[0]).to(device)
	model.eval()
	model.load_state_dict(torch.load('train_tiny_lr/best_5.pth')['state_dict'])

	# create test dataset that batches by sequence
	test_dataset = Evo_Dataset('test', stride, batch_size, 0, progress_bar, USE_DEBUG_DATA, by_seq = True)
	test_loader = DataLoader(dataset = test_dataset, batch_size = 1, drop_last = True)

	best_loss = float('inf')

	losses = []

	sum_loss = 0
	with torch.no_grad():
		# each batch from the test_loader will contain crops from the same sequence
		# these crops do not have a randomized starting position
		for t_batch_idx, (seqs, evos, masks, angs, coords, bb_rs, bb_ts) in enumerate(tqdm(test_loader, disable = True)):
			# send batch to device
			break
			seqs, evos, masks, angs, coords, bb_rs, bb_ts = seqs.to(device), evos.to(device), masks.to(device), angs.to(device), coords.to(device), bb_rs.to(device), bb_ts.to(device)

			pred_coords, L_fape, L_aux = model(seqs, evos, angs, (bb_rs, bb_ts), coords, masks)

			loss = torch.mean((0.5*L_fape + 0.5*L_aux)).item()/seqs.shape[1]
			if loss < best_loss:
				best_preds = pred_coords
				best_coords = coords
			sum_loss += loss
			losses.append(loss)

	np.save('losses.npy', np.array(losses))
	print(f'Test loss per C_alpha: {sum_loss/t_batch_idx}')

	# coords = best_coords.detach().cpu().numpy()
	# x = coords[:, :, 0][0, 1:]
	# y = coords[:, :, 1][0, 1:]
	# z = coords[:, :, 2][0, 1:]
	# x = np.load('best_x.npy')
	# y = np.load('best_y.npy')
	# z = np.load('best_z.npy')


	# # x = x/np.max(x)
	# # y = y/np.max(y)
	# # z = z/np.max(z)

	# ax = plt.gca(projection="3d")

	# ax.scatter(x,y,z, c='b',s=30)

	# ax.plot(x,y,z, color='r')

	# plt.show()
	# plt.clf()

	# preds = best_preds.detach().cpu().numpy()
	# x = preds[:, :, 0][0, 1:]
	# y = preds[:, :, 1][0, 1:]
	# z = preds[:, :, 2][0, 1:]

	# x = x/np.max(x)
	# y = y/np.max(y)
	# z = z/np.max(z)

	# ax = plt.gca(projection="3d")

	# ax.scatter(x,y,z, c='b',s=30)

	# ax.plot(x,y,z, color='r')

	# plt.show()
	# plt.clf()

	# coords = best_coords.detach().cpu().numpy()
	# x = coords[:, :, 0][0, 1:]
	# y = coords[:, :, 1][0, 1:]
	# z = coords[:, :, 2][0, 1:]

	# x = x/np.max(x)
	# y = y/np.max(y)
	# z = z/np.max(z)

	# ax = plt.gca(projection="3d")

	# ax.scatter(x,y,z, c='b',s=30)

	# ax.plot(x,y,z, color='r')

	# plt.show()
	# plt.clf()


if __name__ == '__main__':
	main()
