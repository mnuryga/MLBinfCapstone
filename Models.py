import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from einops import rearrange

class Evoformer_Model(nn.Module):
	def __init__(self):
		super().__init__()
		pass
	
	def forward(self, x):
		pass

class PSSM_Projector(nn.Module):
	def __init__(self, num_layers, c_m):
		super().__init__()
		layers = [nn.Linear(21, c_m) for i in range(num_layers)]
		self.layers = nn.ModuleList(layers)
		self.c_m = c_m
		self.num_layers = num_layers
	
	def forward(self, x):
		out = torch.zeros((x.shape[0], self.num_layers, x.shape[1], self.c_m))
		for i in range(x.shape[0]):
			for j, l in enumerate(self.layers):
				out[i,j] = l(x[i])
		return out

class Input_Feature_Projector(nn.Module):
	def __init__(self, c_2):
		super().__init__()
		self.c_2 = c_2
		self.l1 = nn.Linear(21, c_2)
		self.l2 = nn.Linear(21, c_2)
	
	def forward(self, x):
		x1 = self.l1(x)
		x2 = self.l2(x)
		return x1, x2

class Residue_Index_Projector(nn.Module):
	def __init__(self, c_2):
		super().__init__()
		self.l = nn.Linear(65, c_2)
	
	def forward(self, x):
		return self.l(x)
