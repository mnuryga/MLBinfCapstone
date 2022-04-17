import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from einops import rearrange

class Triangular_Multiplicative_Model(nn.Module):
	def __init__(self, direction, c_z = 128, c = 128):
		super().__init__()
		self.c = c
		self.direction = direction
		self.ln1 = nn.LayerNorm(c_z)
		self.la1 = nn.Linear(c_z, c)
		self.la2 = nn.Linear(c_z, c)
		self.lb1 = nn.Linear(c_z, c)
		self.lb2 = nn.Linear(c_z, c)
		self.ln2 = nn.LayerNorm(c)
		self.lg = nn.Linear(c_z, c_z)
		self.lz = nn.Linear(c, c_z)
	
	def forward(self, x):
		z = self.ln1(x)
		a = torch.sigmoid(torch.mul(self.la1(z), self.la2(z)))
		b = torch.sigmoid(torch.mul(self.lb1(z), self.lb2(z)))
		if self.direction == 'incoming':
			a = rearrange(a, 'b i j k -> b j i k')
			b = rearrange(b, 'b i j k -> b j i k')
		g = torch.sigmoid(self.lg(z))
		z = torch.zeros((z.shape[0], z.shape[1], z.shape[2], self.c))
		for i in range(a.shape[1]):
			for j in range(b.shape[2]):
				ai = a[:, i, :]
				bj = b[:, :, j]
				z[:, i, j] = torch.sum(torch.mul(ai, bj), dim = -2)
		z = torch.mul(g, self.lz(self.ln2(z)))
		return z

class Evoformer(nn.Module):
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
