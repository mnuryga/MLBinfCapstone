import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from einops import rearrange, reduce, repeat

class MHSA(nn.Module):
	def __init__(self, c_m, c_z, heads=8, dim_head=None, bias=True):
		'''
		Gated self-attention with or without pair bias
		c_m: channel dim target attention
		c_z: channel dim of the pair wise bias
		heads: number of heads for the multi-head attention
		dim_head: channel dim of each head
		bias: Apply pair-wise bias or not
		'''
		super().__init__()
		self.bias = bias
		self.dim_head = (int(c_m / heads)) if dim_head is None else dim_head
		_dim = self.dim_head * heads
		self.heads = heads
		self.to_qvk = nn.Linear(c_m, _dim * 4, bias=False)
		self.W_0 = nn.Linear( _dim, c_m, bias=False)
		self.scale_factor = self.dim_head ** -0.5
		
		self.fc_scale_bias = nn.Linear(c_z, heads)

	def forward(self, x, bias_rep=None, mask=None):
		'''
		x: input for self-attention
		bias_rep: pair-wise bias
		'''

		# Step 1
		qkv = self.to_qvk(x)  # [batch, tokens, c_m*3*heads ]

		# Step 2
		# decomposition to q,v,k and cast to tuple
		# the resulted shape before casting to tuple will be:
		# [3, batch, heads, tokens, dim_head]
		q, k, v, g = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d ', k=4, h=self.heads))
		
		# Step 3
		# resulted shape will be: [batch, heads, tokens, tokens]
		scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor

		if mask is not None:
			assert mask.shape == scaled_dot_prod.shape[2:]
			scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

		# pair wise bias
		scaled_bias = 0
		if self.bias:
			scaled_bias = self.fc_scale_bias(bias_rep)
			scaled_bias = rearrange(scaled_bias, 'i j k -> k i j').unsqueeze(0)
			
		attention = torch.softmax(scaled_dot_prod + scaled_bias, dim=-1)
		
		# Step 4. Calc result per batch and per head h
		out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
		
		# gating
		g = torch.sigmoid(g)
		out *= g

		# Step 5. Re-compose: merge heads with dim_head d
		out = rearrange(out, "b h t d -> b t (h d)")

		# Step 6. Apply final linear transformation layer
		return self.W_0(out)

class MSA_Stack(nn.Module):
	def __init__(self, batch_size, c_m, c_z, heads=8, dim_head=None, device = 'cpu'):
		'''
		Do a row-wise MHSA with pair bias follow by a column-wise 
		MHSA without bias. The result is then passed through a two
		layer MLP as transition.
		'''
		super().__init__()
		self.device = device
		# batches of row wise MHSA
		self.row_MHSA = nn.ModuleList([MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=True, dim_head=None) for i in range(batch_size)])
		# batches of col wise MHSA
		self.col_MHSA = nn.ModuleList([MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=False, dim_head=None) for i in range(batch_size)])
		# transition MLP
		self.fc1 = nn.Linear(c_m, 4 * c_m)
		self.fc2 = nn.Linear(4 * c_m, c_m)
		
	def forward(self, x, bias_rep):
		res = torch.empty(x.shape).to(self.device)
		# row wise gated self-attention with pair bias
		for i, mhsa in enumerate(self.row_MHSA):
			res[i] = mhsa(x[i].clone(), bias_rep[i].clone())
		x = x + res # add residuals
		
		res2 = torch.empty(x.shape).to(self.device)
		# column wise gated self-attention
		x_trans = rearrange(x, 'b i j k -> b j i k')
		for i, mhsa in enumerate(self.col_MHSA):
			res2[i] = rearrange(mhsa(x_trans[i]), 'i j k -> j i k')
		x = x + res2 # add residuals
		
		# transiion
		r = F.relu(self.fc1(x))
		r = self.fc2(r) + x
		
		return r

class Outer_Product_Mean(nn.Module):
	
	def __init__(self, c_m, c_z, c=32, device = 'cpu'):
		super().__init__()
		self.device = device
		# linear projections
		self.fc1 = nn.Linear(c_m, c)
		self.fc2 = nn.Linear(c**2, c_z)
		self.flatten = nn.Flatten(start_dim=3)
		self.c = c
		self.c_z = c_z
		
	def forward(self, x):
		'''
		x: B x S x R x C
		res: B x R x R x C
		'''
		# results
		res = torch.empty(x.shape[0], x.shape[-2], x.shape[-2], self.c, self.c).to(self.device)
		
		# project in_c to out_c
		x = self.fc1(x)
		
		# loop over R
		for i in range(x.shape[-2]):
			for j in range(x.shape[-2]):
				mean_s = torch.mean(torch.einsum('bij,bik->bijk', [x[:, :, i, :], x[:, :, j, :]]), dim=1)
				res[:, i, j, :, :] = mean_s
		
		# flatten and project back
		res = self.flatten(res)
		res = self.fc2(res)
		
		return res
				
class Pair_Stack(nn.Module):
	def __init__(self, batch_size, c_z, heads=8, dim_head=None, device = 'cpu'):
		'''
		Do a row-wise MHSA with pair bias on the start edges follow
		by a column-wise MHSA with bias on the end edes. The result 
		is then passed through a two layer MLP as transition.
		'''
		super().__init__()
		self.device = device
		# batches of row wise MHSA
		self.start_MHSA = nn.ModuleList([MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head) for i in range(batch_size)])
		# batches of col wise MHSA
		self.end_MHSA = nn.ModuleList([MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head) for i in range(batch_size)])
		# transition MLP
		self.fc1 = nn.Linear(c_z, 4 * c_z)
		self.fc2 = nn.Linear(4 * c_z, c_z)
		
	def forward(self, x):
		res = torch.empty(x.shape).to(self.device)
		# row wise gated self-attention with pair bias
		for i, mhsa in enumerate(self.start_MHSA):
			res[i] = mhsa(x[i].clone(), x[i].clone())
		x = x + res # add residuals
		
		res2 = torch.empty(x.shape).to(self.device)
		# column wise gated self-attention
		x_trans = rearrange(x, 'b i j k -> b j i k')
		for i, mhsa in enumerate(self.end_MHSA):
			# res[i] = mhsa(x_trans[i], x_trans[i])
			res2[i] = rearrange(mhsa(x_trans[i].clone(), x_trans[i].clone()), 'i j k -> j i k')
		x = x + res2 # add residuals
		
		# transiion
		r = F.relu(self.fc1(x))
		r = self.fc2(r) + x
		
		return r

class Triangular_Multiplicative_Model(nn.Module):
	def __init__(self, direction, c_z = 128, c = 16, device = 'cpu'):
		super().__init__()
		self.c = c
		self.device = device
		self.direction = direction
		# self.ln1 = nn.LayerNorm(c_z)
		self.la1 = nn.Linear(c_z, c)
		self.la2 = nn.Linear(c_z, c)
		self.lb1 = nn.Linear(c_z, c)
		self.lb2 = nn.Linear(c_z, c)
		# self.ln2 = nn.LayerNorm(c)
		self.lg = nn.Linear(c_z, c_z)
		self.lz = nn.Linear(c, c_z)
	
	def forward(self, x):
		# z = self.ln1(x)
		z = x
		a = torch.sigmoid(torch.mul(self.la1(z), self.la2(z)))
		b = torch.sigmoid(torch.mul(self.lb1(z), self.lb2(z)))
		if self.direction == 'incoming':
			a = rearrange(a, 'b i j k -> b j i k')
			b = rearrange(b, 'b i j k -> b j i k')
		g = torch.sigmoid(self.lg(z))
		z = torch.zeros((z.shape[0], z.shape[1], z.shape[2], self.c)).to(self.device)
		for i in range(a.shape[1]):
			for j in range(b.shape[2]):
				ai = a[:, i, :]
				bj = b[:, :, j]
				z[:, i, j] = torch.sum(torch.mul(ai, bj), dim = -2)
		# z = torch.mul(g, self.lz(self.ln2(z)))
		z = torch.mul(g, self.lz(z))
		return z

class PSSM_Projector(nn.Module):
	'''
	model to project pssm data to 16 layers
	'''
	def __init__(self, num_layers, c_m, device = 'cpu'):
		super().__init__()
		layers = [nn.Linear(21, c_m) for i in range(num_layers)]
		self.layers = nn.ModuleList(layers)
		self.c_m = c_m
		self.num_layers = num_layers
		self.device = device
	
	def forward(self, x):
		out = torch.zeros((x.shape[0], self.num_layers, x.shape[1], self.c_m)).to(self.device)
		# for each batch, apply a linear layer to pssm data
		for i in range(x.shape[0]):
			for j, l in enumerate(self.layers):
				out[i,j] = l(x[i])
		return out

class Input_Feature_Projector(nn.Module):
	'''
	projects the input features to c_2 features
	'''
	def __init__(self, c_2):
		super().__init__()
		self.c_2 = c_2
		self.l1 = nn.Linear(21, c_2)
		self.l2 = nn.Linear(21, c_2)
	
	def forward(self, x):
		# pass input thorugh linear layers
		x1 = self.l1(x)
		x2 = self.l2(x)
		return x1, x2

class Residue_Index_Projector(nn.Module):
	'''
	projects onehot residue input to c_2 features
	'''
	def __init__(self, c_2):
		super().__init__()
		self.l = nn.Linear(65, c_2)
	
	def forward(self, x):
		# pass through linear layer
		return self.l(x)

class Representation_Projector(nn.Module):
	def __init__(self, r, s, c_m, c_z, device = 'cpu'):
		super().__init__()
		self.r = r
		self.s = s
		self.c_m = c_m
		self.c_z = c_z
		self.device = device
		self.pssm_projector = PSSM_Projector(s, c_m, device = device)
		self.input_feature_projector = Input_Feature_Projector(c_z)
		self.residue_index_projector = Residue_Index_Projector(c_z)
	
	def forward(self, seqs, evos):
		L = seqs.shape[1]
		# get pssm data projections
		msa_reps = self.pssm_projector(evos)

		# get residue index and target feat projections
		li, lj = self.input_feature_projector(seqs.float())

		# calculate outer sum
		li = repeat(li, 'b i c -> b rep i c', rep = L)
		lj = repeat(lj, 'b i c -> b rep i c', rep = L)
		lj = rearrange(lj, 'b i j c -> b j i c')
		outer_sum = torch.add(li, lj)

		# calculate relative positional encodings
		all_res = torch.arange(L).to(self.device)
		di = repeat(all_res, 'i -> rep i', rep = L)
		dj = repeat(-all_res, 'j -> rep j', rep = L)
		dj = rearrange(dj, 'i j -> j i')

		# clamp differences and encode as onehot
		d = torch.add(torch.clamp(torch.add(di, dj), -32, 32), 32)
		d = F.one_hot(d)

		# pass through linear layer
		relpos_encoding = self.residue_index_projector(d.float())

		# create pairwise representation
		prw_reps = torch.add(outer_sum, relpos_encoding)

		return prw_reps, msa_reps

class Evoformer_Trunk(nn.Module):
	'''
	evoformer trunk as outlined in the alphafold2 paper
	'''
	def __init__(self, batch_size, c_m, c_z, c, device = 'cpu'):
		super().__init__()
		self.msa_stack = MSA_Stack(batch_size, c_m, c_z, heads = 4, dim_head = c, device = device)
		self.outer_product_mean = Outer_Product_Mean(c_m, c_z, c = c, device = device)
		self.triangular_mult_outgoing = Triangular_Multiplicative_Model('outgoing', c_z = c_z, c = c, device = device)
		self.triangular_mult_incoming = Triangular_Multiplicative_Model('incoming', c_z = c_z, c = c, device = device)
		self.pair_stack = Pair_Stack(batch_size, c_z, heads = 4, dim_head = c, device = device)

	def forward(self, prw_rep, msa_rep):
		# pass msa through attention module
		msa_rep = self.msa_stack(msa_rep, prw_rep)

		# calculate outer product of msa and add residual
		x = self.outer_product_mean(msa_rep) + prw_rep

		# pass through triangular multipication for 
		# outgoing and incoming edges
		x = self.triangular_mult_outgoing(x) + x
		x = self.triangular_mult_incoming(x) + x

		# pass pairwise rep through attention module
		prw_rep = self.pair_stack(x) + x
		return prw_rep, msa_rep

class Evo_Model(nn.Module):
	def __init__(self, batch_size, r, s, c_m, c_z, c, device = 'cpu'):
		super().__init__()
		self.rep_proj = Representation_Projector(r, s, c_m, c_z, device = device)
		self.evoformer_trunk = Evoformer_Trunk(batch_size, c_m, c_z, c, device = device)
		self.proj_dmat = nn.Conv2d(c_z, 64, 1)
		self.angs_pool = nn.MaxPool2d((1, 64))
		self.proj_angs = nn.Conv2d(c_z, 1296, 1)
	
	def forward(self, seqs, evos):
		prw_rep, msa_rep = self.rep_proj(seqs, evos)
		prw_rep, msa_rep = self.evoformer_trunk(prw_rep, msa_rep)
		c_first = rearrange(prw_rep, 'b i j c -> b c i j')
		pred_dmat = self.proj_dmat(c_first)
		pred_angs = self.proj_angs(self.angs_pool(c_first))
		return pred_dmat, pred_angs.squeeze(-1)