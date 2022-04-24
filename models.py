'''
Building blocks of the AlphaFold 2 model.

Author: Matthew Uryga, Yu-Kai "Steven" Wang
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from einops import rearrange, repeat
from pytorch3d.transforms import quaternion_to_matrix

class MHSA(nn.Module):
	def __init__(self, c_m, c_z, heads=8, dim_head=None, bias=True):
		'''
		Gated self-attention with or without pair bias
		c_m: channel dim target attention
		c_z: channel dim of the pair wise bias
		heads: number of heads for the multi-head attention
		dim_head: channel dim of each head
		bias: Apply pair-wise bias or not

		Author: Yu-Kai "Steven" Wang
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
		# get q, v, k, g matrix for attention training
		qkv = self.to_qvk(x)
		q, k, v, g = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d ', k=4, h=self.heads))
		
		# dot product attention
		scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor

		if mask is not None:
			assert mask.shape == scaled_dot_prod.shape[2:]
			scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

		# pair-wise bias
		scaled_bias = 0
		if self.bias:
			scaled_bias = self.fc_scale_bias(bias_rep)
			scaled_bias = rearrange(scaled_bias, 'i j k -> k i j').unsqueeze(0)
			
		attention = torch.softmax(scaled_dot_prod + scaled_bias, dim=-1)
		
		# dot product with matrix v
		out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
		
		# gating
		g = torch.sigmoid(g)
		out *= g

		# concat heads
		out = rearrange(out, "b h t d -> b t (h d)")

		# transform back to initial dimension
		return self.W_0(out)

class MSA_Stack(nn.Module):
	def __init__(self, c_m, c_z, heads=8, dim_head=None):
		'''
		Do batches of row-wise MHSA with pair bias follow by a column-wise 
		MHSA without bias. The result is then passed through a two
		layer MLP as transition.

		Author: Yu-Kai "Steven" Wang
		'''
		super().__init__()
		# batches of row wise MHSA
# 		self.row_MHSA = nn.ModuleList([MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=True, dim_head=None) for i in range(batch_size)])
		self.row_MHSA = MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=True, dim_head=dim_head)
		# batches of col wise MHSA
# 		self.col_MHSA = nn.ModuleList([MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=False, dim_head=None) for i in range(batch_size)])
		self.col_MHSA = MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=False, dim_head=dim_head)
		# transition MLP
		self.fc1 = nn.Linear(c_m, 4 * c_m)
		self.fc2 = nn.Linear(4 * c_m, c_m)
		# layer norms
		self.ln1 = nn.LayerNorm(c_m)
		self.ln2 = nn.LayerNorm(c_z)
		self.ln3 = nn.LayerNorm(c_m)
		
	def forward(self, x, bias_rep):
		# results
		res = torch.empty(x.shape).to(x.get_device())

		# layer norms
		x = self.ln1(x)
		bias_rep = self.ln2(bias_rep)

# 		# row wise gated self-attention with pair bias
# 		for i, mhsa in enumerate(self.row_MHSA):
# 			res[i] = mhsa(x[i].clone(), bias_rep[i].clone())
# 		x = x + res # add residuals

		# row wise gated self-attention with pair bias, loop through batch
		for i in range(x.shape[0]):
			res[i] = self.row_MHSA(x[i].clone(), bias_rep[i].clone())
		x = x + res # add residuals

		
		# results
		res2 = torch.empty(x.shape).to(x.get_device())

		# layer norms
		x = self.ln3(x)

# 		# column wise gated self-attention
# 		x_trans = rearrange(x, 'b i j k -> b j i k')
# 		for i, mhsa in enumerate(self.col_MHSA):
# 			res2[i] = rearrange(mhsa(x_trans[i]), 'i j k -> j i k')
# 		x = x + res2 # add residuals

		# column wise gated self-attention
		x_trans = rearrange(x, 'b i j k -> b j i k')
		for i in range(x_trans.shape[0]):
			res2[i] = rearrange(self.col_MHSA(x_trans[i]), 'i j k -> j i k')
		x = x + res2 # add residuals

		
		# transiion
		r = F.relu(self.fc1(x))
		r = self.fc2(r) + x
		
		return r

class Outer_Product_Mean(nn.Module):
	def __init__(self, c_m, c_z, c=32):
		'''
		Do a linear transform, outer-product, mean,
		followed by another linear transform.

		Author: Yu-Kai "Steven" Wang
		'''
		super().__init__()
		# linear projections
		self.fc1 = nn.Linear(c_m, c)
		self.fc2 = nn.Linear(c**2, c_z)
		self.flatten = nn.Flatten(start_dim=3)
		self.c = c
		self.c_z = c_z
		# layer norms
		self.ln = nn.LayerNorm(c_m)
		
	def forward(self, x):
		'''
		x: B x S x R x C
		res: B x R x R x C
		'''
		# results
		res = torch.empty(x.shape[0], x.shape[-2], x.shape[-2], self.c, self.c).to(x.get_device())
		     
		# layer norm
		x = self.ln(x)
		
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
	def __init__(self, c_z, heads=8, dim_head=None):
		'''
		Do a row-wise MHSA with pair bias on the start edges follow
		by a column-wise MHSA with bias on the end edes. The result 
		is then passed through a two layer MLP as transition.

		Author: Yu-Kai "Steven" Wang
		'''
		super().__init__()
		# batches of row wise MHSA
		self.start_MHSA = MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head)
# 		self.start_MHSA = nn.ModuleList([MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head) for i in range(batch_size)])
		# batches of col wise MHSA
		self.end_MHSA = MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head)
# 		self.end_MHSA = nn.ModuleList([MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head) for i in range(batch_size)])
		# transition MLP
		self.fc1 = nn.Linear(c_z, 4 * c_z)
		self.fc2 = nn.Linear(4 * c_z, c_z)
		# layer norms
		self.ln1 = nn.LayerNorm(c_z)
		self.ln2 = nn.LayerNorm(c_z)
		
	def forward(self, x):
		# results
		res = torch.empty(x.shape).to(x.get_device())

		# layer norms
		x = self.ln1(x)

# 		# row wise gated self-attention with pair bias
# 		for i, mhsa in enumerate(self.start_MHSA):
# 			res[i] = mhsa(x[i].clone(), x[i].clone())
# 		x = x + res # add residuals

		# row wise gated self-attention with pair bias
		for i in range(x.shape[0]):
			res[i] = self.start_MHSA(x[i].clone(), x[i].clone())
		x = x + res # add residuals

		
		# results
		res2 = torch.empty(x.shape).to(x.get_device())

		# layer norms
		x = self.ln2(x)

# 		# column wise gated self-attention
# 		x_trans = rearrange(x, 'b i j k -> b j i k')
# 		for i, mhsa in enumerate(self.end_MHSA):
# 			res2[i] = rearrange(mhsa(x_trans[i].clone(), x_trans[i].clone()), 'i j k -> j i k')
# 		x = x + res2 # add residuals

		# column wise gated self-attention
		x_trans = rearrange(x, 'b i j k -> b j i k')
		for i in range(x_trans.shape[0]):
			res2[i] = rearrange(self.end_MHSA(x_trans[i].clone(), x_trans[i].clone()), 'i j k -> j i k')
		x = x + res2 # add residuals

		
		# transiion
		r = F.relu(self.fc1(x))
		r = self.fc2(r) + x
		
		return r

class Triangular_Multiplicative_Model(nn.Module):
	def __init__(self, direction, c_z = 128, c = 16):
		'''
		Do batches of triangular multiplication

		Author: Matthew Uryga
		'''
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
		z = x
		a = torch.sigmoid(torch.mul(self.la1(z), self.la2(z)))
		b = torch.sigmoid(torch.mul(self.lb1(z), self.lb2(z)))
		if self.direction == 'incoming':
			a = rearrange(a, 'b i j k -> b j i k')
			b = rearrange(b, 'b i j k -> b j i k')
		g = torch.sigmoid(self.lg(z))
		z = torch.zeros((z.shape[0], z.shape[1], z.shape[2], self.c)).to(x.get_device())
		for i in range(a.shape[1]):
			for j in range(b.shape[2]):
				ai = a[:, i, :]
				bj = b[:, :, j]
				z[:, i, j] = torch.sum(torch.mul(ai, bj), dim = -2)
		z = torch.mul(g, self.lz(self.ln2(z)))
		return z

class PSSM_Projector(nn.Module):
	'''
	model to project pssm data to s layers

	Author: Matthew Uryga
	'''
	def __init__(self, num_layers, c_m):
		super().__init__()
		layers = [nn.Linear(21, c_m) for i in range(num_layers)]
		self.layers = nn.ModuleList(layers)
		self.c_m = c_m
		self.num_layers = num_layers
	
	def forward(self, x):
		out = torch.zeros((x.shape[0], self.num_layers, x.shape[1], self.c_m)).to(x.get_device())
		# for each batch, apply a linear layer to pssm data
		for i in range(x.shape[0]):
			for j, l in enumerate(self.layers):
				out[i,j] = l(x[i])
		return out

class Input_Feature_Projector(nn.Module):
	'''
	projects the input features to c_2 features

	Author: Matthew Uryga
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

	Author: Matthew Uryga
	'''
	def __init__(self, c_2):
		super().__init__()
		self.l = nn.Linear(65, c_2)
	
	def forward(self, x):
		# pass through linear layer
		return self.l(x)

class Representation_Projector(nn.Module):
	def __init__(self, r, s, c_m, c_z):
		'''
		Takes in batch of sequences and evos, and 
		computes the outer-sum and the relative 
		positional encoding. The PSSM is ran through 
		s different linear projection layers to
		construct the MSA representation.

		Author Matthew Uryga
		'''
		super().__init__()
		self.r = r
		self.s = s
		self.c_m = c_m
		self.c_z = c_z
		self.pssm_projector = PSSM_Projector(s, c_m)
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
		all_res = torch.arange(L).to(seqs.get_device())
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

	Author: Matthew Uryga
	'''
	def __init__(self, c_m, c_z, c):
		super().__init__()
		self.msa_stack = MSA_Stack(c_m, c_z, heads = 4, dim_head = c)
		self.outer_product_mean = Outer_Product_Mean(c_m, c_z, c = c)
		self.triangular_mult_outgoing = Triangular_Multiplicative_Model('outgoing', c_z = c_z, c = c)
		self.triangular_mult_incoming = Triangular_Multiplicative_Model('incoming', c_z = c_z, c = c)
		self.pair_stack = Pair_Stack(c_z, heads = 4, dim_head = c)

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
	'''
	DEPRECATED

	Wrapper class for all the building blocks of the model.
	Takes care of the input embeddings, alphafold model pipeline,
	and finally dmat and angle predictions.

	Author: Matthew Uryga
	'''
	def __init__(self, r, s, c_m, c_z, c):
		super().__init__()
		self.rep_proj = Representation_Projector(r, s, c_m, c_z)
		self.evoformer_trunk = Evoformer_Trunk(c_m, c_z, c)
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

class IPA_Module(nn.Module):
	'''
	IPA module as outlined in alphafold2 paper

	Author: Matthew Uryga, Yu-Kai "Steven" Wang
	'''
	def __init__(self):
		super().__init__()
		pass
	
	def forward(self, x):
		pass

class Backbone_Update(nn.Module):
	def __init__(self, c_s):
		super().__init__()
		self.proj_down = nn.Linear(c_s, 6)
	
	def forward(self, x):
		b, r, c_s = x.shape
		x = self.proj_down(x)
		t = x[:, -3:]
		q = torch.ones((b, 4, r))
		q[:, 1:] = x[:, :3]
		q = q.div(torch.sqrt(1 + torch.square(q[:, 1]) + torch.square(q[:, 2]) + torch.square(q[:, 3])))
		r = quaternion_to_matrix(q)
		return r, t

class Structure_Module(nn.Module):
	'''
	Structure module as outlined in alphafold2 paper
	Takes pair rep and single rep to produce predictions for
	backbone frames and angles

	Author: Matthew Uryga
	'''
	def __init__(self, r, c_s, c_z, c = 64, N_layer = 8):
		super().__init__()
		self.N_layer = N_layer
		self.c = c
		self.dropout = nn.Dropout(p = 0.1)

		# layer norms for inputs
		self.ln_s_i = nn.LayerNorm(c_s)
		self.ln_z = nn.LayerNorm(c_z)

		# linear layer for single rep
		self.lin_s = nn.Linear(c_s, c_s)

		# ipa module and its layer norm
		self.ipa_module = IPA_Module()
		self.ln_ipa = nn.LayerNorm(c_s)

		# transition
		# ffn and layer norm for output of ipa module
		self.lin_s1 = nn.Linear(c_s, c_s)
		self.lin_s2 = nn.Linear(c_s, c_s)
		self.lin_s3 = nn.Linear(c_s, c_s)
		self.ln_s = nn.LayerNorm(c_s)

		# update backbone
		self.bb_update = BackboneUpdate(c_s)

		# predict sidechain and backbone torsion angles
		# linear projections to dim c
		self.lin_a1 = nn.Linear(c_s, c)
		self.lin_a2 = nn.Linear(c_s, c)

		# ffn 1 for a
		self.lin_a3 = nn.Linear(c_s, c)
		self.lin_a4 = nn.Linear(c_s, c)

		# ffn 2 for a
		self.lin_a5 = nn.Linear(c_s, c)
		self.lin_a6 = nn.Linear(c_s, c)

		# project down to dim 4 for torsion angles
		# 0, 1 represent phi, 2, 3 represent psi
		self.lin_a7 = nn.Linear(c, 4)

		self.loss_func = nn.MSELoss()


	def compute_fape(bb_r, bb_t, x, T_labels, x_labels, eps = 1e-4):
		# split labels
		bb_r_labels, bb_t_labels = T_labels

		# get dimensions
		B, I, _, _ = bb_r.shape[0]
		J = x.shape[1]

		# create x_ij matrices
		x_ij = torch.zeros((B, I, J, 3))
		x_ij_labels = torch.zeros((B, I, J, 3))
		for i in range(I):
			x_ij[:, i, :] = torch.matmul(bb_r[:, i], x) + bb_t[:, i]
			x_ij_labels[:, i, :] = torch.matmul(bb_r_labels[:, i], x) + bb_t_labels[:, i]

		# calculate d
		d = torch.sqrt(torch.mse_loss(x_ij, x_ij_labels, reduction = 'none') + eps)

		# calculate fape
		fape = 0.1 * torch.mean(torch.minimum(10, d))

		# return
		return fape


	def forward(self, z, s_i, a_labels):
		b, r, c_s = s_i.shape
		# apply layer norms to input
		s_i = self.ln_s_i(s_i)
		z = self.ln_z(z)

		# pass single rep through linear layer
		s = self.lin_s(s_i)

		# black hole initialization
		bb_r = torch.zeros((b, r, 3, 3))
		for i in range(3):
			bb_r[:, :, i, i] = 1
		bb_t = torch.zeros((b, r, 3))

		l_aux = []
		# loop over N_layers
		for l in range(self.N_layer):
			# pass through ipa module
			s = ipa_module(z, s, bb_r, bb_t) + s

			# apply layer norm and dropout
			s = self.ln_ipa(self.dropout(s))

			# transition
			# pass through ffn
			s_t = F.relu(self.lin_a1(s))
			s_t = F.relu(self.lin_a2(s_t))
			s = self.lin_a3(s_t) + s

			# apply layer norm and dropout
			s = self.ln_s(self.dropout(s))

			# update backbone
			new_r, new_t = self.bb_update(s)
			bb_r = torch.matmul(bb_r, new_r)
			bb_t = torch.add(bb_t, new_t)

			# torsion angle prediction
			a = self.lin_a1(s) + self.lin_a2(s_i)
			a = self.lin_a3(F.relu(self.lin_a4(F.relu(a)))) + a
			a = self.lin_a5(F.relu(self.lin_a6(F.relu(a)))) + a
			a = self.lin_a7(F.relu(a))

			# calculate torsion angle loss
			l_phi = torch.sqrt(torch.square(a[:, :, 0]) + torch.square(a[:, :, 1]))
			l_psi = torch.sqrt(torch.square(a[:, :, 2]) + torch.square(a[:, :, 3]))
			a = a[:, :, :2]/l_phi
			a = a[:, :, 2:]/l_psi
			L_torsion = self.loss_func(a, a_labels)
			L_anglenorm = torch.mean(torch.abs(l_phi - 1) + torch.abs(l_psi - 1))
			L_torsion = L_torsion + 0.02*L_anglenorm

			# calculate FAPE
			x = bb_t
			L_fape = self.compute_fape(bb_r, bb_t, x, T_labels, x_labels, eps = 1e-12)

			# sum fape and torsion loss for aux loss
			l_aux.append(L_fape + L_torsion)

		# mean of l_aux	
		l_aux = np.mean(l_aux)

		# final coordinate prediction
		R = torch.zeros((b, r, 3, 3))
		R[:, :, 0, 0] = 1
		R[:, :, 1, 1] = a[:, :, 0]
		R[:, :, 1,] =  a[:, :, 0]




class Alphafold2_Model(nn.Module):
	'''
	Module to wrap the entire alphafold2 model
	Includes input embeddings/projections, evoformer trunk, and IPA model

	Author: Matthew Uryga, Yu-Kai "Steven" Wang
	'''
	def __init__(self, r, s, c_m, c_z, c):
		super().__init__()
		self.rep_proj = Representation_Projector(r, s, c_m, c_z)
		self.evoformer_trunk = Evoformer_Trunk(c_m, c_z, c)
	
	def forward(self, seqs, evos):
		# pass input through projections
		prw_rep, msa_rep = self.rep_proj(seqs, evos)

		# pass representations through evoformer trunk
		prw_rep, msa_rep = self.evoformer_trunk(prw_rep, msa_rep)

		# pass updated representations through structure module

