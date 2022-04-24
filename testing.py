#%%
# # Testing

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from tqdm import tqdm

#%%
class IPA(nn.Module):
	
	def __init__(self, c_m, c_z, heads=12, dim_head=None, n_qp=4, n_pv=8):
		'''
		dim_head: channel C
		'''
		super().__init__()
		
		# constants
		self.w_c = (2 / (9 * n_qp)) ** -0.5
		self.w_l = (1 / 3) ** -0.5
		self.n_qp = n_qp
		self.n_pv = n_pv
		
		# single rep attention layers
		self.heads = heads
		self.dim_head = (int(c_m / heads)) if dim_head is None else dim_head
		_dim = self.dim_head * heads
		self.to_qvk = nn.Linear(c_m, _dim * 3, bias=False)
		self.W_0 = nn.Linear(_dim, c_m, bias=False)
		self.to_qk = nn.Linear(c_m, (n_qp * heads *3) * 2, bias=False)
		self.W_0 = nn.Linear(_dim, c_m, bias=False)
		self.W_1 = nn.Linear(heads * c_z, c_m, bias=False)
		self.W_2 = nn.Linear(heads * n_pv * 3, c_m)
		self.gamma = nn.Parameter(torch.rand(1))
		self.to_v = nn.Linear(c_m, (n_pv * heads * 3), bias=False)
		
		# pair_rep layers
		self.fc1 = nn.Linear(c_z, heads)

	def forward(self, pair_rep, sing_rep, bbr, bbt):
		'''
		bbr: rotational matrix (B x R x 3 x 3)
		bbt: translatoin matrix (B x R x 3)
		'''
		
		# pair_rep to pair_bias
		pair_bias = self.fc1(pair_rep)
		pair_bias = rearrange(pair_bias, 'b i j h -> b h i j')
		print(f'pair bias shape = {pair_bias.shape}')
		
		### SINGLE REP SQR ATTENTION
		
		# get q and v for attention training (B x P x R x H x 3)
		qk = self.to_qk(sing_rep)
		print(f'qk shape = {qk.shape}')
		gq, gk = tuple(rearrange(qk, 'b r (d k p a) -> k b p r d a', k=2, a=3, p=self.n_qp))
		print(f'qk shape = {gq.shape}')
		gv = rearrange(self.to_v(sing_rep), 'b r (d p a) -> b p r d a', a=3, p=self.n_pv)
		print(f'gv shape = {gv.shape}')
		
		### SINGLE REP DOT ATTENTION
		
		# get q, v, k matrices for attention training (B x H x R x C)
		qkv = self.to_qvk(sing_rep)
		print(f'qkv shape = {qkv.shape}')
		rq, rk, rv = tuple(rearrange(qkv, 'b r (d k h) -> k b h r d', k=3, h=self.heads))
		print(f'qkv shape = {rq.shape}')
	
		# dot product attention (B x H x R x R)
		dot_prod_aff = torch.einsum('b h i d , b h j d -> b h i j', rq, rk) * (self.dim_head ** -0.5)
		print(f'dot_prod_aff shape = {dot_prod_aff.shape}')
		
		# square dist attention
		Tq = torch.einsum('b p r h a , b r a k -> b p h r k', gq, bbr) + bbt
		Tk = -1 * torch.einsum('b p r h a , b r a k -> b p h r k', gk, bbr) + bbt
		print(f'Tq shape = {Tq.shape}')
		# dot product
		sqr_dist_aff = torch.einsum('b p h i k , b p h j k -> b p h i j k', Tq, Tk)
		# norm square
		sqr_dist_aff = torch.sum(torch.square(torch.norm(sqr_dist_aff, dim=-1)), dim=1) # b h r r
		print(f'norm_sqr shape = {sqr_dist_aff.shape}')
		# multiply head weight
		head_w = (F.softplus(self.gamma.repeat(self.heads)) * self.w_c) / 2
		print(f'head_w shape = {head_w.shape}')
		print(f'sqr_dist_aff shape = {sqr_dist_aff.shape}')        
		sqr_dist_aff = rearrange(rearrange(sqr_dist_aff, 'b h i j -> b i j h') * head_w, 'b i j h -> b h i j')
		print(f'sqr_dist_aff shape = {sqr_dist_aff.shape}')
		
		# sum attentions with bias then softmax (B x H x R x R)
		attentions = pair_bias + dot_prod_aff + sqr_dist_aff
		attentions = torch.softmax(self.w_l * attentions, dim=-1)
		print(f'attentions after softmax shape = {attentions.shape}')
		
		
		# dot with pair values (top) 
		# B Rq H R x B Rq R C => B R H C
		top = torch.einsum('b h i j , b h j d -> b h i d', rearrange(attentions, 'b h i j -> b i h j'), pair_rep) # B H Rq R x B C R R -> B C R R
		# concat heads
		top = rearrange(top, 'b r h c -> b r (h c)')
		print(f'top shape = {top.shape}')
		# transform back to initial dimension
		top = self.W_1(top)
		
		# dot with value points (bot)
		# B H Rq Rv x B P Rv H 3 => B R1 H P 3
		Tv = torch.einsum('b p r h a , b r a k -> b p h r k', gv, bbr) + bbt
		print(f'Tv shape = {Tv.shape}')
		bot = torch.einsum('b h i j , b p h j a -> b i h p a', attentions, Tv)
		# invert backbone frames
		bbr_inv = torch.linalg.inv(bbr)
		# affine transform
		bot = torch.einsum('b r h p a , b r a k -> b h p r k', bot, bbr_inv) + bbt
		# concat heads
		bot = rearrange(bot, 'b h p r a -> b r (h p a)')
		# transform back to initial dimension
		bot = self.W_2(bot)
		print(f'bot shape = {bot.shape}')
		
		# dot with matrix v (mid)
		out = torch.einsum('b h i j , b h j d -> b h i d', attentions, rv)        
		# concat heads
		out = rearrange(out, "b h t d -> b t (h d)")
		# transform back to initial dimension
		out = self.W_0(out)
		# sum top, mid, bottom
		out = out + top
		print(f'output shape = {out.shape}')
		
		return out

#%%
# B H Rv Rq x B H Rv P 3
a = torch.rand(1,12,64,64)
b = torch.rand(1,12,64,8,3)
torch.einsum('b h i j , b h i p a -> b h j p a', a, b).shape

#%%
# Testing

#%%
B = 1
R = 64
C_m = 128
C_z = 64
H = 12
C = 16
N_qp = 4
N_pv = 8

#%%
pair_rep = torch.rand(B, R, R, C_z)
sing_rep = torch.rand(B, R, C_m)
bbr = torch.rand(B, R, 3, 3)
bbt = torch.rand(B, R, 3)

#%%
ipa = IPA(C_m, C_z, heads=H, dim_head=C)
ipa(pair_rep, sing_rep, bbr, bbt).shape

#%%
sing_rep.shape

#%%


