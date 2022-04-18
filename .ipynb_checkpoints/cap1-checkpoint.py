#%%
# # Capstone 1

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as du
import torch.optim as optim

from tqdm import tqdm
import sys
from einops import rearrange

import sidechainnet

#%%
# Model

#%%
class MHSA(nn.Module):
    
    def __init__(self, c_m, c_z, heads=8, dim_head=None, bias=True):
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
        x = MSA
        bias_rep = Pair bias
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
    

#%%
class MSA_Stack(nn.Module):
    def __init__(self, batch_size, c_m, c_z, heads=8, dim_head=None):
        super().__init__()
        # batches of row wise MHSA
        self.row_MHSA = nn.ModuleList([MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=True, dim_head=None) for i in range(batch_size)])
        # batches of col wise MHSA
        self.col_MHSA = nn.ModuleList([MHSA(c_m=c_m, c_z=c_z, heads=heads, bias=False, dim_head=None) for i in range(batch_size)])
        # transition MLP
        self.fc1 = nn.Linear(c_m, 4 * c_m)
        self.fc2 = nn.Linear(4 * c_m, c_m)
        
    def forward(self, x, bias_rep):
        res = torch.empty(x.shape)
        # row wise gated self-attention with pair bias
        for i, mhsa in enumerate(self.row_MHSA):
            res[i] = mhsa(x[i], bias_rep[i])
        x += res # add residuals
        
        # column wise gated self-attention
        x_trans = rearrange(x, 'b i j k -> b j i k')
        for i, mhsa in enumerate(self.col_MHSA):
            res[i] = rearrange(mhsa(x_trans[i]), 'i j k -> j i k')
        x += res # add residuals
        
        # transiion
        r = F.relu(self.fc1(x))
        r = self.fc2(r) + x
        
        return r

#%%
class Pair_Stack(nn.Module):
    def __init__(self, batch_size, c_z, heads=8, dim_head=None):
        super().__init__()
        # batches of row wise MHSA
        self.start_MHSA = nn.ModuleList([MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head) for i in range(batch_size)])
        # batches of col wise MHSA
        self.end_MHSA = nn.ModuleList([MHSA(c_m=c_z, c_z=c_z, heads=heads, bias=True, dim_head=dim_head) for i in range(batch_size)])
        # transition MLP
        self.fc1 = nn.Linear(c_z, 4 * c_z)
        self.fc2 = nn.Linear(4 * c_z, c_z)
        
    def forward(self, x):
        res = torch.empty(x.shape)
        # row wise gated self-attention with pair bias
        for i, mhsa in enumerate(self.start_MHSA):
            res[i] = mhsa(x[i], x[i])
        x += res # add residuals
        
        # column wise gated self-attention
        x_trans = rearrange(x, 'b i j k -> b j i k')
        for i, mhsa in enumerate(self.end_MHSA):
            # res[i] = mhsa(x_trans[i], x_trans[i])
            res[i] = rearrange(mhsa(x_trans[i], x_trans[i]), 'i j k -> j i k')
        x += res # add residuals
        
        # transiion
        r = F.relu(self.fc1(x))
        r = self.fc2(r) + x
        
        return r

#%%
class Outer_Product_Mean(nn.Module):
    
    def __init__(self, c_m, c_z, c=32):
        super().__init__()
        # linear projections
        self.fc1 = nn.Linear(c_m, c)
        self.fc2 = nn.Linear(c**2, c_z)
        self.flatten = nn.Flatten()
        self.c = c
        self.c_z = c_z
        
    def forward(self, x):
        '''
        x: B x S x R x C
        res: B x R x R x C
        '''
        # results
        res = torch.empty(x.shape[0], x.shape[-2], x.shape[-2], self.c_z)
        
        # project in_c to out_c
        x = self.fc1(x)
        
        # loop over R
        for i in range(x.shape[-2]):
            for j in range(x.shape[-2]):
                mean_s = torch.mean(torch.einsum('bij,bik->bijk', [x[:, :, i, :], x[:, :, j, :]]), dim=1)
                # print(mean_s.shape)
                # project B x C x C to B x C_z
                mean_s = self.flatten(mean_s)
                mean_s = self.fc2(mean_s)
                # print(mean_s.shape)
                # print(res[:, i, j, :].shape)
                res[:, i, j, :] = mean_s
        
        return res
                

#%%
S = 16
R = 256
C_m = 256
C_z = 128
B = 4
C = 16

src_test = torch.rand((B, S, R, C_m))
rrc_test = torch.rand((B, R, R, C_z))

#%%
# Testing

#%%
mhsa_stack = MSA_Stack(batch_size=B, c_m=C_m, c_z=C_z, heads=4, dim_head=C)
mhsa_stack(x=src_test, bias_rep=rrc_test).shape

#%%
opm = Outer_Product_Mean(c_m=C_m, c_z=C_z, c=C)
opm(src_test).shape

#%%
pair_stack = Pair_Stack(batch_size=B, c_z=C_z, heads=4, dim_head=C)
pair_stack(x=rrc_test).shape

#%%
# playground

#%%


