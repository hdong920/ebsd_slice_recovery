# parts adapted from https://github.com/lucidrains/axial-attention

import torch
import torch.nn as nn 


class ChanLayerNorm3D(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class SelfAttention(nn.Module): # no output projection--it is redundant
    def __init__(self, dim, heads):
        super().__init__()
        self.dim_heads = dim // heads
        self.heads = heads
        
        self.to_qkv = nn.Linear(dim, 3 * dim, bias = False)
        
        self.attn = None
        
    def forward(self, x):
        # x is (B, S, D)
        B = x.shape[0]
        H = self.heads
        D = self.dim_heads
        queries, keys, values = self.to_qkv(x).chunk(3, dim=-1)
        
        queries = queries.reshape((B, -1, H, D)).transpose(1, 2).reshape((B*H, -1, D))
        keys = keys.reshape((B, -1, H, D)).transpose(1, 2).reshape((B*H, -1, D))
        values = values.reshape((B, -1, H, D)).transpose(1, 2).reshape((B*H, -1, D))
        
        dots = torch.einsum('bid,bjd->bij', queries, keys) * (D ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bjd->bid', dots, values)

        out = out.reshape((B, H, -1, D)).transpose(1, 2).reshape((B, -1, D))
        return out


class AxialAttentionBlock(nn.Module):
    def __init__(self, dim, num_dimensions, heads):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        
        self.attns = nn.ModuleList([SelfAttention(dim, heads) for _ in range(num_dimensions)])
        self.num_dims = num_dimensions
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        #(B, X, Y, Z, D)
        B = x.shape[0]
        D = x.shape[-1]
        for i in range(self.num_dims):
            dim_len = x.shape[i+1]
            x = x.transpose(i+1, -2)
            intermed_shape = x.shape
            x = x.reshape((-1, dim_len, D))
            x = self.attns[i](x)
            x = x.reshape(intermed_shape).transpose(i+1, -2)
        return self.out(x)

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2 
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x

class Axial3DTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout, input_shape, ff_kernel=1):
        super().__init__()
        
        self.depth = depth
        self.emb_cu = nn.Conv3d(3, dim, 1)
        self.pos_emb = AxialPositionalEmbedding(dim, input_shape)
        
        self.attns = []
        self.ffs = []
        for d in range(depth):
            self.attns.append(
                nn.Sequential(
                    AxialAttentionBlock(dim, len(input_shape), heads),
                    nn.Dropout(dropout)
                )
            )
            
            self.ffs.append(
                nn.Sequential(
                    nn.Conv3d(dim, dim * 4, ff_kernel, padding = ff_kernel//2),
                    nn.GELU(),
                    nn.Conv3d(dim * 4, dim, ff_kernel, padding = ff_kernel//2),
                    nn.Dropout(dropout)
                )
            )
        self.attns = nn.ModuleList(self.attns)
        self.ffs = nn.ModuleList(self.ffs)
        
        self.norms = nn.ModuleList([ChanLayerNorm3D(dim) for _ in range(2*depth)])
        
        self.head = nn.Linear(dim, 3)
        
        self.emb_drop = nn.Dropout(dropout)

    def forward(self, x):
        # (B, 3, X, Y, Z)
        x = self.emb_drop(self.pos_emb(self.emb_cu(x)))
        for l in range(self.depth):
            x = self.norms[2*l](x + self.attns[l](x.permute((0, 2, 3, 4, 1))).permute((0, 4, 1, 2, 3)))
            x = self.norms[(2*l) + 1](x + self.ffs[l](x))
        return self.head(x.permute((0, 2, 3, 4, 1))).permute((0, 4, 1, 2, 3))
    
