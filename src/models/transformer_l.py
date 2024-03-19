# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
# import sys
# root = "/home/bounoua/work/pid/"
# sys.path.append(root)
from src.libs.util import concat_vect
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        #use_cfg_embedding = dropout_prob > 0
        #self.embedding_table = nn.Embedding(num_classes , hidden_size)
        #self.lin_layer = nn.Linear(hidden_size*num_classes,hidden_size)
        
        self.num_classes = num_classes
        
        self.mlp = nn.Sequential(
            nn.Linear(num_classes * 2 , hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        #self.dropout_prob = dropout_prob

    # def token_drop(self, labels, force_drop_ids=None):
    #     """
    #     Drops labels to enable classifier-free guidance.
    #     """
    #     if force_drop_ids is None:
    #         drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
    #     else:
    #         drop_ids = force_drop_ids == 1
    #     labels = torch.where(drop_ids, self.num_classes, labels)
    #     return labels

    def forward(self, labels):
        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     labels = self.token_drop(labels, force_drop_ids)
        label = torch.cat([labels.clip(0,1),( labels < 0 ).int()],axis =1)

        # print("mask_time_marg")
        # print(mask_time_marg.shape)
        # print(mask_time_marg [:1])
 
        
        return self.mlp(label.float())
        # embeddings = self.embedding_table(labels+1)
      
        # embeddings = self.lin_layer(embeddings.view(embeddings.shape[0],-1))
        # return embeddings

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
       
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
   
    
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayer_2(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ModDeEmbed(nn.Module):
    """ Mod to mod Embedding
    """
    def __init__(
            self,
            sizes = [],
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = sizes
        self.sizes = sizes
        
        norm_l = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.proj = nn.ModuleList([nn.Linear(embed_dim,size, bias=bias) for size in sizes ]) 
        self.norm = nn.ModuleList([norm_l for size in sizes ]) 
        

    def forward(self, x_mod, i = []):
        x_mod = x_mod.permute(1,0,2)
        if  len(i) == 0:
            i = np.arange(len(x_mod) )
        proj = np.array(self.proj)[i] 
        norm = np.array(self.norm)[i]
        # print("mod_dembed")
        # print(x_mod.shape)
        # x = [
        #     norm_x(x[idx]) for  idx,norm_x in enumerate(norm) 
        # ]
        
        x = [
            proj_x(x_mod[idx]) for  idx,proj_x in enumerate(proj) 
        ]
        
        
        
        return x



class ModEmbed(nn.Module):
    """ Mod to mod Embedding
    """

    def __init__(
            self,
            sizes = [],
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = sizes
        self.sizes = sizes
        norm_l = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.proj = nn.ModuleList([nn.Linear(size,embed_dim, bias=bias) for size in sizes ]) 
        self.norm = nn.ModuleList([norm_l for size in sizes ]) 
        

    def forward(self, x_mod, i = []):
        
        if  len(i) == 0:
            i = np.arange(len(x_mod) )
            
        proj = np.array(self.proj)[i] 
        norm = np.array(self.norm)[i]
       
        x = [
            proj_x(x_mod[idx]) for  idx,proj_x in enumerate(proj) 
        ]
        
        x = [
            norm_x(x[idx]) for  idx,norm_x in enumerate(norm) 
        ]
        
        return torch.stack(x).permute(1,0,2)



class ModEncode(nn.Module):
    """ Mod to mod Embedding
    """

    def __init__(
            self,
            sizes = [],
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = sizes
        self.sizes = sizes
        norm_l = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.proj = nn.ModuleList([nn.Linear(embed_dim,embed_dim, bias=bias) for size in sizes]) 
        self.norm = nn.ModuleList([norm_l for size in sizes ]) 
        
        

    def forward(self, x_mod ):
        x_mod = x_mod.permute(1,0,2)
        # if  len(i) == 0:
        #     i = np.arange(len(x_mod) ) 
             
        # proj = np.array(self.proj)[i] 
        # norm = np.array(self.norm)[i]
        # # print("enc")
        # # print(x_mod.shape)
        # x = [
        #     proj_x(x_mod[idx]) for  idx,proj_x in enumerate(proj) 
        # ]
        # x = [
        #     norm_x(x[idx]) for  idx,norm_x in enumerate(norm) 
        # ]
        x = x_mod.sum(dim=0)
        #print(x.shape)
        return x



class DiT_Enc(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        hidden_size=1152,
        mod_sizes = [],
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        variable_input=True,
        latent_size = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.mod_sizes =mod_sizes
        self.hidden_size =hidden_size
        self.variable_input = variable_input
        #self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        #self.x_embedder = nn.Linear(patch_size,hidden_size)
        #self.x_embedder = ModEmbed(sizes=mod_sizes,embed_dim=hidden_size,norm_layer=None)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer_2(hidden_size, hidden_size)
        #if latent_size!=None:
        self.encode_mod = ModEncode(embed_dim=hidden_size,sizes= mod_sizes)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def get_pos_embed(self,i):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, i)  
  
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    

    def forward(self, x, t,i=[]):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        #if  len(i) == 0:
        #    i = np.arange(len(x) )
  
        #pos_embed=self.get_pos_embed(np.array(i)).to(x[0].device)
 
        #x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        #x = self.x_embedder(x, i=i) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        
        c = t                                # (N, D)
       
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.encode_mod(x)
        #x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x



class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        hidden_size=1152,
        mod_sizes = [],
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        variable_input=True,
 
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mod_sizes =mod_sizes
        self.hidden_size =hidden_size
        self.variable_input = variable_input
        #self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        #self.x_embedder = nn.Linear(patch_size,hidden_size)
        self.mod_enc = DiT_Enc(hidden_size=hidden_size,
                               mod_sizes=mod_sizes,
                               depth= depth//2,
                               variable_input=variable_input,
                               latent_size=None,
                               num_heads=num_heads,
                               mlp_ratio=mlp_ratio
                               )
        self.x_embedder = ModEmbed(sizes=mod_sizes,embed_dim=hidden_size,norm_layer=nn.LayerNorm)
        self.t_embedder = TimestepEmbedder(hidden_size)
        #self.y_embedder = LabelEmbedder(len(mod_sizes), hidden_size)
    
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer_2(hidden_size, hidden_size)
        self.unembed_mod = ModDeEmbed(sizes=mod_sizes,embed_dim=hidden_size,norm_layer=None)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ))
        
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        for mod in self.x_embedder.proj:
            w = mod.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(mod.bias, 0)

        # Initialize label embedding table:
        #nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        for mod in self.unembed_mod.proj:
            w = mod.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(mod.bias, 0)
            
        
    def get_pos_embed(self,i):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, i)  
        return torch.from_numpy(pos_embed).float().unsqueeze(0)
     
    

    def forward(self, x#,x_c
                ,t,mask = None, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        #if  len(i) == 0:
        i_all = np.arange(len(x) )

        pos_embed=self.get_pos_embed(np.array(i_all)).to(x[0].device)

        x_all = self.x_embedder(x, i=i_all) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t).squeeze() 
        t_zeros = torch.zeros_like(t)
            
        if self.variable_input:
            i = [idx for idx,k in enumerate(mask[0]) if k>0] ## remove marginals form the set 
            i_c = [idx for idx,k in enumerate(mask[0]) if k==0] ## get conditonal idx 
            
            x = x_all[:,torch.from_numpy(i).long().to(x[0].device),:]
            
            if len(i_c)>0:
                x_c = x_all[:,torch.from_numpy(i_c).long().to(x[0].device),:]
                y = self.mod_enc(x_c,t=t_zeros,i=i_c)
            else:
               #  print("hey_no_cond")
                y = t_zeros
        else:

            # x = x_all[:,torch.from_numpy(i).long().to(x[0].device),:]
            
            mask = mask.view(mask.shape[0],mask.shape[1],1)
            
            x = ( mask>0).int() * x_all

            x_c = ( mask==0).int() * x_all
            
            y = self.mod_enc(x_c,t=t_zeros)
        
        # if len(i_c)>0:
        #     x_c = x_all[:,torch.from_numpy(i_c).long().to(x[0].device),:]
        #     y = self.mod_enc(x_c,t=t_zeros,i=i_c)
        # else:
        #   #  print("hey_no_cond")
        #     y = t_zeros
            
            
        c = t + y                                # (N, D)
        
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unembed_mod(x)

        return x




    def forward_mt(self,x, t, mask,std = None):
        """_summary_

        Args:
            dit (_type_): tx
            x (_type_): dictionnary(x)
            t (_type_): time (N,1)
            mask : N,M , M is nb mod
        Returns:
            _type_: interface to be used with soi multitime
        """
        
        data = [ x[k] for idx, k in enumerate(x.keys() )  ]
     
        out = self.forward( x= data, 
                            mask= mask,
                            t = t)
        #return fill_missing_and_std(out,i,x,std)
        # print("out")
        # print(len(out))
        # print(torch.stack(out).shape)
        return concat_vect( fill_missing_and_std(out,np.arange(len(x)),x,std ) )


def fill_missing_and_std(out, i , origin_data,std=None):
    k = 0
    out_f={}
    for idx,key in enumerate( origin_data.keys()):
        if idx in i:
            out_f[key] = out[k]
            if std!=None:
                out_f[key] =out_f[key]/std
            k+=1
        else:
            out_f[key] = torch.zeros_like(origin_data[key])
    return out_f
#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_SS(patch_size = 1,nb_mod = 5,depth=2,sizes=[],**kwargs):
    return DiT(depth=depth, hidden_size=384,sizes= sizes, patch_size=patch_size, num_heads=6,input_size=nb_mod,in_channels=1, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT_SS':DiT_SS
}

if __name__ =="__main__":
    m = DiT(depth=1,
            hidden_size=384,
            mod_sizes= [5,10,15], 
            num_heads=6)
  
    #print( sum(p.numel() for p in m.parameters() if p.requires_grad) )
    
    
    # out = m( { "x0":torch.randn(32,5),
    #           "x1":torch.randn(32,10),
    #           "x2":torch.randn(32,15)}, 
    #         t =torch.randn(32,1), 
    #         y = torch.zeros(32,3).long() ,
    #         i = [0,1,2]
            
    #         )
    out =m.forward_mt(x ={ "x0":torch.randn(32,5),
              "x1":torch.randn(32,10),
              "x2":torch.randn(32,15)},
                 t= torch.randn(32,1),
                 mask=torch.tensor([[1,0,0] ])
                 )
    print("=================output=================")
    for o in out.keys():
        print("o.shape")
        print(out[o].shape)
