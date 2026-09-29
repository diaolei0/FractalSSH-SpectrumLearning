# -*- coding: utf-8 -*-
"""Torch-only neural-network building blocks for Fractal SSH v12.

This module deliberately has no SciPy, scikit-learn or matplotlib imports.
The PyTorch training process therefore never needs to import the operator-
construction stack used in STEP 1.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from v9_common import GLOBAL_FEATURE_DIM, MAX_B, NUM_BRANCHES


class MultiScaleCNN2D(nn.Module):
    def __init__(self,in_channels=4,branch_channels=48,d_model=128,dropout=.1):
        super().__init__()
        self.c3=nn.Conv2d(in_channels,branch_channels,3,padding=1)
        self.c5=nn.Conv2d(in_channels,branch_channels,5,padding=2)
        self.c7=nn.Conv2d(in_channels,branch_channels,7,padding=3)
        self.fuse=nn.Sequential(
            nn.Conv2d(branch_channels*3,d_model,1),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
    def forward(self,x):
        return self.fuse(torch.cat([
            F.relu(self.c3(x)),F.relu(self.c5(x)),F.relu(self.c7(x))
        ],dim=1))


class GeometryEncoder(nn.Module):
    def __init__(self,d_model=128,nhead=8,layers=3,ff=512,dropout=.1,cnn_channels=48):
        super().__init__()
        self.cnn=MultiScaleCNN2D(branch_channels=cnn_channels,d_model=d_model,dropout=dropout)
        self.gmlp=nn.Sequential(nn.Linear(GLOBAL_FEATURE_DIM,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.pos=nn.Parameter(torch.zeros(1,1+MAX_B*MAX_B,d_model))
        nn.init.normal_(self.pos,std=.02)
        layer=nn.TransformerEncoderLayer(
            d_model,nhead,ff,dropout,activation='gelu',batch_first=True,norm_first=True
        )
        self.enc=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d_model)
    def forward(self,grid,gf):
        f=self.cnn(grid)
        tok=f.flatten(2).transpose(1,2)
        g=self.gmlp(gf).unsqueeze(1)
        tok=torch.cat([g,tok],dim=1)
        tok=tok+self.pos[:,:tok.size(1)]
        valid=grid[:,1].flatten(1)>.5
        valid=torch.cat([
            torch.ones((grid.size(0),1),dtype=torch.bool,device=grid.device),valid
        ],dim=1)
        mask=~valid
        return self.norm(self.enc(tok,src_key_padding_mask=mask)),mask


class FourierQueryEmbedding(nn.Module):
    def __init__(self,d_model=128,freq=(1,2,4,8,16,32,64,128)):
        super().__init__()
        self.freq=tuple(freq)
        self.net=nn.Sequential(
            nn.Linear(1+2*len(freq),d_model),nn.GELU(),nn.Linear(d_model,d_model)
        )
    def forward(self,q):
        feat=[q]
        for f in self.freq:
            a=2*math.pi*float(f)*q
            feat += [torch.sin(a),torch.cos(a)]
        return self.net(torch.cat(feat,dim=-1))


class CrossBlock(nn.Module):
    def __init__(self,d_model=128,nhead=8,ff=512,dropout=.1):
        super().__init__()
        self.attn=nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=True)
        self.n1=nn.LayerNorm(d_model)
        self.n2=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(
            nn.Linear(d_model,ff),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(ff,d_model),nn.Dropout(dropout)
        )
    def forward(self,q,kv,kv_mask=None):
        x,_=self.attn(q,kv,kv,key_padding_mask=kv_mask,need_weights=False)
        q=self.n1(q+x)
        return self.n2(q+self.ff(q))


class LatentSpectrumBlock(nn.Module):
    """Global all-state communication at O(N*m), not O(N^2)."""
    def __init__(self,d_model=128,nhead=8,ff=512,dropout=.1,n_latents=32):
        super().__init__()
        self.latent=nn.Parameter(torch.randn(1,n_latents,d_model)*.02)
        self.latent_from_states=CrossBlock(d_model,nhead,ff,dropout)
        self.states_from_latent=CrossBlock(d_model,nhead,ff,dropout)
    def forward(self,states):
        lat=self.latent.expand(states.size(0),-1,-1)
        lat=self.latent_from_states(lat,states,None)
        return self.states_from_latent(states,lat,None),lat


class BranchSlotInteraction(nn.Module):
    """Nine parent-space slots conditioned on exact geometry-derived rank fractions."""
    def __init__(self,d_model=128,nhead=8,ff=512,dropout=.1):
        super().__init__()
        self.slot=nn.Parameter(torch.randn(1,NUM_BRANCHES,d_model)*.02)
        self.prior_embed=nn.Sequential(nn.Linear(1,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.slots_from_states=CrossBlock(d_model,nhead,ff,dropout)
        self.states_from_slots=CrossBlock(d_model,nhead,ff,dropout)
        self.slot_norm=nn.LayerNorm(d_model)
    def forward(self,states,prior,geom_global):
        B=states.size(0)
        slots=self.slot.expand(B,-1,-1)
        slots=slots+self.prior_embed(prior.unsqueeze(-1))+geom_global.unsqueeze(1)
        slots=self.slots_from_states(slots,states,None)
        slots=self.slot_norm(slots)
        states=self.states_from_slots(states,slots,None)
        return states,slots


class OrderedChiralEnergyDecoder(nn.Module):
    """Exactly non-decreasing and exactly chiral spectrum without post-hoc sorting.

    For an even spectrum length N=2h, the network predicts h non-negative
    positive-half spacings.  Zero spacings are representable exactly, so true
    zero modes and exact degeneracies are compatible with the decoder. Their
    cumulative sum defines
        0 <= E_h <= ... <= E_{N-1},
    and the negative half is its exact reversed negative partner.

    Both members of each chiral pair contribute to the spacing logit, so the
    energy loss remains aligned with the same state/query tokens used by the
    projector head.  This avoids the index/projector mismatch caused by sorting
    energies independently of topology outputs.
    """
    def __init__(self,d_model=128):
        super().__init__()
        self.gap_head=nn.Sequential(
            nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,1)
        )
    def forward(self,x):
        n=x.size(1)
        if n%2:
            raise RuntimeError(f'OrderedChiralEnergyDecoder requires an even spectrum length, got {n}')
        h=n//2
        raw=self.gap_head(x)[...,0]
        # Pair symmetric state tokens before converting to positive spacings.
        pair_raw=0.5*(raw[:,h:]+torch.flip(raw[:,:h],dims=[1]))
        # Degeneracy-capable non-negative spacing.  The shift gives an initial
        # scale comparable to softplus(0)=log(2), while the square can attain
        # an exact zero gap (unlike softplus). Division by h keeps the initial
        # spectrum O(1) for spectra from 10^2 to 10^4 states.
        gap_shift=math.sqrt(math.log(2.0))
        gaps=torch.square(pair_raw+gap_shift)/float(max(h,1))
        pos=torch.cumsum(gaps,dim=1)
        neg=-torch.flip(pos,dims=[1])
        return torch.cat([neg,pos],dim=1).unsqueeze(-1)
