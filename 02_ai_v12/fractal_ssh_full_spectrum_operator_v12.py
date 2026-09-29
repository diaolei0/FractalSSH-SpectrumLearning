# -*- coding: utf-8 -*-
"""Fractal SSH v12 PAPER — K1/K2/K3 operator-hierarchy, interference-aware, gauge-factorized full-spectrum network.

Primary methodological points
-----------------------------
1. Geometry-derived K1=PVP, K2=-PVRVP, Hermitian K3, Keff3, eta2 and path/order hierarchy descriptors are explicit inputs.
2. The projector head factorizes zero-manifold mass from OC/IC-II/BULK-0 conditional parentage. Exact-degeneracy supervision is cluster/projector-trace based; arbitrary
   eigenvector gauge choices are not used as primary state-level targets.
3. The energy decoder is intrinsically ordered and chiral.  It does not sort
   an independently predicted energy vector, so energy/projector state indices
   remain aligned.
4. Operator construction (SciPy) is precomputed in a separate process.  The
   training process imports no SciPy/scikit-learn/matplotlib.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from v9_common import (
    BRANCH_LABELS, NUM_BRANCHES, ZERO_SECTOR_INDICES, ISOLATED_INDICES,
    ISOLATED_LABELS, CHIRAL_PARTNER,
    set_seed, seed_worker, load_available_families, print_family_stats,
    stratified_family_split, save_split, load_split, fit_energy_scaler,
    fit_global_scaler, save_scalers, load_scalers, raw_global_features,
    theory_graph_features_from_generator, make_geometry_grid, random_d4,
    make_family_sampler, fit_isolated_class_weights, classification_metrics_from_confusion,
)
from model_blocks_v12 import (
    GeometryEncoder, FourierQueryEmbedding, CrossBlock, LatentSpectrumBlock,
    BranchSlotInteraction, OrderedChiralEnergyDecoder,
)
from operator_cache_v12 import (
    load_operator_cache, load_operator_meta, validate_cache_audit,
    OP_TOKEN_DIM, OP_NUM_TOKENS, OP_GLOBAL_DIM,
    G_DIMFRAC, G_MEAN_K1, G_MEAN_K2, G_MEAN_K3, G_MEAN_KEFF,
    G_ALIGN12, G_ALIGN23, G_D01, G_D02, G_D12,
    G_TRACE_OC, G_TRACE_ICII, G_TRACE_B0, G_ETA2, G_CANCEL2,
    G_PATH2_PRESENT, G_PATH2_COUNT, G_PATH2_ABS, G_ORDER_ONEHOT,
)

VERSION='v12-paper-operator-hierarchy-2026-09-13-r1'
MAP9TO6={k:i for i,k in enumerate(ISOLATED_INDICES)}
ISO_PARTNER_9=[int(CHIRAL_PARTNER[i]) for i in ISOLATED_INDICES]
ZLIST=list(ZERO_SECTOR_INDICES)
ILIST=list(ISOLATED_INDICES)
ISO_PARTNER_6=[1,0,3,2,5,4]


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def numpy_r2(y_true,y_pred):
    y=np.asarray(y_true,dtype=np.float64);p=np.asarray(y_pred,dtype=np.float64)
    den=float(np.square(y-y.mean()).sum());num=float(np.square(y-p).sum())
    return float('nan') if den<=1e-30 else 1.0-num/den


def resolve_device(spec):
    """Return a canonical torch.device with an explicit CUDA index.

    torch.device('cuda') and torch.device('cuda:0') address the same GPU in the
    usual single-GPU case but compare unequal as Python objects.  All v12
    runtime paths therefore canonicalize the device before moving modules,
    buffers, batches, checkpoints or AMP state.
    """
    requested=torch.device(str(spec))
    if requested.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device requested ({spec}) but torch.cuda.is_available() is False')
        idx=torch.cuda.current_device() if requested.index is None else int(requested.index)
        ndev=torch.cuda.device_count()
        if idx<0 or idx>=ndev:
            raise RuntimeError(f'CUDA device index {idx} is invalid; visible CUDA device count={ndev}')
        torch.cuda.set_device(idx)
        return torch.device('cuda',idx)
    if requested.type=='cpu':
        return torch.device('cpu')
    return requested


def assert_runtime_devices(model,criterion,device,batch=None):
    """Fail only on a real placement mismatch, never on cuda vs cuda:0 aliases."""
    device=resolve_device(device)
    mdev=next(model.parameters()).device
    cdev=criterion.isolated_class_weights.device
    if mdev!=device:
        raise RuntimeError(f'Model device mismatch: model={mdev}, expected={device}')
    if cdev!=device:
        raise RuntimeError(f'Loss buffer device mismatch: buffer={cdev}, expected={device}')
    if batch is not None:
        bad={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v) and v.device!=device}
        if bad:
            raise RuntimeError(f'Batch device mismatch: expected={device}, found={bad}')
    return device


class FullSpectrumOperatorDataset(Dataset):
    def __init__(self,records,energy_scaler,global_scaler,operator_cache,training=False,d4_augment=False,seed=20260912):
        self.records=list(records);self.es=energy_scaler;self.gs=global_scaler
        self.training=bool(training);self.d4_augment=bool(d4_augment);self.seed=int(seed);self.epoch=0
        self.operator_cache=Path(operator_cache);self.cache={}
        for r in self.records:
            with np.load(r.mask_path,allow_pickle=False) as z:M=z['M'].astype(np.uint8)
            prior,ref,_,meta=theory_graph_features_from_generator(M)
            gf=raw_global_features(r,M).astype(np.float32)
            optok,opglob=load_operator_cache(self.operator_cache,r.family_id)
            with np.load(r.state_path,allow_pickle=False) as z:
                need={'eigenvalues','q_state','branch_scores','branch_index','cluster_id','cluster_size','is_degenerate'}
                miss=need-set(z.files)
                if miss: raise RuntimeError(f'{r.family_id}: missing database fields {sorted(miss)}')
                if len(z['eigenvalues'])!=int(round(float(ref.sum()))):
                    raise RuntimeError(f'{r.family_id}: state count != exact strong-coupling rank sum')
                if 'ref_dims' in z.files and not np.array_equal(z['ref_dims'].astype(np.int64),ref.astype(np.int64)):
                    raise RuntimeError(f'{r.family_id}: stored ref_dims disagree with geometry-derived ranks')
                cid=z['cluster_id'].astype(np.int64);csize=z['cluster_size'].astype(np.int64);deg=z['is_degenerate'].astype(bool);nst=len(z['eigenvalues'])
                if cid.size!=nst or cid.min()<0 or cid.max()>=nst:
                    raise RuntimeError(f'{r.family_id}: cluster_id must lie in [0,Nstate-1] for vectorized aggregation')
                if not np.array_equal(deg,csize>=2):
                    raise RuntimeError(f'{r.family_id}: is_degenerate disagrees with cluster_size>=2')
            self.cache[r.family_id]=(M,prior.astype(np.float32),ref.astype(np.float32),gf,optok,opglob,meta)
    def set_epoch(self,e):self.epoch=int(e)
    def __len__(self):return len(self.records)
    def __getitem__(self,idx):
        r=self.records[idx];M,prior,ref,gf,optok,opglob,_=self.cache[r.family_id];Muse=M
        if self.training and self.d4_augment:
            rng=np.random.default_rng(self.seed+1000003*self.epoch+7919*idx);Muse=random_d4(M,rng)
        with np.load(r.state_path,allow_pickle=False) as z:
            E=z['eigenvalues'].astype(np.float64);q=z['q_state'].astype(np.float64)
            scores=z['branch_scores'].astype(np.float64);hard=z['branch_index'].astype(np.int64)
            cid=z['cluster_id'].astype(np.int64);csize=z['cluster_size'].astype(np.float64);deg=z['is_degenerate'].astype(bool)
        scores=np.clip(scores,0,None);scores/=np.clip(scores.sum(1,keepdims=True),1e-12,None)
        return {'family_id':r.family_id,'grid':torch.from_numpy(make_geometry_grid(Muse,r.b)),
                'global_features':torch.from_numpy(self.gs.transform_np(gf).astype(np.float32)),
                'topology_prior':torch.from_numpy(prior),'ref_dims':torch.from_numpy(ref),
                'operator_tokens':torch.from_numpy(optok),'operator_global':torch.from_numpy(opglob),
                'q':torch.tensor(q,dtype=torch.float32).unsqueeze(-1),
                'energy':torch.tensor(self.es.transform_np(E),dtype=torch.float32).unsqueeze(-1),
                'branch_scores':torch.tensor(scores,dtype=torch.float32),
                'legacy_hard_branch':torch.tensor(hard,dtype=torch.long),
                'cluster_id':torch.tensor(cid,dtype=torch.long),'cluster_size':torch.tensor(csize,dtype=torch.float32),
                'degenerate_mask':torch.tensor(deg,dtype=torch.bool)}


def make_loader(records,es,gs,training,args):
    ds=FullSpectrumOperatorDataset(records,es,gs,args.operator_cache,training=training,d4_augment=(training and not args.no_d4_augment),seed=args.seed+(0 if training else 1))
    sampler=make_family_sampler(records,args.balance_strength) if training else None
    gen=torch.Generator();gen.manual_seed(args.seed)
    loader=DataLoader(ds,batch_size=1,sampler=sampler,shuffle=False,num_workers=args.num_workers,pin_memory=torch.cuda.is_available(),worker_init_fn=seed_worker,generator=gen)
    return ds,loader


class OperatorEncoder(nn.Module):
    def __init__(self,d_model=128,nhead=8,layers=2,ff=512,dropout=.1):
        super().__init__();self.token_in=nn.Sequential(nn.Linear(OP_TOKEN_DIM,d_model),nn.GELU(),nn.Linear(d_model,d_model));self.global_in=nn.Sequential(nn.Linear(OP_GLOBAL_DIM,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.pos=nn.Parameter(torch.randn(1,1+OP_NUM_TOKENS,d_model)*.02)
        layer=nn.TransformerEncoderLayer(d_model,nhead,ff,dropout,activation='gelu',batch_first=True,norm_first=True);self.enc=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False);self.norm=nn.LayerNorm(d_model)
    def forward(self,tok,glob):
        t=self.token_in(tok);g=self.global_in(glob).unsqueeze(1);x=torch.cat([g,t],1)+self.pos[:,:1+t.size(1)];return self.norm(self.enc(x))


class OperatorInformedFullSpectrumModel(nn.Module):
    """Full-spectrum model with a physics-factorized projector head.

    The nine projector weights are represented as
        C_zero = m0 * softmax(zero_parent_logits),
        C_iso  = (1-m0) * softmax(isolated_logits),
    where m0 is the total OC+IC-II+BULK-0 mass.  This exactly separates the
    two physically different questions: whether a state belongs to the zero
    manifold, and its conditional parentage inside that manifold.
    """
    def __init__(self,d_model=128,nhead=8,encoder_layers=3,operator_layers=2,query_layers=2,spectral_layers=3,n_latents=40,dim_feedforward=512,dropout=.1,cnn_channels=48,operator_mode='full',projector_head_mode='factorized'):
        super().__init__();ff=dim_feedforward
        self.operator_mode=str(operator_mode);self.projector_head_mode=str(projector_head_mode)
        if self.operator_mode not in {'none','k1','k1k2','k1k2k3','full_no_hierarchy','full'}:raise ValueError(self.operator_mode)
        if self.projector_head_mode not in {'flat','factorized'}:raise ValueError(self.projector_head_mode)
        self.geometry=GeometryEncoder(d_model,nhead,encoder_layers,ff,dropout,cnn_channels)
        self.operator=OperatorEncoder(d_model,nhead,operator_layers,ff,dropout)
        self.qembed=FourierQueryEmbedding(d_model)
        self.geo_query=nn.ModuleList([CrossBlock(d_model,nhead,ff,dropout) for _ in range(query_layers)])
        self.op_query=nn.ModuleList([CrossBlock(d_model,nhead,ff,dropout) for _ in range(query_layers)])
        self.spectrum_blocks=nn.ModuleList([LatentSpectrumBlock(d_model,nhead,ff,dropout,n_latents) for _ in range(spectral_layers)])
        self.branch_slots=BranchSlotInteraction(d_model,nhead,ff,dropout)
        self.context_fuse=nn.Sequential(nn.LayerNorm(d_model*2),nn.Linear(d_model*2,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.energy_decoder=OrderedChiralEnergyDecoder(d_model)
        # Flat head is retained only for controlled ablation.
        self.flat_projector_head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,NUM_BRANCHES))
        # Main v12 factorized head.
        self.hierarchy_direct=nn.Sequential(nn.LayerNorm(OP_GLOBAL_DIM),nn.Linear(OP_GLOBAL_DIM,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.zero_context=nn.Sequential(nn.LayerNorm(d_model*3),nn.Linear(d_model*3,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.zero_mass_head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model//2),nn.GELU(),nn.Linear(d_model//2,1))
        self.zero_parent_head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,3))
        self.isolated_head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,6))

    def _mask_operator(self,tok,glob):
        """Controlled physics ablations with a fixed tensor shape.

        Token layout: K1[0:6], K2[6:12], K3[12:18], Keff3[18:24].
        `full_no_hierarchy` keeps all operator matrices but removes eta2/path/
        first-order descriptors, isolating the value of the hierarchy summary.
        """
        if self.operator_mode=='full':return tok,glob
        if self.operator_mode=='none':return torch.zeros_like(tok),torch.zeros_like(glob)
        t=tok.clone();g=torch.zeros_like(glob)
        # Shared exact rank fractions and graph distances are retained whenever
        # an operator path is enabled; these were already part of v10/v11.
        g[:,0:3]=glob[:,0:3]
        g[:,G_D01:G_D12+1]=glob[:,G_D01:G_D12+1]
        if self.operator_mode=='k1':
            t[:,6:]=0;g[:,G_MEAN_K1]=glob[:,G_MEAN_K1]
        elif self.operator_mode=='k1k2':
            t[:,12:]=0
            g[:,G_MEAN_K1:G_MEAN_K2+1]=glob[:,G_MEAN_K1:G_MEAN_K2+1]
            g[:,G_ALIGN12]=glob[:,G_ALIGN12]
        elif self.operator_mode=='k1k2k3':
            t[:,18:]=0
            g[:,G_MEAN_K1:G_MEAN_K3+1]=glob[:,G_MEAN_K1:G_MEAN_K3+1]
            g[:,G_ALIGN12:G_ALIGN23+1]=glob[:,G_ALIGN12:G_ALIGN23+1]
        elif self.operator_mode=='full_no_hierarchy':
            # All K1/K2/K3/Keff tokens and base globals, but intentionally hide
            # the continuous interference efficiency and order class.
            g[:,:G_ETA2]=glob[:,:G_ETA2]
        return t,g

    def _factorized_logprob(self,px,op_global_token,op_global_raw):
        op=op_global_token.unsqueeze(1).expand(-1,px.size(1),-1)
        direct=self.hierarchy_direct(op_global_raw).unsqueeze(1).expand(-1,px.size(1),-1)
        zh=self.zero_context(torch.cat([px,op,direct],dim=-1))
        zm0=torch.sigmoid(self.zero_mass_head(zh)[...,0])
        zp0=torch.softmax(self.zero_parent_head(zh),dim=-1)
        ip0=torch.softmax(self.isolated_head(px),dim=-1)
        # Exact projector-level chiral construction.  The three zero sectors are
        # self-partners, while the isolated six pair as EDI<->EDII,
        # ICI<->ICIII, BULK-L<->BULK-U.  Averaging probabilities (not logits)
        # preserves normalization exactly and removes redundant +/-E freedom.
        zm=0.5*(zm0+torch.flip(zm0,dims=[1]))
        zp=0.5*(zp0+torch.flip(zp0,dims=[1]))
        ip_partner=torch.flip(ip0,dims=[1])[...,ISO_PARTNER_6]
        ip=0.5*(ip0+ip_partner)
        one=1.0-zm
        prob=torch.stack([
            zm*zp[...,0], one*ip[...,0], one*ip[...,1], one*ip[...,2],
            zm*zp[...,1], one*ip[...,3], one*ip[...,4], zm*zp[...,2], one*ip[...,5]
        ],dim=-1)
        return torch.log(prob.clamp_min(1e-9))

    def forward(self,grid,gf,prior,q,operator_tokens,operator_global):
        mem,gmask=self.geometry(grid,gf)
        operator_tokens,operator_global=self._mask_operator(operator_tokens,operator_global)
        if self.operator_mode=='none':
            # No geometry-dependent information may enter through the ablated operator path.
            omem=mem.new_zeros((mem.size(0),1+OP_NUM_TOKENS,mem.size(-1)))
        else:
            omem=self.operator(operator_tokens,operator_global)
        x=self.qembed(q)
        for bg,bo in zip(self.geo_query,self.op_query):
            x=bg(x,mem,gmask)
            if self.operator_mode!='none':x=bo(x,omem,None)
        for block in self.spectrum_blocks:x,_=block(x)
        pred_E=self.energy_decoder(x)
        fused=self.context_fuse(torch.cat([mem[:,0],omem[:,0]],-1))
        px,slots=self.branch_slots(x,prior,fused)
        if self.projector_head_mode=='factorized':plog=self._factorized_logprob(px,omem[:,0],operator_global)
        else:plog=self.flat_projector_head(px)
        return pred_E,plog,slots

def zero_parentage(prob,eps=1e-8):
    z=prob[...,ZLIST];mass=z.sum(-1,keepdim=True);return z/mass.clamp_min(eps),mass[...,0]


def isolated_target6(hard9):
    out=torch.full_like(hard9,-100)
    for k9,k6 in MAP9TO6.items():out=torch.where(hard9==k9,torch.as_tensor(k6,device=hard9.device),out)
    return out


def state_supervision_mask(true_scores,deg,purity_threshold):
    """State-level targets are used for nondegenerate states and only very pure degenerate states.

    This preserves useful high-confidence supervision without treating an arbitrary basis inside a
    genuinely mixed degenerate subspace as a unique physical truth.
    """
    purity=true_scores.max(dim=-1).values
    return (~deg) | (purity>=float(purity_threshold))


def vectorized_cluster_terms(pprob,true_scores,cluster_id,deg,eps=1e-8):
    """Gauge-invariant cluster losses with ONE vectorized GPU aggregation.

    No Python loop over clusters, no .item()/float()/int() inside a cluster loop, and no
    GPU->CPU synchronization per cluster.  cluster_id is assumed to be an integer state-cluster
    label in [0,N-1], which is audited when the dataset is loaded.

    Returns
    -------
    normalized_trace_mae:
        |Tr_D(P_pred)-Tr_D(P_true)| / multiplicity, averaged over degenerate clusters and branches.
    zero_parentage_mae:
        Zero-sector parentage from aggregate cluster traces, weighted by the true zero-sector
        trace per state.
    zero_mass_mae:
        Absolute zero-sector trace error / multiplicity, averaged over degenerate clusters.
    """
    # Accumulate in float32 even under AMP; projector traces can sum many states.
    pv=pprob.float(); tv=true_scores.float()
    B,N,C=pv.shape
    offsets=(torch.arange(B,device=cluster_id.device,dtype=cluster_id.dtype)*N).view(B,1)
    seg=(cluster_id+offsets).reshape(-1)
    m=deg.reshape(-1)
    idx=seg[m]
    pflat=pv.reshape(-1,C)[m]
    tflat=tv.reshape(-1,C)[m]

    S=B*N
    ps=pv.new_zeros((S,C)); ts=tv.new_zeros((S,C)); cnt=pv.new_zeros((S,))
    ps.index_add_(0,idx,pflat); ts.index_add_(0,idx,tflat)
    cnt.index_add_(0,idx,torch.ones_like(idx,dtype=pv.dtype))
    valid=(cnt>=1.5).to(pv.dtype)
    denom=cnt.clamp_min(1.0)
    ncluster=valid.sum().clamp_min(1.0)

    per_full=(torch.abs(ps-ts)/denom[:,None]).mean(-1)
    cluster_trace=(per_full*valid).sum()/ncluster

    pz=ps[:,ZLIST]; tz=ts[:,ZLIST]
    pzmass=pz.sum(-1); tzmass=tz.sum(-1)
    ppz=pz/pzmass[:,None].clamp_min(eps); tpz=tz/tzmass[:,None].clamp_min(eps)
    parent_err=torch.abs(ppz-tpz).mean(-1)
    # Weight by physical zero-sector trace per state; clusters with no true zero content vanish.
    zw=valid*(tzmass/denom)
    zero_cluster_parentage=(parent_err*zw).sum()/zw.sum().clamp_min(eps)

    zero_mass_per=torch.abs(pzmass-tzmass)/denom
    zero_cluster_mass=(zero_mass_per*valid).sum()/ncluster
    return cluster_trace,zero_cluster_parentage,zero_cluster_mass


class OperatorInformedLoss(nn.Module):
    def __init__(self,isolated_class_weights,
                 projector_ce_weight=.22,projector_l1_weight=.35,
                 zero_parentage_weight=.30,zero_mass_weight=.18,
                 cluster_trace_weight=.12,zero_cluster_parentage_weight=.20,zero_cluster_mass_weight=.08,
                 rank_weight=.12,isolated_ce_weight=.08,chiral_projector_weight=.05,
                 monotonic_weight=.03,zero_entropy_weight=.05,degenerate_state_purity=.90):
        super().__init__()
        self.register_buffer('isolated_class_weights',torch.as_tensor(isolated_class_weights,dtype=torch.float32))
        self.w_pce=projector_ce_weight;self.w_pl1=projector_l1_weight;self.w_zp=zero_parentage_weight;self.w_zm=zero_mass_weight
        self.w_ct=cluster_trace_weight;self.w_zcp=zero_cluster_parentage_weight;self.w_zcm=zero_cluster_mass_weight
        self.w_rank=rank_weight;self.w_iso=isolated_ce_weight;self.w_ch=chiral_projector_weight
        self.w_mono=monotonic_weight;self.w_zent=zero_entropy_weight;self.purity=float(degenerate_state_purity)

    def forward(self,pE,tE,plog,true_scores,legacy_hard,ref_dims,deg,csize,cluster_id):
        pprob=torch.softmax(plog,-1);logp=F.log_softmax(plog,-1)
        energy=F.smooth_l1_loss(pE,tE,beta=.05)

        smask=state_supervision_mask(true_scores,deg,self.purity)
        sf=smask.to(pprob.dtype);sden=sf.sum().clamp_min(1.0)
        state_ce=-(true_scores*logp).sum(-1);state_l1=torch.abs(pprob-true_scores).mean(-1)
        projector_ce=(state_ce*sf).sum()/sden
        projector_l1=(state_l1*sf).sum()/sden

        tp,tm=zero_parentage(true_scores);pp,pm=zero_parentage(pprob)
        state_w=tm.detach()*sf
        zero_parentage_l1=(torch.abs(tp-pp).mean(-1)*state_w).sum()/state_w.sum().clamp_min(1e-8)
        zero_mass_state=(torch.abs(pm-tm)*sf).sum()/sden
        tent=-(tp*torch.log(tp.clamp_min(1e-8))).sum(-1);pent=-(pp*torch.log(pp.clamp_min(1e-8))).sum(-1)
        zero_entropy=(torch.abs(tent-pent)*state_w).sum()/state_w.sum().clamp_min(1e-8)

        cluster_trace,zero_cluster_parentage,zero_cluster_mass=vectorized_cluster_terms(pprob,true_scores,cluster_id,deg)

        N=float(pprob.size(1));rank=F.smooth_l1_loss(pprob.sum(1)/N,ref_dims/N,beta=.01)
        target6=isolated_target6(legacy_hard);m6=target6>=0
        iso_logits=plog[...,ILIST][m6];iso_target=target6[m6]
        iso_vec=F.cross_entropy(iso_logits,iso_target,weight=self.isolated_class_weights.to(device=plog.device,dtype=plog.dtype),reduction='none')
        isolated_ce=iso_vec.sum()/m6.to(pprob.dtype).sum().clamp_min(1.0)

        iso_left=pprob[...,ILIST];iso_right=torch.flip(pprob,dims=[1])[...,ISO_PARTNER_9]
        iso_true_mass=true_scores[...,ILIST].sum(-1)
        chiral_projector=(torch.abs(iso_left-iso_right).mean(-1)*iso_true_mass).sum()/iso_true_mass.sum().clamp_min(1.)
        mono=F.relu(pE[:,:-1,0]-pE[:,1:,0]).mean()

        total=(energy+self.w_pce*projector_ce+self.w_pl1*projector_l1+
               self.w_zp*zero_parentage_l1+self.w_zm*zero_mass_state+
               self.w_ct*cluster_trace+self.w_zcp*zero_cluster_parentage+self.w_zcm*zero_cluster_mass+
               self.w_rank*rank+self.w_iso*isolated_ce+self.w_ch*chiral_projector+
               self.w_mono*mono+self.w_zent*zero_entropy)
        parts={'energy':energy,'projector_ce':projector_ce,'projector_l1':projector_l1,
               'zero_parentage_state':zero_parentage_l1,'zero_mass_state':zero_mass_state,
               'cluster_trace':cluster_trace,'zero_cluster_parentage':zero_cluster_parentage,
               'zero_cluster_mass':zero_cluster_mass,'rank':rank,'isolated_ce':isolated_ce,
               'chiral_projector':chiral_projector,'monotonic':mono,'zero_entropy':zero_entropy}
        return total,parts

def build_model(args):
    return OperatorInformedFullSpectrumModel(args.d_model,args.nhead,args.encoder_layers,args.operator_layers,args.query_layers,args.spectral_layers,args.n_latents,args.dim_feedforward,args.dropout,args.cnn_channels,args.operator_mode,args.projector_head_mode)

def model_config(args):
    return {k:getattr(args,k) for k in ['d_model','nhead','encoder_layers','operator_layers','query_layers','spectral_layers','n_latents','dim_feedforward','dropout','cnn_channels','operator_mode','projector_head_mode']}

def loss_config(args):
    keys=['projector_ce_weight','projector_l1_weight','zero_parentage_weight','zero_mass_weight',
          'cluster_trace_weight','zero_cluster_parentage_weight','zero_cluster_mass_weight','rank_weight',
          'isolated_ce_weight','chiral_projector_weight','monotonic_weight','zero_entropy_weight','degenerate_state_purity']
    return {k:float(getattr(args,k)) for k in keys}

def move(batch,device):return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


def rng_state():
    out={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available():out['cuda']=torch.cuda.get_rng_state_all()
    return out

def restore_rng_state(x):
    if not x:return
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if torch.cuda.is_available() and 'cuda' in x:torch.cuda.set_rng_state_all(x['cuda'])


def save_ckpt(path,model,opt,sched,scaler,epoch,best,bad_epochs,args,metrics):
    torch.save({'version':VERSION,'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':opt.state_dict() if opt else None,'scheduler_state':sched.state_dict() if sched else None,'scaler_state':scaler.state_dict() if scaler else None,'best':best,'bad_epochs':bad_epochs,'metrics':metrics,'model_config':model_config(args),'loss_config':loss_config(args),'rng_state':rng_state()},path)

def load_model_ckpt(path,device):
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck.get('version')!=VERSION:raise RuntimeError(f'Checkpoint version {ck.get("version")} is not compatible with {VERSION}')
    m=OperatorInformedFullSpectrumModel(**ck['model_config']).to(device);m.load_state_dict(ck['model_state']);return m,ck

def load_v9_initialization(model,path):
    ck=torch.load(path,map_location='cpu',weights_only=False);src=ck.get('model_state',ck);dst=model.state_dict();use={k:v for k,v in src.items() if k in dst and tuple(v.shape)==tuple(dst[k].shape) and not k.startswith('energy_head')};model.load_state_dict(use,strict=False);print(f'v9 warm start: loaded {len(use)} compatible tensors; ordered-energy decoder/operator path remain newly initialized')


def _metric_tensors(pp,true_scores,legacy,deg,cluster_id,purity_threshold,parts):
    abs_state=torch.abs(pp-true_scores).mean(-1)
    smask=state_supervision_mask(true_scores,deg,purity_threshold);sf=smask.to(pp.dtype)
    proj_state=(abs_state*sf).sum()/sf.sum().clamp_min(1.0)
    proj_all=abs_state.mean()

    pz,pm=zero_parentage(pp);tz,tm=zero_parentage(true_scores);w=tm.detach();sw=w*sf
    zp_state=(torch.abs(pz-tz).mean(-1)*sw).sum()/sw.sum().clamp_min(1e-8)
    zp_all=(torch.abs(pz-tz).mean(-1)*w).sum()/w.sum().clamp_min(1e-8)
    zm_state=(torch.abs(pm-tm)*sf).sum()/sf.sum().clamp_min(1.0)
    zm_all=torch.abs(pm-tm).mean()

    # Reuse the exact vectorized cluster quantities already computed by the loss.
    md={'projector_state_mae':proj_state,'projector_legacy_all_mae':proj_all,
        'zero_parentage_state_mae':zp_state,'zero_parentage_legacy_all_mae':zp_all,
        'zero_mass_state_mae':zm_state,'zero_mass_legacy_all_mae':zm_all,
        'cluster_trace_mae':parts['cluster_trace'],'zero_cluster_parentage_mae':parts['zero_cluster_parentage'],
        'zero_cluster_mass_mae':parts['zero_cluster_mass']}

    target6=isolated_target6(legacy);m6=target6>=0
    pred6=torch.argmax(pp[...,ILIST],-1)
    code=(target6[m6]*6+pred6[m6]).to(torch.long)
    conf=torch.bincount(code,minlength=36).reshape(6,6)
    return md,conf


def run_epoch(model,loader,criterion,device,optimizer=None,grad_scaler=None,accum_steps=1,grad_clip=1.,amp=True,
              progress_every=0,progress_label=''):
    training=optimizer is not None;model.train(training)
    loss_keys=['loss','energy','projector_ce','projector_l1','zero_parentage_state','zero_mass_state',
               'cluster_trace','zero_cluster_parentage','zero_cluster_mass','rank','isolated_ce',
               'chiral_projector','monotonic','zero_entropy']
    sums={k:torch.zeros((),device=device,dtype=torch.float32) for k in loss_keys}
    metric_keys=['projector_state_mae','projector_legacy_all_mae','zero_parentage_state_mae',
                 'zero_parentage_legacy_all_mae','zero_mass_state_mae','zero_mass_legacy_all_mae',
                 'cluster_trace_mae','zero_cluster_parentage_mae','zero_cluster_mass_mae']
    metric_sums={k:torch.zeros((),device=device,dtype=torch.float32) for k in metric_keys}
    iconf=torch.zeros((6,6),device=device,dtype=torch.long);nf=0
    if training:optimizer.zero_grad(set_to_none=True)
    use_amp=amp and device.type=='cuda';t0=time.time();total_steps=len(loader)
    for step,b in enumerate(loader):
        b=move(b,device)
        if step==0:assert_runtime_devices(model,criterion,device,b)
        with torch.autocast(device_type=device.type,dtype=torch.float16 if device.type=='cuda' else torch.bfloat16,enabled=use_amp):
            pE,plog,_=model(b['grid'],b['global_features'],b['topology_prior'],b['q'],b['operator_tokens'],b['operator_global'])
            loss,parts=criterion(pE,b['energy'],plog,b['branch_scores'],b['legacy_hard_branch'],b['ref_dims'],b['degenerate_mask'],b['cluster_size'],b['cluster_id'])
            back=loss/float(accum_steps)
        if training:
            if grad_scaler is not None and use_amp:grad_scaler.scale(back).backward()
            else:back.backward()
            if (step+1)%accum_steps==0 or (step+1)==total_steps:
                if grad_scaler is not None and use_amp:
                    grad_scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),grad_clip);grad_scaler.step(optimizer);grad_scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(),grad_clip);optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pp=torch.softmax(plog,-1)
            md,cf=_metric_tensors(pp,b['branch_scores'],b['legacy_hard_branch'],b['degenerate_mask'],b['cluster_id'],criterion.purity,parts)
            for k,v in md.items():metric_sums[k].add_(v.detach().float())
            iconf.add_(cf)
            sums['loss'].add_(loss.detach().float())
            for k,v in parts.items():sums[k].add_(v.detach().float())
        nf+=1
        if progress_every and ((step+1)%progress_every==0 or (step+1)==total_steps):
            print(f"{progress_label} {step+1}/{total_steps} families | elapsed={time.time()-t0:.1f}s",flush=True)

    # Exactly one bulk device->CPU transfer for scalar epoch statistics, instead of one per cluster/state.
    names=list(sums)+list(metric_sums)
    vals=torch.stack([sums[k] for k in sums]+[metric_sums[k] for k in metric_sums])/float(max(nf,1))
    arr=vals.detach().cpu().numpy().tolist();out=dict(zip(names,arr))
    conf_np=iconf.detach().cpu().numpy();im=classification_metrics_from_confusion(conf_np)
    out['isolated_macro_f1']=float(im['macro_f1'])
    return out,conf_np

def topology_score(m):
    return float(m['projector_state_mae']+.75*m['cluster_trace_mae']+.75*m['zero_parentage_state_mae']+.75*m['zero_cluster_parentage_mae']+.25*m['zero_mass_state_mae']+.25*m['zero_cluster_mass_mae'])
def zero_topology_score(m):return float(m['zero_parentage_state_mae']+m['zero_cluster_parentage_mae']+.25*m['zero_mass_state_mae']+.25*m['zero_cluster_mass_mae'])


def prepare_split_and_scalers(args,records):
    if (args.run_dir/'family_split.json').exists():train,val,test=load_split(args.run_dir,records)
    else:train,val,test=stratified_family_split(records,args.seed,args.val_fraction,args.test_fraction);save_split(args.run_dir,train,val,test)
    if (args.run_dir/'scalers.json').exists():es,gs=load_scalers(args.run_dir)
    else:es=fit_energy_scaler(train);gs=fit_global_scaler(train);save_scalers(args.run_dir,es,gs)
    return train,val,test,es,gs


def make_criterion(args,weights):
    return OperatorInformedLoss(weights,
        args.projector_ce_weight,args.projector_l1_weight,args.zero_parentage_weight,args.zero_mass_weight,
        args.cluster_trace_weight,args.zero_cluster_parentage_weight,args.zero_cluster_mass_weight,
        args.rank_weight,args.isolated_ce_weight,args.chiral_projector_weight,args.monotonic_weight,args.zero_entropy_weight,
        args.degenerate_state_purity)


def mode_check(args):
    device=resolve_device(args.device);args.device=str(device);records=load_available_families(args.db_root);validate_cache_audit(args.operator_cache,len(records),args.db_root/'geometry_manifest.csv');train,val,test,es,gs=prepare_split_and_scalers(args,records);weights,_=fit_isolated_class_weights(train);ds,ld=make_loader(train[:1],es,gs,False,args);b=move(next(iter(ld)),device);model=build_model(args).to(device);criterion=make_criterion(args,weights).to(device);assert_runtime_devices(model,criterion,device,b);opt=torch.optim.AdamW(model.parameters(),lr=1e-5);model.train();pE,pl,_=model(b['grid'],b['global_features'],b['topology_prior'],b['q'],b['operator_tokens'],b['operator_global']);loss,parts=criterion(pE,b['energy'],pl,b['branch_scores'],b['legacy_hard_branch'],b['ref_dims'],b['degenerate_mask'],b['cluster_size'],b['cluster_id']);loss.backward();finite_grad=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();pp=pl.softmax(-1);ch=float(torch.max(torch.abs(pE+torch.flip(pE,[1]))).detach());mono=int((pE[:,:-1,0]>pE[:,1:,0]+1e-7).sum().detach());print('family=',b['family_id'][0],'Nstate=',pE.size(1),'device=',device);print('operator_tokens=',tuple(b['operator_tokens'].shape),'loss_buffer_device=',criterion.isolated_class_weights.device);print('energy chiral maxerr=',ch,'monotonic violations=',mono);print('projector row-sum maxerr=',float(torch.max(torch.abs(pp.sum(-1)-1)).detach()));print('loss=',float(loss.detach()),'finite_grad=',bool(finite_grad),'operator finite=',bool(torch.isfinite(b['operator_tokens']).all()));
    if ch>1e-6 or mono!=0 or not finite_grad or not torch.isfinite(loss):raise RuntimeError('Model/loss backward smoke test failed')
    print('*** V12 PAPER FORWARD + LOSS + BACKWARD CHECK PASSED ***')


def write_run_metadata(args,records,train,val,test):
    split_text='|'.join(r.family_id for r in train)+'||'+'|'.join(r.family_id for r in val)+'||'+'|'.join(r.family_id for r in test);files=['fractal_ssh_full_spectrum_operator_v12.py','model_blocks_v12.py','v9_common.py','operator_cache_v12.py','operator_core_v12.py','operator_features_v12.py','validate_operator_informed_v12.py','run_full_pipeline_operator_v12_REUSE_V7_DB.py']
    meta={'version':VERSION,'seed':args.seed,'db_root':str(args.db_root),'run_dir':str(args.run_dir),'operator_cache':str(args.operator_cache),'split_sha256':hashlib.sha256(split_text.encode()).hexdigest(),'num_families':len(records),'num_train':len(train),'num_val':len(val),'num_test':len(test),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'cuda_available':torch.cuda.is_available(),'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'args':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},'code_sha256':{f:sha256_file(Path(__file__).resolve().parent/f) for f in files},'physics_policy':{'finite_hamiltonian':'database t1=-0.1,t2=-1','operator_input':'geometry-only K1/K2/K3, Keff3=t1*K1+t1^2*K2+t1^3*K3, eta2 second-order interference efficiency, path-presence/path-strength and first-nonzero-order descriptors from the t1=0 parent subspace','degenerate_supervision':'single-pass GPU multiplicity-normalized projector-trace + aggregate zero-sector trace','state_level_projector_supervision':'non-degenerate + high-purity degenerate states; factorized zero-mass and conditional zero-parentage head with exact projector-level chiral pairing by default','energy_decoder':'ordered chiral nonnegative-spacing decoder with exact zero-gap/degeneracy support; no post-hoc sort'}}
    (args.run_dir/'run_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')


def mode_train(args):
    args.run_dir.mkdir(parents=True,exist_ok=True);set_seed(args.seed);device=resolve_device(args.device);args.device=str(device);records=load_available_families(args.db_root);validate_cache_audit(args.operator_cache,len(records),args.db_root/'geometry_manifest.csv');print_family_stats('ALL',records);train,val,test,es,gs=prepare_split_and_scalers(args,records);print_family_stats('TRAIN',train);print_family_stats('VAL',val);print_family_stats('TEST',test);write_run_metadata(args,records,train,val,test)
    trds,trld=make_loader(train,es,gs,True,args);_,vald=make_loader(val,es,gs,False,args);weights,counts=fit_isolated_class_weights(train);print('Isolated branch counts:',dict(zip(ISOLATED_LABELS,counts.astype(int).tolist())));criterion=make_criterion(args,weights).to(device);model=build_model(args).to(device)
    assert_runtime_devices(model,criterion,device)
    last=args.run_dir/'last_model.pt';
    if args.init_v9 and not last.exists():load_v9_initialization(model,Path(args.init_v9))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=args.lr_patience,min_lr=args.min_lr)
    try:scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and not args.no_amp))
    except Exception:scaler=torch.cuda.amp.GradScaler(enabled=(device.type=='cuda' and not args.no_amp)) if device.type=='cuda' else None
    start=1;best={'topology':float('inf'),'projector':float('inf'),'zero_parentage':float('inf'),'cluster_trace':float('inf'),'v9_zero':float('inf'),'zero_cluster':float('inf')};bad=0;hist=[]
    if args.resume and last.exists():
        ck=torch.load(last,map_location=device,weights_only=False)
        if ck.get('version')!=VERSION:raise RuntimeError(f'Cannot resume incompatible checkpoint {ck.get("version")} with {VERSION}. Use a fresh run directory.')
        saved_loss=ck.get('loss_config',{});current_loss=loss_config(args)
        if saved_loss and any(abs(float(saved_loss[k])-float(current_loss[k]))>1e-12 for k in current_loss if k in saved_loss):
            raise RuntimeError('Resume loss configuration differs from checkpoint. Use the original settings or a fresh run directory.')
        model.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer_state']);sched.load_state_dict(ck['scheduler_state']);
        if scaler is not None and ck.get('scaler_state'):scaler.load_state_dict(ck['scaler_state'])
        start=int(ck['epoch'])+1;best.update(ck.get('best',{}));bad=int(ck.get('bad_epochs',0));restore_rng_state(ck.get('rng_state'));hp=args.run_dir/'training_history_v12.csv'
        if hp.exists():
            try:hist=pd.read_csv(hp).to_dict('records')
            except Exception:hist=[]
        print('Resumed from epoch',start-1,'bad_epochs=',bad)
    if bad>=args.early_stop_patience:print('Previous run had already reached early-stopping patience; skipping further training.')
    for epoch in range(start,args.epochs+1):
        if bad>=args.early_stop_patience:break
        trds.set_epoch(epoch);t=time.time();tr,_=run_epoch(model,trld,criterion,device,opt,scaler,args.grad_accum_steps,args.grad_clip,not args.no_amp,args.progress_every,f'Epoch {epoch:03d} TRAIN');va,_=run_epoch(model,vald,criterion,device,None,None,1,args.grad_clip,not args.no_amp,args.progress_every,f'Epoch {epoch:03d} VAL');score=topology_score(va);zscore=zero_topology_score(va);sched.step(score);row={'epoch':epoch,'lr':opt.param_groups[0]['lr'],'seconds':time.time()-t,'train_topology_score':topology_score(tr),'val_topology_score':score,'val_zero_topology_score':zscore};row.update({'train_'+k:v for k,v in tr.items()});row.update({'val_'+k:v for k,v in va.items()});hist.append(row);pd.DataFrame(hist).to_csv(args.run_dir/'training_history_v12.csv',index=False)
        improved=False
        if score<best['topology']-1e-6:best['topology']=score;improved=True
        if va['projector_state_mae']<best['projector']-1e-6:best['projector']=va['projector_state_mae']
        if zscore<best['zero_parentage']-1e-6:best['zero_parentage']=zscore
        if va['cluster_trace_mae']<best['cluster_trace']-1e-6:best['cluster_trace']=va['cluster_trace_mae']
        if va['zero_parentage_legacy_all_mae']<best['v9_zero']-1e-6:best['v9_zero']=va['zero_parentage_legacy_all_mae']
        if va['zero_cluster_parentage_mae']<best['zero_cluster']-1e-6:best['zero_cluster']=va['zero_cluster_parentage_mae']
        bad=0 if improved else bad+1;metrics={'train':tr,'val':va,'topology_score':score,'zero_topology_score':zscore}
        # Save specialized checkpoints based on the metric value attained at this epoch.
        if abs(score-best['topology'])<1e-12:save_ckpt(args.run_dir/'best_model.pt',model,opt,sched,scaler,epoch,best,bad,args,metrics)
        if abs(va['projector_state_mae']-best['projector'])<1e-12:save_ckpt(args.run_dir/'best_projector_model.pt',model,opt,sched,scaler,epoch,best,bad,args,metrics)
        if abs(zscore-best['zero_parentage'])<1e-12:save_ckpt(args.run_dir/'best_zero_parentage_model.pt',model,opt,sched,scaler,epoch,best,bad,args,metrics)
        if abs(va['cluster_trace_mae']-best['cluster_trace'])<1e-12:save_ckpt(args.run_dir/'best_cluster_trace_model.pt',model,opt,sched,scaler,epoch,best,bad,args,metrics)
        if abs(va['zero_parentage_legacy_all_mae']-best['v9_zero'])<1e-12:save_ckpt(args.run_dir/'best_v9_comparable_zero_model.pt',model,opt,sched,scaler,epoch,best,bad,args,metrics)
        if abs(va['zero_cluster_parentage_mae']-best['zero_cluster'])<1e-12:save_ckpt(args.run_dir/'best_zero_cluster_model.pt',model,opt,sched,scaler,epoch,best,bad,args,metrics)
        save_ckpt(last,model,opt,sched,scaler,epoch,best,bad,args,metrics)
        print(f"Epoch {epoch:03d} | topo={score:.5f} projState={va['projector_state_mae']:.5f} projLegacy={va['projector_legacy_all_mae']:.5f} zeroState={va['zero_parentage_state_mae']:.5f} zeroLegacy={va['zero_parentage_legacy_all_mae']:.5f} zeroCluster={va['zero_cluster_parentage_mae']:.5f} cluster={va['cluster_trace_mae']:.5f} isoF1={va['isolated_macro_f1']:.4f} lr={opt.param_groups[0]['lr']:.2e}")
    print('Training finished. best=',best)


def predict_family(model,r,es,gs,device,operator_cache):
    with np.load(r.mask_path,allow_pickle=False) as z:M=z['M'].astype(np.uint8)
    prior,ref,_,_=theory_graph_features_from_generator(M);gf=gs.transform_np(raw_global_features(r,M)).astype(np.float32);optok,opglob=load_operator_cache(operator_cache,r.family_id)
    with np.load(r.state_path,allow_pickle=False) as z:E=z['eigenvalues'].astype(np.float64);q=z['q_state'].astype(np.float64);scores=z['branch_scores'].astype(np.float64);hard=z['branch_index'].astype(np.int64);cid=z['cluster_id'].astype(np.int64);deg=z['is_degenerate'].astype(bool)
    scores=np.clip(scores,0,None);scores/=np.clip(scores.sum(1,keepdims=True),1e-12,None);T=lambda x:torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():pE,pl,_=model(T(make_geometry_grid(M,r.b)),T(gf),T(prior.astype(np.float32)),T(q.astype(np.float32)[:,None]),T(optok),T(opglob));pp=pl.softmax(-1)[0].cpu().numpy();predE=es.inverse_np(pE[0,:,0].cpu().numpy())
    return {'true_E':E,'pred_E':predE,'true_scores':scores,'pred_scores':pp,'legacy_hard':hard,'cluster_id':cid,'deg':deg,'ref_dims':ref}


def cluster_terms_np(ps,ts,cid,deg,eps=1e-12):
    """NumPy equivalent of the vectorized GPU cluster aggregation used for final test reporting."""
    N,C=ps.shape
    idx=cid[deg].astype(np.int64,copy=False)
    psum=np.zeros((N,C),dtype=np.float64);tsum=np.zeros((N,C),dtype=np.float64);cnt=np.zeros(N,dtype=np.float64)
    np.add.at(psum,idx,ps[deg]);np.add.at(tsum,idx,ts[deg]);np.add.at(cnt,idx,1.0)
    valid=cnt>=1.5;ncl=int(valid.sum());den=np.clip(cnt,1.0,None)
    full_per=(np.abs(psum-tsum)/den[:,None]).mean(1)
    full_sum=float(full_per[valid].sum())
    pz=psum[:,ZLIST];tz=tsum[:,ZLIST];pm=pz.sum(1);tm=tz.sum(1)
    pp=pz/np.clip(pm[:,None],eps,None);tp=tz/np.clip(tm[:,None],eps,None)
    perzp=np.abs(pp-tp).mean(1);zw=np.where(valid,tm/den,0.0)
    zp_num=float((perzp*zw).sum());zp_den=float(zw.sum())
    zm_per=np.abs(pm-tm)/den;zm_sum=float(zm_per[valid].sum())
    return {'cluster_trace_MAE':full_sum/max(ncl,1),'zero_cluster_parentage_MAE':zp_num/max(zp_den,eps),
            'zero_cluster_mass_MAE':zm_sum/max(ncl,1),'cluster_trace_sum':full_sum,'cluster_count':ncl,
            'zero_parentage_num':zp_num,'zero_parentage_den':zp_den,'zero_mass_sum':zm_sum}


def family_metrics_np(p,purity_threshold=.90):
    E,PE,ts,ps,cid,deg=p['true_E'],p['pred_E'],p['true_scores'],p['pred_scores'],p['cluster_id'],p['deg']
    purity=ts.max(1);smask=(~deg)|(purity>=float(purity_threshold))
    absproj=np.abs(ts-ps).mean(1);tz=ts[:,ZLIST];pz=ps[:,ZLIST];tm=tz.sum(1);pm=pz.sum(1)
    tp=tz/np.clip(tm[:,None],1e-12,None);pp=pz/np.clip(pm[:,None],1e-12,None)
    def wz(mask):
        w=tm*mask.astype(float);return float(np.sum(np.abs(tp-pp).mean(1)*w)/max(w.sum(),1e-12))
    c=cluster_terms_np(ps,ts,cid,deg)
    return {'energy_MAE':float(np.mean(np.abs(E-PE))),'energy_RMSE':float(np.sqrt(np.mean((E-PE)**2))),'R2':numpy_r2(E,PE),
            'projector_state_MAE':float(absproj[smask].mean()) if smask.any() else 0.,'projector_legacy_all_MAE':float(absproj.mean()),
            'zero_parentage_state_MAE':wz(smask),'zero_parentage_legacy_all_MAE':wz(np.ones_like(deg,bool)),
            'zero_mass_state_MAE':float(np.abs(pm[smask]-tm[smask]).mean()) if smask.any() else 0.,'zero_mass_legacy_all_MAE':float(np.abs(pm-tm).mean()),
            'cluster_trace_MAE':c['cluster_trace_MAE'],'zero_cluster_parentage_MAE':c['zero_cluster_parentage_MAE'],
            'zero_cluster_mass_MAE':c['zero_cluster_mass_MAE'],'pred_chiral_maxerr':float(np.max(np.abs(PE+PE[::-1]))),
            'pred_monotonic_violations':int(np.sum(np.diff(PE)<-1e-9)),**{k:c[k] for k in ['cluster_trace_sum','cluster_count','zero_parentage_num','zero_parentage_den','zero_mass_sum']}}

def mode_test(args):
    args.run_dir.mkdir(parents=True,exist_ok=True)
    device=resolve_device(args.device);args.device=str(device)
    records=load_available_families(args.db_root)
    validate_cache_audit(args.operator_cache,len(records),args.db_root/'geometry_manifest.csv')
    train,val,test,es,gs=prepare_split_and_scalers(args,records)
    model,ck=load_model_ckpt(args.run_dir/args.checkpoint_name,device);model.eval();test_purity=float(ck.get('loss_config',{}).get('degenerate_state_purity',args.degenerate_state_purity))
    rows=[];allE=[];allPE=[];allTS=[];allPS=[];allDEG=[];allH=[]
    pred_dir=args.run_dir/f'test_predictions_{args.test_tag}';pred_dir.mkdir(exist_ok=True)
    cluster_full_sum=cluster_full_n=0.0
    cluster_zp_sum=cluster_zp_n=0.0
    cluster_zm_sum=cluster_zm_n=0.0
    for i,r in enumerate(test,1):
        p=predict_family(model,r,es,gs,device,args.operator_cache);m=family_metrics_np(p,test_purity)
        om=load_operator_meta(args.operator_cache,r.family_id)
        opdiag={k:om.get(k) for k in ['d_OC_B0','K2_OC_B0_fro','K3_OC_B0_fro','K2_path_piece_count','K2_path_abs_sum','eta2_OC_B0','cancel2_OC_B0','two_step_path_present','first_nonzero_OC_B0_order']}
        rows.append({'family_id':r.family_id,'b':r.b,'N':r.N,'Df':r.Df,'n_sites':len(p['true_E']),**opdiag,**m})
        np.savez_compressed(pred_dir/f'{r.family_id}_prediction.npz',**p)
        allE.append(p['true_E']);allPE.append(p['pred_E']);allTS.append(p['true_scores']);allPS.append(p['pred_scores']);allDEG.append(p['deg']);allH.append(p['legacy_hard'])
        cluster_full_sum += m['cluster_trace_sum']; cluster_full_n += m['cluster_count']
        cluster_zp_sum += m['zero_parentage_num']; cluster_zp_n += m['zero_parentage_den']
        cluster_zm_sum += m['zero_mass_sum']; cluster_zm_n += m['cluster_count']
        print(f"[{i}/{len(test)}] {r.family_id} E={m['energy_MAE']:.5f} projState={m['projector_state_MAE']:.5f} zeroState={m['zero_parentage_state_MAE']:.5f} zeroCluster={m['zero_cluster_parentage_MAE']:.5f}")
    df=pd.DataFrame(rows);tag=args.test_tag;df.to_csv(args.run_dir/f'test_family_metrics_{tag}.csv',index=False)
    te=np.concatenate(allE);pe=np.concatenate(allPE);ts=np.concatenate(allTS);ps=np.concatenate(allPS);deg=np.concatenate(allDEG);hard=np.concatenate(allH)
    purity=ts.max(1);primary_mask=(~deg)|(purity>=test_purity);abs_state=np.abs(ts-ps).mean(1)
    tz=ts[:,ZLIST];pz=ps[:,ZLIST];tm=tz.sum(1);pm=pz.sum(1);tp=tz/np.clip(tm[:,None],1e-12,None);ppz=pz/np.clip(pm[:,None],1e-12,None)
    zero_err=np.abs(tp-ppz).mean(1)
    w_primary=tm*primary_mask.astype(np.float64)
    # Exact v9-compatible all-state metrics (same state-weighted convention).
    legacy_projector=float(np.abs(ts-ps).mean())
    legacy_zero_parentage=float(np.sum(zero_err*tm)/max(float(tm.sum()),1e-12))
    legacy_zero_mass=float(np.mean(np.abs(pm-tm)))
    primary_projector=float(abs_state[primary_mask].mean()) if primary_mask.any() else 0.0
    primary_zero_parentage=float(np.sum(zero_err*w_primary)/max(float(w_primary.sum()),1e-12))
    primary_zero_mass=float(np.abs(pm[primary_mask]-tm[primary_mask]).mean()) if primary_mask.any() else 0.0
    map9=np.full(9,-1,int)
    for k9,k6 in MAP9TO6.items():map9[k9]=k6
    mask=np.isin(hard,np.asarray(ISOLATED_INDICES));it=map9[hard[mask]];ip=ps[mask][:,ILIST].argmax(1);conf=np.zeros((6,6),np.int64);np.add.at(conf,(it,ip),1);im=classification_metrics_from_confusion(conf)
    summ={
        'version':VERSION,'checkpoint':args.checkpoint_name,'checkpoint_epoch':int(ck['epoch']),
        'operator_mode':ck.get('model_config',{}).get('operator_mode',args.operator_mode),
        'projector_head_mode':ck.get('model_config',{}).get('projector_head_mode',args.projector_head_mode),
        'num_test_families':len(test),'num_test_states':len(te),
        'state_weighted_energy':{
            'MAE':float(np.mean(np.abs(te-pe))),'RMSE':float(np.sqrt(np.mean((te-pe)**2))),
            'pooled_R2':numpy_r2(te,pe),'pred_chiral_maxerr':float(df.pred_chiral_maxerr.max()),
            'total_monotonic_violations':int(df.pred_monotonic_violations.sum())},
        'primary_gauge_aware':{
            'state_projector_MAE_nondeg_or_highpurity':primary_projector,
            'state_zero_parentage_MAE_weighted_nondeg_or_highpurity':primary_zero_parentage,
            'state_zero_mass_MAE_nondeg_or_highpurity':primary_zero_mass,
            'degenerate_cluster_normalized_projector_trace_MAE':float(cluster_full_sum/max(cluster_full_n,1.0)),
            'degenerate_cluster_zero_parentage_MAE':float(cluster_zp_sum/max(cluster_zp_n,1e-12)),
            'degenerate_cluster_zero_mass_MAE':float(cluster_zm_sum/max(cluster_zm_n,1.0)),
            'isolated6_macro_F1':float(im['macro_f1'])},
        'v9_comparable_state_weighted':{
            'projector_score_MAE':legacy_projector,
            'zero_parentage_MAE_weighted':legacy_zero_parentage,
            'zero_mass_MAE':legacy_zero_mass,
            'isolated6_macro_F1_PRIMARY_AUX':float(im['macro_f1'])},
        'family_mean_diagnostics':{
            'projector_state_MAE':float(df.projector_state_MAE.mean()),
            'cluster_trace_MAE':float(df.cluster_trace_MAE.mean()),
            'zero_parentage_state_MAE':float(df.zero_parentage_state_MAE.mean()),
            'zero_cluster_parentage_MAE':float(df.zero_cluster_parentage_MAE.mean())},
        'interpretation':{
            'primary':'State-level supervision uses nondegenerate states plus high-purity degenerate states; mixed exact-degenerate clusters use aggregate gauge-invariant projector traces.',
            'v9_comparable':'The v9_comparable_state_weighted block intentionally retains the original all-state v9 metric convention for direct baseline tables.'}}
    (args.run_dir/f'test_summary_{tag}.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False),encoding='utf-8')
    np.savetxt(args.run_dir/f'isolated_confusion_{tag}.csv',conf,delimiter=',',fmt='%d')
    print(json.dumps(summ,indent=2,ensure_ascii=False))


def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['check','train','test'],required=True);p.add_argument('--db-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--operator-cache',type=Path,default=None);p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu');p.add_argument('--seed',type=int,default=20260912);p.add_argument('--num-workers',type=int,default=0);p.add_argument('--val-fraction',type=float,default=.10);p.add_argument('--test-fraction',type=float,default=.10)
    p.add_argument('--d-model',type=int,default=128);p.add_argument('--nhead',type=int,default=8);p.add_argument('--encoder-layers',type=int,default=3);p.add_argument('--operator-layers',type=int,default=2);p.add_argument('--query-layers',type=int,default=2);p.add_argument('--spectral-layers',type=int,default=3);p.add_argument('--n-latents',type=int,default=40);p.add_argument('--dim-feedforward',type=int,default=512);p.add_argument('--dropout',type=float,default=.10);p.add_argument('--cnn-channels',type=int,default=48)
    p.add_argument('--operator-mode',choices=['none','k1','k1k2','k1k2k3','full_no_hierarchy','full'],default='full');p.add_argument('--projector-head-mode',choices=['flat','factorized'],default='factorized')
    p.add_argument('--epochs',type=int,default=140);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--min-lr',type=float,default=1e-6);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--lr-patience',type=int,default=7);p.add_argument('--early-stop-patience',type=int,default=28);p.add_argument('--grad-accum-steps',type=int,default=4);p.add_argument('--grad-clip',type=float,default=1.0);p.add_argument('--balance-strength',type=float,default=.35)
    p.add_argument('--projector-ce-weight',type=float,default=.20);p.add_argument('--projector-l1-weight',type=float,default=.32);p.add_argument('--zero-parentage-weight',type=float,default=.40);p.add_argument('--zero-mass-weight',type=float,default=.20);p.add_argument('--cluster-trace-weight',type=float,default=.12);p.add_argument('--zero-cluster-parentage-weight',type=float,default=.30);p.add_argument('--zero-cluster-mass-weight',type=float,default=.10);p.add_argument('--rank-weight',type=float,default=.12);p.add_argument('--isolated-ce-weight',type=float,default=.08);p.add_argument('--chiral-projector-weight',type=float,default=.05);p.add_argument('--monotonic-weight',type=float,default=.03);p.add_argument('--zero-entropy-weight',type=float,default=.05);p.add_argument('--degenerate-state-purity',type=float,default=.90);p.add_argument('--progress-every',type=int,default=50)
    p.add_argument('--no-d4-augment',action='store_true');p.add_argument('--no-amp',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--init-v9',default='');p.add_argument('--checkpoint-name',default='best_model.pt');p.add_argument('--test-tag',default='best_model');return p.parse_args()


def main():
    a=parse_args();a.db_root=a.db_root.resolve();a.run_dir=a.run_dir.resolve();a.run_dir.mkdir(parents=True,exist_ok=True);a.operator_cache=(a.operator_cache.resolve() if a.operator_cache is not None else (a.run_dir/'operator_cache').resolve())
    if not a.operator_cache.exists():raise FileNotFoundError(f'Operator cache not found: {a.operator_cache}. Run STEP 1 first.')
    print('='*110);print('FRACTAL SSH v12 PAPER — K1/K2/K3 OPERATOR HIERARCHY + INTERFERENCE-AWARE ZERO MANIFOLD');print('v9-scale backbone; K1/K2/K3 + Keff3; eta2/path-order descriptors; factorized gauge-aware zero projector.');print('='*110)
    if a.mode=='check':mode_check(a)
    elif a.mode=='train':mode_train(a)
    else:mode_test(a)
if __name__=='__main__':main()
