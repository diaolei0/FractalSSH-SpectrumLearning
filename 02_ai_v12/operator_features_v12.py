# -*- coding: utf-8 -*-
"""Gauge-invariant perturbative-operator hierarchy for Fractal SSH v12.

STEP-1 only (NumPy/SciPy; no PyTorch).

The zero-parent subspace P = P_OC + P_IC-II + P_BULK-0 is defined at t1=0.
For V equal to the unit-amplitude weak-bond operator and
R = Q H0^{-1} Q, v12 constructs

    K1 = P V P
    K2 = - P V R V P
    K3 = P V R V R V P - 1/2 {P V P, P V R^2 V P}

and the fixed-coupling truncated effective operator

    Keff(3) = t1 K1 + t1^2 K2 + t1^3 K3,  t1 = -0.1.

In addition to block norms, v12 encodes the physically validated second-order
interference efficiency

    eta2 = ||sum_c A_c^(2)||_F / sum_c ||A_c^(2)||_F,

where c labels strong-coupling intermediate components.  Thus `no two-step
path` is kept distinct from `many paths that destructively cancel`.

No finite-t1 full-Hamiltonian eigendecomposition is used in this module.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple
import numpy as np
from scipy.sparse import hstack

from operator_core_v12 import (
    site_graph_from_generator, build_zero_data, weak_matrix,
    fro_norm, spectral_norm, shortest_component_distance, _local_h0,
)

PAIR_ORDER=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
SECTOR_NAMES=('OC','IC-II','BULK-0')
T1_FIXED=-0.1

# 4 operator orders/kinds x 6 symmetric sector pairs.
OP_NUM_TOKENS=24
OP_TOKEN_DIM=20
OP_GLOBAL_DIM=24
EPS=1e-18
NZ_TOL=1e-10

# Global feature positions.  Kept explicit because the torch process uses
# them for controlled ablations without importing SciPy.
G_DIMFRAC=slice(0,3)
G_MEAN_K1=3
G_MEAN_K2=4
G_MEAN_K3=5
G_MEAN_KEFF=6
G_ALIGN12=7
G_ALIGN23=8
G_D01=9
G_D02=10
G_D12=11
G_TRACE_OC=12
G_TRACE_ICII=13
G_TRACE_B0=14
G_ETA2=15
G_CANCEL2=16
G_PATH2_PRESENT=17
G_PATH2_COUNT=18
G_PATH2_ABS=19
G_ORDER_ONEHOT=slice(20,24)  # no B0 / K2 first / K3 first / >K3


def _fro_inner(A,B)->float:
    if hasattr(A,'multiply'):
        return float(A.multiply(B).sum())
    return float(np.sum(np.asarray(A)*np.asarray(B)))


def _trace(A)->float:
    if A.shape[0]!=A.shape[1] or A.shape[0]==0:return 0.0
    return float(np.asarray(A.diagonal()).sum()) if hasattr(A,'diagonal') else float(np.trace(np.asarray(A)))


def _compress_trace(x:float)->float:
    return math.copysign(math.log1p(abs(x)),x) if x else 0.0


def _stats(A,di:int,dj:int,exact_spectral:bool=True):
    if di==0 or dj==0:
        return dict(f=0.,s=0.,fn=0.,sn=0.,nz=0.,sr=0.,tr=0.,trc=0.)
    f=fro_norm(A)
    # K1/K2 retain the exact block spectral norm used in v10/v11.  K3 can
    # be much denser, so its Frobenius norm is used as a deterministic
    # spectral upper bound; this is basis/D4 invariant and avoids a large
    # dense SVD for every family.
    s=spectral_norm(A) if exact_spectral else f
    scale=math.sqrt(max(di*dj,1))
    tr=_trace(A)/max(float(di),1.) if di==dj else 0.
    sr=float((f*f)/(s*s+EPS)/max(min(di,dj),1)) if s>0 else 0.
    return dict(
        f=f,s=s,fn=math.log1p(f/scale),sn=math.log1p(s),
        nz=float(f>NZ_TOL),sr=sr,tr=tr,trc=_compress_trace(tr)
    )


def _cosine_abs(A,B,fa:float,fb:float)->float:
    if fa<=1e-14 or fb<=1e-14:return 0.0
    return float(np.clip(abs(_fro_inner(A,B))/(fa*fb+EPS),0.,1.))


def _make_token(kind:int,pidx:int,dimfrac,st,alignment=0.,ratio=0.):
    i,j=PAIR_ORDER[pidx]
    feat=np.zeros(OP_TOKEN_DIM,np.float32)
    feat[kind]=1.0                              # 0:4 kind one-hot
    feat[4+pidx]=1.0                           # 4:10 pair one-hot
    feat[10]=st['fn']
    feat[11]=st['sn']
    feat[12]=st['nz']
    feat[13]=dimfrac[i]
    feat[14]=dimfrac[j]
    feat[15]=float(i==j)
    feat[16]=float(np.clip(alignment,0.,1.))
    feat[17]=float(np.clip(ratio,0.,8.)/8.)
    feat[18]=st['trc']
    feat[19]=float(np.clip(st['sr'],0.,1.))
    return feat


def _second_order_path_interference(g,z,V,total_k2_ocb0_fro:float):
    """Partition K2_OC,B0 by strong intermediate component.

    Returns (piece_count, absolute_piece_sum, eta2).  The implementation is
    local, so it stays fast even for the largest b=7 families and never forms
    a dense OC x B0 matrix per component.
    """
    if z.dims[0]==0 or z.dims[2]==0:
        return 0,0.0,0.0
    strong_set={tuple(sorted(e)) for e in g.strong_edges}
    Rloc={};ZB_loc={};pos_loc={}
    for cid,comp in enumerate(g.comps):
        pos_loc[cid]={site:i for i,site in enumerate(comp)}
        Hc=_local_h0(comp,strong_set)
        ev,U=np.linalg.eigh(Hc)
        inv=np.zeros_like(ev);nz=np.abs(ev)>=1e-10;inv[nz]=1.0/ev[nz]
        Rloc[cid]=(U*inv)@U.T
        if len(comp)==4:
            zi=np.flatnonzero(np.abs(ev)<1e-10)
            ZB_loc[cid]=U[:,zi]

    edges_by_c={cid:[] for cid in range(len(g.comps))}
    for a,b in g.weak_edges:
        ca,cb=int(g.comp_id[a]),int(g.comp_id[b])
        if ca==cb:continue
        edges_by_c[ca].append((a,b,cb));edges_by_c[cb].append((b,a,ca))

    count=0;abs_sum=0.0
    for cid,comp in enumerate(g.comps):
        if len(comp)==1:continue
        rc=Rloc[cid]
        if not np.any(np.abs(rc)>1e-14):continue
        ocvec={};bvec={}
        for u,v,ncid in edges_by_c[cid]:
            iu=pos_loc[cid][u];nsz=len(g.comps[ncid])
            if nsz==1:
                ocvec.setdefault(ncid,np.zeros(len(comp)))[iu]+=1.0
            elif nsz==4:
                iv=pos_loc[ncid][v]
                bvec.setdefault(ncid,np.zeros((len(comp),2)))[iu,:]+=ZB_loc[ncid][iv,:]
        if not ocvec or not bvec:continue
        L=np.stack(list(ocvec.values()),axis=0)
        Rr=np.concatenate(list(bvec.values()),axis=1)
        A=-(L@rc@Rr)
        nrm=float(np.linalg.norm(A,'fro'))
        if nrm>1e-14:
            count+=1;abs_sum+=nrm
    eta=(float(total_k2_ocb0_fro)/abs_sum) if abs_sum>0 else 0.0
    # Numerical roundoff can very slightly exceed one.
    return count,abs_sum,float(np.clip(eta,0.,1.))


def operator_features_from_generator(M:np.ndarray)->Tuple[np.ndarray,np.ndarray,Dict[str,float]]:
    M=np.asarray(M,dtype=np.uint8)
    g=site_graph_from_generator(M)
    z=build_zero_data(g)
    V=weak_matrix(g)
    Zs=(z.ZO,z.ZI,z.ZB)
    dims=np.asarray(z.dims,dtype=np.int64)
    d0=int(dims.sum())
    dimfrac=dims.astype(np.float64)/max(float(d0),1.)

    Zall=hstack(Zs,format='csr')
    VZ=V@Zall
    RVZ=z.R@VZ
    K1=(Zall.T@VZ).tocsr()
    K2=(-(Zall.T@(V@RVZ))).tocsr()

    # K3 = PVRVRVP - 1/2 {PVP, PVR^2VP}.
    VRVZ=V@RVZ
    RVRVZ=z.R@VRVZ
    raw3=(Zall.T@(V@RVRVZ)).tocsr()
    R2VZ=z.R@RVZ
    S2=(Zall.T@(V@R2VZ)).tocsr()
    K3=(raw3-0.5*(K1@S2+S2@K1)).tocsr()

    a=T1_FIXED;b=a*a;c=a*a*a
    Keff=(a*K1+b*K2+c*K3).tocsr()
    off=np.cumsum(np.r_[0,dims])
    def blk(A,i,j):return A[off[i]:off[i+1],off[j]:off[j+1]].tocsr()

    k1tok=[];k2tok=[];k3tok=[];ketok=[]
    k1_strength=[];k2_strength=[];k3_strength=[];ke_strength=[]
    align12=[];align23=[];diag_eff=[]
    block_meta={}
    k2_ocb0_fro=0.0;k3_ocb0_fro=0.0
    for pidx,(i,j) in enumerate(PAIR_ORDER):
        di,dj=int(dims[i]),int(dims[j])
        A1,A2,A3,AE=blk(K1,i,j),blk(K2,i,j),blk(K3,i,j),blk(Keff,i,j)
        s1=_stats(A1,di,dj,True);s2=_stats(A2,di,dj,True);s3=_stats(A3,di,dj,False)
        # Avoid an additional dense SVD: use a rigorous triangle bound for the
        # Keff spectral slot and the exact sparse Frobenius norm/trace.
        se_bound=abs(a)*s1['s']+abs(b)*s2['s']+abs(c)*s3['s']
        fe=fro_norm(AE);scale=math.sqrt(max(di*dj,1));tre=_trace(AE)/max(float(di),1.) if i==j and di else 0.
        se=dict(f=fe,s=se_bound,fn=math.log1p(fe/scale) if di and dj else 0.,sn=math.log1p(se_bound),nz=float(fe>NZ_TOL),sr=float((fe*fe)/(se_bound*se_bound+EPS)/max(min(di,dj),1)) if di and dj and se_bound>0 else 0.,tr=tre,trc=_compress_trace(tre))
        al12=_cosine_abs(A1,A2,s1['f'],s2['f']);al23=_cosine_abs(A2,A3,s2['f'],s3['f'])
        r21=s2['f']/(s1['f']+1e-12);r32=s3['f']/(s2['f']+1e-12);re=s3['f']/(s1['f']+s2['f']+1e-12)
        k1tok.append(_make_token(0,pidx,dimfrac,s1,0.,0.))
        k2tok.append(_make_token(1,pidx,dimfrac,s2,al12,r21))
        k3tok.append(_make_token(2,pidx,dimfrac,s3,al23,r32))
        ketok.append(_make_token(3,pidx,dimfrac,se,0.5*(al12+al23),re))
        k1_strength.append(s1['fn']);k2_strength.append(s2['fn']);k3_strength.append(s3['fn']);ke_strength.append(se['fn'])
        align12.append(al12);align23.append(al23)
        if i==j:diag_eff.append(se['trc'])
        block_meta[f'K1_{SECTOR_NAMES[i]}_{SECTOR_NAMES[j]}_fro']=float(s1['f'])
        block_meta[f'K2_{SECTOR_NAMES[i]}_{SECTOR_NAMES[j]}_fro']=float(s2['f'])
        block_meta[f'K3_{SECTOR_NAMES[i]}_{SECTOR_NAMES[j]}_fro']=float(s3['f'])
        if (i,j)==(0,2):
            k2_ocb0_fro=float(s2['f']);k3_ocb0_fro=float(s3['f'])

    tokens=np.stack(k1tok+k2tok+k3tok+ketok).astype(np.float32)
    d01=shortest_component_distance(g,1,3);d02=shortest_component_distance(g,1,4);d12=shortest_component_distance(g,3,4)
    def dn(x):return 0.0 if x<0 else min(float(x),8.)/8.
    while len(diag_eff)<3:diag_eff.append(0.)

    pcount,pabs,eta2=_second_order_path_interference(g,z,V,k2_ocb0_fro)
    path_present=float(pcount>0)
    cancel2=(1.0-eta2) if pcount>0 else 0.0
    # Log-compressed extensive path descriptors; normalized to stay O(1).
    path_count_feat=math.log1p(float(pcount))/math.log(1025.0)
    path_abs_feat=math.log1p(float(pabs))/math.log(1025.0)

    # First nonvanishing direct OC-BULK0 effective order within our controlled
    # hierarchy.  K1 is exactly forbidden by the geometry selection rule.
    order=np.zeros(4,dtype=np.float32)
    if dims[2]==0:
        order_idx=0;order_name='no_BULK0'
    elif k2_ocb0_fro>NZ_TOL:
        order_idx=1;order_name='K2'
    elif k3_ocb0_fro>NZ_TOL:
        order_idx=2;order_name='K3'
    else:
        order_idx=3;order_name='>K3'
    order[order_idx]=1.0

    glob=np.asarray([
        *dimfrac.tolist(),
        float(np.mean(k1_strength)),float(np.mean(k2_strength)),float(np.mean(k3_strength)),float(np.mean(ke_strength)),
        float(np.mean(align12)),float(np.mean(align23)),
        dn(d01),dn(d02),dn(d12),
        float(diag_eff[0]),float(diag_eff[1]),float(diag_eff[2]),
        float(eta2),float(cancel2),float(path_present),float(path_count_feat),float(path_abs_feat),
        *order.tolist(),
    ],dtype=np.float32)
    if glob.shape!=(OP_GLOBAL_DIM,):raise RuntimeError(f'Internal v12 global feature size {glob.shape}')

    meta={
        'zdim_OC':int(dims[0]),'zdim_ICII':int(dims[1]),'zdim_B0':int(dims[2]),
        'd_OC_ICII':int(d01),'d_OC_B0':int(d02),'d_ICII_B0':int(d12),
        'K1_mean_block_strength':float(np.mean(k1_strength)),
        'K2_mean_block_strength':float(np.mean(k2_strength)),
        'K3_mean_block_strength':float(np.mean(k3_strength)),
        'Keff3_mean_block_strength':float(np.mean(ke_strength)),
        'K1K2_mean_alignment':float(np.mean(align12)),
        'K2K3_mean_alignment':float(np.mean(align23)),
        'Keff3_trace_OC':float(diag_eff[0]),'Keff3_trace_ICII':float(diag_eff[1]),'Keff3_trace_B0':float(diag_eff[2]),
        'K2_OC_B0_fro':float(k2_ocb0_fro),'K3_OC_B0_fro':float(k3_ocb0_fro),
        'K2_path_piece_count':int(pcount),'K2_path_abs_sum':float(pabs),'eta2_OC_B0':float(eta2),
        'cancel2_OC_B0':float(cancel2),'two_step_path_present':bool(pcount>0),
        'first_nonzero_OC_B0_order':order_name,
        **block_meta,
    }
    return tokens,glob,meta
