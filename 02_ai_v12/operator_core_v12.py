# -*- coding: utf-8 -*-
"""Minimal NumPy/SciPy core for geometry-only zero-mode effective operators.

This module is used only by STEP 1.  It deliberately contains no PyTorch,
scikit-learn, pandas or plotting imports.
"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy.sparse import csr_matrix

EPS_ZERO=1.0e-10

@dataclass
class FamilyGeometry:
    family: str
    b: int
    N: int
    G2: np.ndarray
    n_sites: int
    strong_edges: List[Tuple[int,int]]
    weak_edges: List[Tuple[int,int]]
    comp_id: np.ndarray
    comps: List[List[int]]
    comp_sizes: np.ndarray
    comp_adj: List[set]
    edge_type_counts: Counter

@dataclass
class ZeroData:
    ZO: csr_matrix
    ZI: csr_matrix
    ZB: csr_matrix
    R: csr_matrix
    dims: Tuple[int,int,int]


def site_graph_from_generator(M:np.ndarray)->FamilyGeometry:
    M=np.asarray(M,dtype=np.uint8)
    if M.ndim!=2 or M.shape[0]!=M.shape[1]:
        raise ValueError(f'Generator must be square, got {M.shape}')
    b=int(M.shape[0]); N=int(M.sum()); G2=np.kron(M,M).astype(np.uint8)
    ny,nx=G2.shape
    index:Dict[Tuple[int,int,int,int],int]={}
    keys=[]
    for cy in range(ny):
        for cx in range(nx):
            if not G2[cy,cx]: continue
            for sy in (0,1):
                for sx in (0,1):
                    index[(cx,cy,sx,sy)]=len(keys); keys.append((cx,cy,sx,sy))
    n_sites=len(keys)
    strong=[]; weak=[]
    for cy in range(ny):
        for cx in range(nx):
            if not G2[cy,cx]: continue
            for sy in (0,1): weak.append((index[(cx,cy,0,sy)],index[(cx,cy,1,sy)]))
            for sx in (0,1): weak.append((index[(cx,cy,sx,0)],index[(cx,cy,sx,1)]))
            if cx+1<nx and G2[cy,cx+1]:
                for sy in (0,1): strong.append((index[(cx,cy,1,sy)],index[(cx+1,cy,0,sy)]))
            if cy+1<ny and G2[cy+1,cx]:
                for sx in (0,1): strong.append((index[(cx,cy,sx,1)],index[(cx,cy+1,sx,0)]))
    adj=[[] for _ in range(n_sites)]
    for a,c in strong:
        adj[a].append(c); adj[c].append(a)
    comp_id=np.full(n_sites,-1,dtype=np.int32); comps=[]
    for s in range(n_sites):
        if comp_id[s]>=0: continue
        cid=len(comps); q=[s]; comp_id[s]=cid; comp=[]
        for u in q:
            comp.append(u)
            for v in adj[u]:
                if comp_id[v]<0:
                    comp_id[v]=cid; q.append(v)
        comps.append(comp)
    sizes=np.asarray([len(c) for c in comps],dtype=np.int16)
    unexpected=sorted(set(int(x) for x in sizes if x not in (1,2,3,4)))
    if unexpected:
        raise RuntimeError(f'Unexpected t1=0 strong-component sizes {unexpected}')
    comp_adj=[set() for _ in comps]; edge_counts=Counter()
    for a,c in weak:
        ca,cc=int(comp_id[a]),int(comp_id[c])
        if ca==cc: continue
        comp_adj[ca].add(cc); comp_adj[cc].add(ca)
        edge_counts[tuple(sorted((int(sizes[ca]),int(sizes[cc]))))]+=1
    # The previously established zero-mode selection rule.
    if edge_counts.get((1,4),0)!=0:
        raise RuntimeError('Theory selection rule violated: direct monomer-tetramer weak edge found')
    return FamilyGeometry('in_memory',b,N,G2,n_sites,strong,weak,comp_id,comps,sizes,comp_adj,edge_counts)


def _local_h0(comp:Sequence[int], strong_set:set)->np.ndarray:
    m=len(comp); pos={g:i for i,g in enumerate(comp)}; H=np.zeros((m,m),dtype=np.float64)
    for i,g in enumerate(comp):
        for h in comp:
            if g<h and (g,h) in strong_set:
                j=pos[h]; H[i,j]=H[j,i]=-1.0
    return H


def build_zero_data(g:FamilyGeometry)->ZeroData:
    n=g.n_sites; strong_set={tuple(sorted(e)) for e in g.strong_edges}
    zrows=[[],[],[]]; zcols=[[],[],[]]; zvals=[[],[],[]]; counts=[0,0,0]
    rr=[]; cc=[]; vv=[]
    expected_zero={1:1,2:0,3:1,4:2}; sector_for_size={1:0,3:1,4:2}
    for comp in g.comps:
        sz=len(comp); Hc=_local_h0(comp,strong_set); ev,U=np.linalg.eigh(Hc)
        zi=np.flatnonzero(np.abs(ev)<EPS_ZERO)
        if len(zi)!=expected_zero[sz]:
            raise RuntimeError(f'Zero-mode multiplicity mismatch: component size={sz}, eig={ev}')
        inv=np.zeros_like(ev); nz=np.abs(ev)>=EPS_ZERO; inv[nz]=1.0/ev[nz]
        Rc=(U*inv)@U.T
        for i,gi in enumerate(comp):
            for j,gj in enumerate(comp):
                val=float(Rc[i,j])
                if abs(val)>1e-14: rr.append(gi); cc.append(gj); vv.append(val)
        if sz in sector_for_size:
            sec=sector_for_size[sz]
            for k in zi:
                col=counts[sec]; counts[sec]+=1; vec=U[:,k]
                for i,gi in enumerate(comp):
                    val=float(vec[i])
                    if abs(val)>1e-14:
                        zrows[sec].append(gi); zcols[sec].append(col); zvals[sec].append(val)
    Z=[]
    for s in range(3):
        Z.append(csr_matrix((zvals[s],(zrows[s],zcols[s])),shape=(n,counts[s])))
    R=csr_matrix((vv,(rr,cc)),shape=(n,n))
    return ZeroData(Z[0],Z[1],Z[2],R,tuple(counts))


def weak_matrix(g:FamilyGeometry)->csr_matrix:
    rr=[]; cc=[]; vv=[]
    for a,b in g.weak_edges:
        rr += [a,b]; cc += [b,a]; vv += [1.0,1.0]
    return csr_matrix((vv,(rr,cc)),shape=(g.n_sites,g.n_sites))


def fro_norm(A)->float:
    if hasattr(A,'data'):
        return float(np.sqrt(np.sum(np.abs(A.data)**2))) if A.nnz else 0.0
    return float(np.linalg.norm(A,'fro'))


def spectral_norm(A)->float:
    if min(A.shape)==0: return 0.0
    arr=A.toarray() if hasattr(A,'toarray') else np.asarray(A)
    if not np.any(arr): return 0.0
    return float(np.linalg.svd(arr,compute_uv=False)[0])


def shortest_component_distance(g:FamilyGeometry,size_a:int,size_b:int)->int:
    src=np.flatnonzero(g.comp_sizes==size_a); tar=set(np.flatnonzero(g.comp_sizes==size_b).tolist())
    if len(src)==0 or not tar: return -1
    dist=np.full(len(g.comps),-1,dtype=np.int16); dq=deque()
    for s in src: dist[s]=0; dq.append(int(s))
    while dq:
        u=dq.popleft()
        if u in tar: return int(dist[u])
        for v in g.comp_adj[u]:
            if dist[v]<0: dist[v]=dist[u]+1; dq.append(v)
    return -1
