# -*- coding: utf-8 -*-
from __future__ import annotations

import json, math, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from collections import Counter, deque

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

BRANCH_LABELS = ['OC','ED-I','ED-II','IC-I','IC-II','IC-III','BULK-L','BULK-0','BULK-U']
NUM_BRANCHES = len(BRANCH_LABELS)
MAX_B = 7
ZERO_SECTOR_INDICES = (0,4,7)
ISOLATED_INDICES = (1,2,3,5,6,8)
ISOLATED_LABELS = tuple(BRANCH_LABELS[i] for i in ISOLATED_INDICES)
CHIRAL_PARTNER = np.asarray([0,2,1,5,4,3,8,7,6], dtype=np.int64)

# raw [b,N,Df] + 9 exact strong-coupling rank fractions + 6 graph-theory descriptors
GLOBAL_FEATURE_NAMES = (
    'b','N','Df',
    'rankfrac_OC','rankfrac_EDI','rankfrac_EDII','rankfrac_ICI','rankfrac_ICII','rankfrac_ICIII','rankfrac_BL','rankfrac_B0','rankfrac_BU',
    'weak13_per_site','weak34_per_site','weak14_per_site','interweak_per_site','d_OB_norm','has_bulk'
)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

@dataclass(frozen=True)
class FamilyRecord:
    family_id: str
    b: int
    N: int
    Df: float
    n_sites: int
    mask_path: Path
    state_path: Path


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id:int):
    s=torch.initial_seed()%2**32
    np.random.seed(s); random.seed(s)


def load_available_families(db_root:Path, only_b:Optional[List[int]]=None)->List[FamilyRecord]:
    manifest=db_root/'geometry_manifest.csv'
    if not manifest.exists(): raise FileNotFoundError(manifest)
    df=pd.read_csv(manifest)
    wanted=set(only_b) if only_b else None
    out=[]; missing=0
    for _,row in df.iterrows():
        b=int(row['b'])
        if wanted is not None and b not in wanted: continue
        fid=str(row['family_id'])
        mask_path=db_root/str(row['mask_file'])
        state_path=db_root/'states'/f'{fid}.npz'
        if not state_path.exists(): missing+=1; continue
        if not mask_path.exists(): raise FileNotFoundError(mask_path)
        out.append(FamilyRecord(fid,b,int(row['N']),float(row['Df']),int(row['n_sites']),mask_path,state_path))
    if not out: raise RuntimeError('No completed states/*.npz files are available.')
    print(f'Available completed families: {len(out)} (missing state files in selected scope: {missing})')
    return out


def print_family_stats(name,records):
    d={}
    for r in records: d[r.b]=d.get(r.b,0)+1
    print(f"{name}: {len(records)} families | "+', '.join(f'b={b}:{d[b]}' for b in sorted(d)))


def stratified_family_split(records,seed:int,val_fraction=.10,test_fraction=.10):
    rng=random.Random(seed); by={}
    for r in records: by.setdefault(r.b,[]).append(r)
    train=[];val=[];test=[]
    for b in sorted(by):
        g=by[b][:]; rng.shuffle(g); n=len(g)
        if n==1: train+=g; continue
        if n==2: train.append(g[0]); test.append(g[1]); continue
        if n==3: train+=g[:2]; test.append(g[2]); continue
        nt=max(1,int(round(n*test_fraction))); nv=max(1,int(round(n*val_fraction)))
        if nt+nv>=n: nv=max(0,n-nt-1)
        test+=g[:nt]; val+=g[nt:nt+nv]; train+=g[nt+nv:]
    return train,val,test


def save_split(run_dir:Path,train,val,test):
    run_dir.mkdir(parents=True,exist_ok=True)
    obj={'train':[r.family_id for r in train],'val':[r.family_id for r in val],'test':[r.family_id for r in test]}
    (run_dir/'family_split.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    rows=[]
    for s,g in [('train',train),('val',val),('test',test)]:
        for r in g: rows.append({'family_id':r.family_id,'split':s,'b':r.b,'N':r.N,'Df':r.Df,'n_sites':r.n_sites})
    pd.DataFrame(rows).to_csv(run_dir/'split_manifest.csv',index=False)


def load_split(run_dir:Path,records):
    obj=json.loads((run_dir/'family_split.json').read_text(encoding='utf-8'))
    table={r.family_id:r for r in records}
    ids=set(obj['train'])|set(obj['val'])|set(obj['test'])
    if ids!=set(table): raise RuntimeError('Current completed-family set differs from saved split. Use a fresh run directory or --refresh-split.')
    return [table[x] for x in obj['train']],[table[x] for x in obj['val']],[table[x] for x in obj['test']]

@dataclass
class EnergyScaler:
    # Chiral SSH spectrum is exactly centered at zero. Only scale is learned.
    std: float
    def transform_np(self,x): return x/self.std
    def inverse_np(self,x): return x*self.std

@dataclass
class GlobalFeatureScaler:
    mean: List[float]
    std: List[float]
    def transform_np(self,x):
        m=np.asarray(self.mean,np.float32); s=np.asarray(self.std,np.float32)
        return (x-m)/s


def fit_energy_scaler(records):
    n=0; s2=0.; s1=0.
    for r in records:
        with np.load(r.state_path,allow_pickle=False) as z:
            E=z['eigenvalues'].astype(np.float64)
        n+=len(E); s1+=float(E.sum()); s2+=float(np.square(E).sum())
    mean=s1/max(n,1)
    if abs(mean)>1e-8:
        raise RuntimeError(f'Chiral energy centering failed: pooled mean={mean:.3e}')
    return EnergyScaler(math.sqrt(max(s2/max(n,1),1e-12)))


def save_scalers(run_dir,energy,global_scaler):
    (run_dir/'scalers.json').write_text(json.dumps({'energy':asdict(energy),'global_features':asdict(global_scaler),'global_feature_names':GLOBAL_FEATURE_NAMES},indent=2),encoding='utf-8')


def load_scalers(run_dir):
    x=json.loads((run_dir/'scalers.json').read_text(encoding='utf-8'))
    return EnergyScaler(**x['energy']),GlobalFeatureScaler(**x['global_features'])


def random_d4(M,rng):
    X=np.rot90(M,int(rng.integers(0,4)))
    if bool(rng.integers(0,2)): X=np.fliplr(X)
    return np.ascontiguousarray(X)


def make_geometry_grid(M,b):
    grid=np.zeros((4,MAX_B,MAX_B),dtype=np.float32)
    grid[0,:b,:b]=M.astype(np.float32); grid[1,:b,:b]=1.
    coords=np.linspace(-1.,1.,b,dtype=np.float32) if b>1 else np.zeros(1,np.float32)
    xx,yy=np.meshgrid(coords,coords); grid[2,:b,:b]=xx; grid[3,:b,:b]=yy
    return grid


def _strong_component_graph_from_generator(M):
    M=np.asarray(M,dtype=np.uint8)
    G2=np.kron(M,M).astype(np.uint8)
    ny,nx=G2.shape
    index={}; keys=[]
    for cy in range(ny):
        for cx in range(nx):
            if not G2[cy,cx]: continue
            for sy in (0,1):
                for sx in (0,1):
                    index[(cx,cy,sx,sy)] = len(keys); keys.append((cx,cy,sx,sy))
    nsite=len(keys)
    parent=np.arange(nsite,dtype=np.int64); size=np.ones(nsite,dtype=np.int64)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return int(a)
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if size[ra]<size[rb]:ra,rb=rb,ra
        parent[rb]=ra; size[ra]+=size[rb]
    # t2 strong edges only
    for cy in range(ny):
        for cx in range(nx):
            if not G2[cy,cx]: continue
            if cx+1<nx and G2[cy,cx+1]:
                for sy in (0,1): union(index[(cx,cy,1,sy)],index[(cx+1,cy,0,sy)])
            if cy+1<ny and G2[cy+1,cx]:
                for sx in (0,1): union(index[(cx,cy,sx,1)],index[(cx,cy+1,sx,0)])
    roots={}; comp_id=np.full(nsite,-1,dtype=np.int32); comps=[]
    for i in range(nsite):
        r=find(i)
        if r not in roots: roots[r]=len(comps); comps.append([])
        c=roots[r]; comp_id[i]=c; comps[c].append(i)
    sizes=np.asarray([len(c) for c in comps],dtype=np.int16)
    unexpected=sorted(set(int(s) for s in sizes if s not in (1,2,3,4)))
    if unexpected: raise RuntimeError(f'Unexpected t1=0 component sizes: {unexpected}')
    comp_adj=[set() for _ in comps]; edge_counts=Counter(); interweak=0
    # t1 weak edges, projected to strong-component graph
    weak=[]
    for cy in range(ny):
        for cx in range(nx):
            if not G2[cy,cx]: continue
            for sy in (0,1): weak.append((index[(cx,cy,0,sy)],index[(cx,cy,1,sy)]))
            for sx in (0,1): weak.append((index[(cx,cy,sx,0)],index[(cx,cy,sx,1)]))
    for a,b in weak:
        ca,cb=int(comp_id[a]),int(comp_id[b])
        if ca==cb: continue
        interweak+=1; comp_adj[ca].add(cb); comp_adj[cb].add(ca)
        edge_counts[tuple(sorted((int(sizes[ca]),int(sizes[cb]))))]+=1
    # shortest monomer -> tetramer distance in component graph
    src=np.flatnonzero(sizes==1); targets=set(np.flatnonzero(sizes==4).tolist()); d_ob=-1
    if len(src) and targets:
        dist=np.full(len(comps),-1,dtype=np.int16); dq=deque()
        for s in src: dist[s]=0; dq.append(int(s))
        while dq:
            u=dq.popleft()
            if u in targets: d_ob=int(dist[u]); break
            for v in comp_adj[u]:
                if dist[v]<0: dist[v]=dist[u]+1; dq.append(v)
    hist=Counter(int(s) for s in sizes)
    return G2,nsite,hist,edge_counts,interweak,d_ob


def strong_coupling_component_counts_from_generator(M):
    _,nsite,hist,_,_,_=_strong_component_graph_from_generator(M)
    n1,n2,n3,n4=(int(hist.get(i,0)) for i in (1,2,3,4))
    counts={'n_monomer':n1,'n_dimer':n2,'n_trimer':n3,'n_tetramer':n4}
    ref=np.asarray([n1,n2,n2,n3,n3,n3,n4,2*n4,n4],dtype=np.float32)
    if int(round(float(ref.sum())))!=nsite: raise RuntimeError('Reference-rank completeness failure')
    return counts,ref,(ref/float(nsite)).astype(np.float32)


def theory_graph_features_from_generator(M):
    G2,nsite,hist,edges,interweak,dob=_strong_component_graph_from_generator(M)
    n1,n2,n3,n4=(int(hist.get(i,0)) for i in (1,2,3,4))
    ref=np.asarray([n1,n2,n2,n3,n3,n3,n4,2*n4,n4],dtype=np.float32)
    prior=ref/float(nsite)
    f=np.asarray([
        float(edges.get((1,3),0))/nsite,
        float(edges.get((3,4),0))/nsite,
        float(edges.get((1,4),0))/nsite,
        float(interweak)/nsite,
        0.0 if dob<0 else float(dob)/8.0,
        1.0 if n4>0 else 0.0,
    ],dtype=np.float32)
    # Selection-rule audit: direct monomer<->tetramer weak-component edge must vanish.
    if edges.get((1,4),0)!=0:
        raise RuntimeError('Theory selection rule violated: direct monomer-tetramer weak edge found')
    return prior,ref,f,{'d_OB':dob,'n1':n1,'n2':n2,'n3':n3,'n4':n4,'n_sites':nsite}


def raw_global_features(r:FamilyRecord,M):
    prior,_,tf,_=theory_graph_features_from_generator(M)
    return np.concatenate([np.asarray([r.b,r.N,r.Df],dtype=np.float32),prior.astype(np.float32),tf]).astype(np.float32)


def fit_global_scaler(records):
    xs=[]
    for r in records:
        with np.load(r.mask_path,allow_pickle=False) as z: M=z['M'].astype(np.uint8)
        xs.append(raw_global_features(r,M))
    X=np.stack(xs)
    m=X.mean(0); s=X.std(0); s=np.where(s<1e-8,1.,s)
    return GlobalFeatureScaler(m.tolist(),s.tolist())


def fit_isolated_class_weights(records,power=.10,max_weight=2.0):
    counts=np.zeros(len(ISOLATED_INDICES),dtype=np.float64)
    map9to6={k:i for i,k in enumerate(ISOLATED_INDICES)}
    for r in records:
        with np.load(r.state_path,allow_pickle=False) as z: y=z['branch_index'].astype(np.int64)
        for k9,k6 in map9to6.items(): counts[k6]+=np.sum(y==k9)
    present=counts>0; w=np.ones_like(counts)
    f=counts[present]/counts[present].sum()
    w[present]=np.power(1/np.clip(f,1e-12,None),power); w[present]/=np.mean(w[present])
    return np.clip(w,1/max_weight,max_weight).astype(np.float32),counts


def make_family_sampler(records,strength=.35):
    count={}
    for r in records: count[r.b]=count.get(r.b,0)+1
    w=[1.0/(count[r.b]**strength) for r in records]
    return WeightedRandomSampler(torch.tensor(w,dtype=torch.double),num_samples=len(records),replacement=True)


def classification_metrics_from_confusion(confusion):
    confusion=np.asarray(confusion)
    p=[];r=[];f=[]
    for k in range(confusion.shape[0]):
        tp=float(confusion[k,k]); sup=float(confusion[k,:].sum()); pred=float(confusion[:,k].sum())
        pp=tp/pred if pred>0 else np.nan; rr=tp/sup if sup>0 else np.nan
        ff=2*pp*rr/(pp+rr) if np.isfinite(pp) and np.isfinite(rr) and pp+rr>0 else (0. if sup>0 else np.nan)
        p.append(pp);r.append(rr);f.append(ff)
    valid=[x for x in f if np.isfinite(x)]
    return {'precision':np.asarray(p),'recall':np.asarray(r),'f1':np.asarray(f),'macro_f1':float(np.mean(valid)) if valid else np.nan}
