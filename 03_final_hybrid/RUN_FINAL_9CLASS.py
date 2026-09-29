# -*- coding: utf-8 -*-
"""
Fractal SSH FINAL — K3-CANONICAL + v12-AI 9-branch classifier
==============================================================
ONE FILE TO RUN:
    python RUN_FINAL_9CLASS.py

Fixed production strategy:
  zero sector OC / IC-II / BULK-0:
      geometry-only H_eff^(3) = t1*K1 + t1^2*K2 + t1^3*K3
      + deterministic gauge canonicalization inside exact/near degeneracies
      + exact geometry rank quotas;
  isolated six branches:
      v12 best_projector predicted projector scores, restricted to the six
      isolated labels.

No residual learning is used.  The failed v13 four-coefficient residual pilot
is intentionally not part of this release.

Reference labels / true projector scores, if present in the NPZ files, are
used ONLY to compute metrics after predictions are already fixed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack

VERSION = "final-k3canonical-v12ai-9class-2026-09-13-r2"
T1_DEFAULT = -0.1
T2_FIXED = -1.0
EPS_ZERO = 1.0e-10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_ROOT = PROJECT_ROOT / "work" / "database_v7"
DEFAULT_V12_RUN = PROJECT_ROOT / "work" / "run_v12"
DEFAULT_OUT = PROJECT_ROOT / "work" / "final_9class"

BRANCHES = ["OC", "ED-I", "ED-II", "IC-I", "IC-II", "IC-III", "BULK-L", "BULK-0", "BULK-U"]
ZERO_BRANCHES = np.array([0,4,7], dtype=np.int64)
ISO_BRANCHES = np.array([1,2,3,5,6,8], dtype=np.int64)

@dataclass
class FamilyGeometry:
    b: int
    N: int
    G2: np.ndarray
    n_sites: int
    strong_edges: List[Tuple[int,int]]
    weak_edges: List[Tuple[int,int]]
    comp_id: np.ndarray
    comps: List[List[int]]
    comp_sizes: np.ndarray

@dataclass
class ZeroData:
    ZO: csr_matrix
    ZI: csr_matrix
    ZB: csr_matrix
    R: csr_matrix
    dims: Tuple[int,int,int]


def site_graph_from_generator(M: np.ndarray) -> FamilyGeometry:
    M=np.asarray(M,dtype=np.uint8)
    if M.ndim!=2 or M.shape[0]!=M.shape[1]:
        raise ValueError(f"Generator must be square, got {M.shape}")
    b=int(M.shape[0]); N=int(M.sum()); G2=np.kron(M,M).astype(np.uint8)
    ny,nx=G2.shape
    index: Dict[Tuple[int,int,int,int],int]={}; keys=[]
    for cy in range(ny):
        for cx in range(nx):
            if not G2[cy,cx]: continue
            for sy in (0,1):
                for sx in (0,1):
                    index[(cx,cy,sx,sy)] = len(keys); keys.append((cx,cy,sx,sy))
    n_sites=len(keys); strong=[]; weak=[]
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
    for a,c in strong: adj[a].append(c); adj[c].append(a)
    comp_id=np.full(n_sites,-1,dtype=np.int32); comps=[]
    for s in range(n_sites):
        if comp_id[s]>=0: continue
        cid=len(comps); q=[s]; comp_id[s]=cid; comp=[]
        for u in q:
            comp.append(u)
            for v in adj[u]:
                if comp_id[v]<0: comp_id[v]=cid; q.append(v)
        comps.append(comp)
    sizes=np.asarray([len(c) for c in comps],dtype=np.int16)
    bad=sorted(set(int(x) for x in sizes if x not in (1,2,3,4)))
    if bad: raise RuntimeError(f"Unexpected t1=0 strong-component sizes {bad}")
    return FamilyGeometry(b,N,G2,n_sites,strong,weak,comp_id,comps,sizes)


def _local_h0(comp: Sequence[int], strong_set:set) -> np.ndarray:
    m=len(comp); pos={g:i for i,g in enumerate(comp)}; H=np.zeros((m,m),float)
    for i,g in enumerate(comp):
        for h in comp:
            if g<h and (g,h) in strong_set:
                j=pos[h]; H[i,j]=H[j,i]=T2_FIXED
    return H


def build_zero_data(g:FamilyGeometry)->ZeroData:
    n=g.n_sites; strong_set={tuple(sorted(e)) for e in g.strong_edges}
    zrows=[[],[],[]]; zcols=[[],[],[]]; zvals=[[],[],[]]; counts=[0,0,0]
    rr=[];cc=[];vv=[]
    expected={1:1,2:0,3:1,4:2}; sector={1:0,3:1,4:2}
    for comp in g.comps:
        sz=len(comp); Hc=_local_h0(comp,strong_set); ev,U=np.linalg.eigh(Hc)
        zi=np.flatnonzero(np.abs(ev)<EPS_ZERO)
        if len(zi)!=expected[sz]: raise RuntimeError(f"Zero multiplicity mismatch size={sz}, eig={ev}")
        inv=np.zeros_like(ev); nz=np.abs(ev)>=EPS_ZERO; inv[nz]=1.0/ev[nz]
        Rc=(U*inv)@U.T
        for i,gi in enumerate(comp):
            for j,gj in enumerate(comp):
                v=float(Rc[i,j])
                if abs(v)>1e-14: rr.append(gi);cc.append(gj);vv.append(v)
        if sz in sector:
            s=sector[sz]
            for k in zi:
                col=counts[s];counts[s]+=1;vec=U[:,k]
                for i,gi in enumerate(comp):
                    v=float(vec[i])
                    if abs(v)>1e-14: zrows[s].append(gi);zcols[s].append(col);zvals[s].append(v)
    Z=[csr_matrix((zvals[s],(zrows[s],zcols[s])),shape=(n,counts[s])) for s in range(3)]
    R=csr_matrix((vv,(rr,cc)),shape=(n,n))
    return ZeroData(Z[0],Z[1],Z[2],R,tuple(counts))


def weak_matrix(g:FamilyGeometry)->csr_matrix:
    rr=[];cc=[];vv=[]
    for a,b in g.weak_edges:
        rr += [a,b]; cc += [b,a]; vv += [1.0,1.0]
    return csr_matrix((vv,(rr,cc)),shape=(g.n_sites,g.n_sites))


def build_k3(M:np.ndarray,t1:float=T1_DEFAULT):
    g=site_graph_from_generator(M); z=build_zero_data(g); V=weak_matrix(g)
    Z=hstack((z.ZO,z.ZI,z.ZB),format="csr"); dims=np.asarray(z.dims,dtype=np.int64)
    VZ=V@Z; RVZ=z.R@VZ
    K1=(Z.T@VZ).tocsr()
    K2=(-(Z.T@(V@RVZ))).tocsr()
    VRVZ=V@RVZ; RVRVZ=z.R@VRVZ
    raw3=(Z.T@(V@RVRVZ)).tocsr()
    R2VZ=z.R@RVZ; S2=(Z.T@(V@R2VZ)).tocsr()
    K3=(raw3-0.5*(K1@S2+S2@K1)).tocsr()
    H3=(t1*K1+(t1**2)*K2+(t1**3)*K3).tocsr()
    # Audit Hermiticity without perturbing exact sparse arithmetic.
    for name,A in (("K1",K1),("K2",K2),("K3",K3),("H3",H3)):
        D=(A-A.T).tocsr(); err=float(np.sqrt(np.square(D.data).sum())) if D.nnz else 0.0
        if err>1e-9: raise RuntimeError(f"{name} Hermiticity error {err:.3e}")
    return dims,H3,g.n_sites


def sector_weights(U:np.ndarray,dims:Sequence[int])->np.ndarray:
    d=np.asarray(dims,dtype=int); off=np.cumsum(np.r_[0,d])
    W=np.column_stack([np.square(U[off[i]:off[i+1],:]).sum(axis=0) for i in range(3)])
    W=np.clip(W,0.0,1.0); W/=np.clip(W.sum(axis=1,keepdims=True),1e-15,None)
    return W


def _clusters(e,at=1e-9,rt=1e-10):
    out=[];s=0;n=len(e)
    for i in range(1,n+1):
        close=False
        if i<n:
            a=float(e[s]);x=float(e[i]);close=abs(x-a)<=max(at,rt*max(1.0,abs(a),abs(x)))
        if i==n or not close: out.append(np.arange(s,i));s=i
    return out


def _jacobi_projector_purity(Gs,U0,max_sweeps=12,angle_tol=1e-11,flat_rel_tol=5e-13,gain_rel_tol=5e-14):
    m=U0.shape[0]
    if m<=1: return U0.copy()
    U=np.asarray(U0,dtype=float).copy(); mats=[U.T@A@U for A in Gs]; mats=[(A+A.T)*0.5 for A in mats]
    for _ in range(max_sweeps):
        nrot=0; max_angle=0.0
        for p in range(m-1):
            for q in range(p+1,m):
                M00=M01=M11=local_before=0.0
                for A in mats:
                    app=float(A[p,p]);aqq=float(A[q,q]);apq=float(A[p,q]);delta=app-aqq
                    M00+=delta*delta;M01+=2.0*delta*apq;M11+=4.0*apq*apq;local_before+=app*app+aqq*aqq
                qscale=max(1.0,abs(M00),abs(M11),abs(M01)); gap=float(np.hypot(M00-M11,2.0*M01))
                if gap<=flat_rel_tol*qscale: continue
                phi=0.5*np.arctan2(2.0*M01,M00-M11)
                ca,sa=np.cos(phi),np.sin(phi);qa=M00*ca*ca+2*M01*ca*sa+M11*sa*sa
                phib=phi+0.5*np.pi;cb,sb=np.cos(phib),np.sin(phib);qb=M00*cb*cb+2*M01*cb*sb+M11*sb*sb
                phi2=phib if qb>qa else phi
                while phi2>0.5*np.pi: phi2-=np.pi
                while phi2<-0.5*np.pi: phi2+=np.pi
                theta=0.5*phi2
                if abs(theta)<=angle_tol: continue
                c=float(np.cos(theta));ss=float(np.sin(theta));c2=c*c;s2=ss*ss;cs2=2.0*c*ss;local_after=0.0
                for A in mats:
                    app=float(A[p,p]);aqq=float(A[q,q]);apq=float(A[p,q])
                    an=c2*app+cs2*apq+s2*aqq; bn=s2*app-cs2*apq+c2*aqq
                    local_after+=an*an+bn*bn
                gain=local_after-local_before;scale=max(1.0,abs(local_before),abs(local_after))
                if gain<=gain_rel_tol*scale: continue
                for A in mats:
                    cp=A[:,p].copy();cq=A[:,q].copy();A[:,p]=c*cp+ss*cq;A[:,q]=-ss*cp+c*cq
                    rp=A[p,:].copy();rq=A[q,:].copy();A[p,:]=c*rp+ss*rq;A[q,:]=-ss*rp+c*rq
                    v=0.5*(A[p,q]+A[q,p]);A[p,q]=A[q,p]=v
                up=U[:,p].copy();uq=U[:,q].copy();U[:,p]=c*up+ss*uq;U[:,q]=-ss*up+c*uq
                nrot+=1;max_angle=max(max_angle,abs(theta))
        if nrot==0 or max_angle<=angle_tol: break
    return U


def canonicalize_degenerate_k3(e:np.ndarray,U:np.ndarray,dims:Sequence[int]):
    U=U.copy();d=np.asarray(dims,dtype=int);off=np.cumsum(np.r_[0,d]);nclusters=0;maxdeg=1
    for ids in _clusters(e):
        if len(ids)<=1: continue
        nclusters+=1;maxdeg=max(maxdeg,len(ids));V=U[:,ids];Gs=[]
        for s in range(3):
            X=V[off[s]:off[s+1],:];G=X.T@X;Gs.append((G+G.T)*0.5)
        # Distinct irrational-like sector weights provide deterministic initializer.
        Gw=0.17320508075688773*Gs[0]+1.0*Gs[1]+2.4142135623730951*Gs[2]
        _,R0=np.linalg.eigh((Gw+Gw.T)*0.5)
        R=_jacobi_projector_purity(Gs,R0,max_sweeps=12)
        V=V@R;W=sector_weights(V,dims);dom=np.argmax(W,axis=1);pur=np.max(W,axis=1)
        order=np.lexsort((-pur,dom));U[:,ids]=V[:,order]
    return U,nclusters,maxdeg


def quota3_exact(scores:np.ndarray,quotas:Sequence[int])->np.ndarray:
    """Exact O(n log n) max-score assignment for exactly three quota classes."""
    S=np.asarray(scores,float);q=np.asarray(quotas,int);n=len(S)
    if S.shape!=(n,3) or q.shape!=(3,) or int(q.sum())!=n: raise ValueError((S.shape,q))
    a=S[:,0]-S[:,2];b=S[:,1]-S[:,2];d=a-b;order=np.lexsort((np.arange(n),-d))
    pref=np.full(n+1,-np.inf);h=[];sm=0.0
    if q[0]==0: pref[:]=0.0
    else:
        for r in range(1,n+1):
            i=int(order[r-1]);item=(float(a[i]),-i,i);heapq.heappush(h,item);sm+=item[0]
            if len(h)>q[0]: sm-=heapq.heappop(h)[0]
            if len(h)==q[0]: pref[r]=sm
    suff=np.full(n+1,-np.inf);h=[];sm=0.0
    if q[1]==0: suff[:]=0.0
    else:
        for r in range(n-1,-1,-1):
            i=int(order[r]);item=(float(b[i]),-i,i);heapq.heappush(h,item);sm+=item[0]
            if len(h)>q[1]: sm-=heapq.heappop(h)[0]
            if len(h)==q[1]: suff[r]=sm
    r=int(np.argmax(pref+suff));lab=np.full(n,2,dtype=np.int64)
    if q[0]>0:
        cand=order[:r];sel=cand[np.lexsort((cand,-a[cand]))[:q[0]]];lab[sel]=0
    if q[1]>0:
        cand=order[r:];sel=cand[np.lexsort((cand,-b[cand]))[:q[1]]];lab[sel]=1
    got=np.bincount(lab,minlength=3)
    if not np.array_equal(got,q): raise RuntimeError(f"quota mismatch got={got}, expected={q}")
    return lab


def confusion(y,p,nclass):
    C=np.zeros((nclass,nclass),dtype=np.int64)
    for a,b in zip(np.asarray(y,int),np.asarray(p,int)):
        if 0<=a<nclass and 0<=b<nclass:C[a,b]+=1
    return C


def metrics(C,active=None):
    C=np.asarray(C,dtype=np.int64)
    if active is None: active=(C.sum(0)+C.sum(1))>0
    f1=[];prec=[];rec=[]
    for i in range(C.shape[0]):
        tp=int(C[i,i]);fp=int(C[:,i].sum()-tp);fn=int(C[i,:].sum()-tp)
        p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0;f=2*p*r/(p+r) if p+r else 0.0
        prec.append(p);rec.append(r);f1.append(f)
    idx=np.flatnonzero(np.asarray(active,bool))
    return {"accuracy":float(np.trace(C)/C.sum()) if C.sum() else float('nan'),"macro_f1":float(np.mean(np.asarray(f1)[idx])) if len(idx) else float('nan'),"precision":prec,"recall":rec,"f1":f1}


def _get(z,names):
    for k in names:
        if k in z.files:return z[k]
    raise KeyError(f"none of {names}; keys={z.files}")


def find_pred_dir(v12_run:Path,explicit:Path|None)->Path:
    if explicit is not None:
        p=explicit.resolve()
        if not list(p.glob('*_prediction.npz')):raise FileNotFoundError(f"No prediction NPZ in {p}")
        return p
    target='test_predictions_best_projector'
    roots=[v12_run,v12_run.parent,Path(__file__).resolve().parent]
    for r in roots:
        p=r/target
        if p.exists() and list(p.glob('*_prediction.npz')):return p.resolve()
    for r in roots[:2]:
        if r.exists():
            for p in r.rglob(target):
                if p.is_dir() and list(p.glob('*_prediction.npz')):return p.resolve()
    raise FileNotFoundError("Cannot find v12 test_predictions_best_projector. Pass --pred-dir explicitly.")


def load_mask(db:Path,family:str)->np.ndarray:
    p=db/'masks'/f'{family}.npz'
    if not p.exists():raise FileNotFoundError(p)
    with np.load(p) as z:return np.asarray(z['M'],dtype=np.uint8)


def process_family(pred_file:Path,db:Path,t1:float,outpred:Path):
    fam=pred_file.name.replace('_prediction.npz','');t0=time.time()
    with np.load(pred_file) as z:
        ps=np.asarray(_get(z,['pred_scores']),float); rd=np.asarray(_get(z,['ref_dims_geometry','ref_dims']),int)
        ytrue=None
        for k in ('legacy_hard_branch','legacy_hard'):
            if k in z.files:ytrue=np.asarray(z[k],int);break
    M=load_mask(db,fam);dims,H3,n_sites=build_k3(M,t1=t1)
    if not np.array_equal(dims,rd[ZERO_BRANCHES]):raise RuntimeError(f"{fam}: geometry dims {dims} != v12 metadata {rd[ZERO_BRANCHES]}")
    e,U=np.linalg.eigh(H3.toarray());Uc,ndeg,maxdeg=canonicalize_degenerate_k3(e,U,dims);W=sector_weights(Uc,dims)
    qlab=quota3_exact(W,dims);zero_hard=ZERO_BRANCHES[qlab]
    d0=int(dims.sum());n=len(ps)
    if (n-d0)%2:raise RuntimeError(f"{fam}: n-d0 odd")
    lo=(n-d0)//2;zids=np.arange(lo,lo+d0,dtype=np.int64);mask=np.ones(n,bool);mask[zids]=False;nids=np.flatnonzero(mask)
    iso_hard=ISO_BRANCHES[np.argmax(ps[nids][:,ISO_BRANCHES],axis=1)]
    hard=np.empty(n,dtype=np.int64);hard[zids]=zero_hard;hard[nids]=iso_hard
    outpred.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outpred/f'{fam}_prediction.npz',family=np.asarray(fam),hard9=hard,zero_state_indices=zids,zero_effective_energy=e,zero_parent_weights=W,zero_hard=zero_hard,zero_dims=dims,branch_names=np.asarray(BRANCHES))
    row={'family':fam,'n_sites':int(n_sites),'n_states':int(n),'d_OC':int(dims[0]),'d_ICII':int(dims[1]),'d_B0':int(dims[2]),'d0':d0,'degenerate_clusters':int(ndeg),'max_degeneracy':int(maxdeg),'seconds':float(time.time()-t0)}
    C9=Cz=None
    if ytrue is not None:
        true_zero=np.isin(ytrue,ZERO_BRANCHES);pred_zero=np.zeros(n,bool);pred_zero[zids]=True;row['zero_membership_errors']=int(np.sum(true_zero!=pred_zero))
        C9=confusion(ytrue,hard,9);m9=metrics(C9);row['acc9']=m9['accuracy'];row['macro_f1_9']=m9['macro_f1']
        mp={0:0,4:1,7:2};yz=np.asarray([mp.get(int(v),-1) for v in ytrue[zids]],int);pz=np.asarray([mp[int(v)] for v in zero_hard],int)
        if np.any(yz<0):raise RuntimeError(f"{fam}: central geometry zero block disagrees with reference")
        Cz=confusion(yz,pz,3);mz=metrics(Cz,active=dims>0);row['acc_zero3']=mz['accuracy'];row['macro_f1_zero3']=mz['macro_f1']
    return row,C9,Cz


def write_csv(path:Path,rows):
    if not rows:return
    keys=list(rows[0].keys())
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def write_matrix(path:Path,C,names):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['true\\pred']+list(names))
        for name,row in zip(names,C):w.writerow([name]+[int(x) for x in row])


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--db-root',type=Path,default=DEFAULT_DB_ROOT)
    p.add_argument('--v12-run-dir',type=Path,default=DEFAULT_V12_RUN)
    p.add_argument('--pred-dir',type=Path,default=None)
    p.add_argument('--outdir',type=Path,default=DEFAULT_OUT)
    p.add_argument('--t1',type=float,default=T1_DEFAULT)
    p.add_argument('--max-families',type=int,default=0,help='software smoke only; 0 = all')
    return p.parse_args()


def main():
    a=parse_args();db=a.db_root.resolve();v12=a.v12_run_dir.resolve();out=a.outdir.resolve();pred=find_pred_dir(v12,a.pred_dir)
    files=sorted(pred.glob('*_prediction.npz'))
    if a.max_families>0:files=files[:a.max_families]
    if not files:raise RuntimeError('No prediction files selected')
    out.mkdir(parents=True,exist_ok=True);outpred=out/'predictions';rows=[];C9=np.zeros((9,9),int);Cz=np.zeros((3,3),int);have=True;tall=time.time()
    print('='*112);print('FRACTAL SSH — FINAL K3-CANONICAL + v12-AI 9-CLASS');print('version :',VERSION);print('db      :',db);print('pred dir:',pred);print('out     :',out);print('families:',len(files));print('zero    : K1+K2+K3 + gauge canonicalization + exact quotas');print('isolated: v12 best_projector restricted to six isolated branches');print('='*112)
    for i,f in enumerate(files,1):
        row,c9,cz=process_family(f,db,a.t1,outpred);rows.append(row)
        if c9 is None:have=False
        else:C9+=c9;Cz+=cz
        msg=f"[{i:03d}/{len(files):03d}] {row['family']:<20s} d0={row['d0']:4d} sec={row['seconds']:.2f}"
        if 'macro_f1_9' in row:msg+=f" 9F1={row['macro_f1_9']:.5f} zero3={row['macro_f1_zero3']:.5f}"
        print(msg,flush=True)
    write_csv(out/'family_metrics.csv',rows)
    summary={'version':VERSION,'num_families':len(files),'runtime_seconds':time.time()-tall,'zero_decoder':'geometry-only H_eff^(3), deterministic parent-projector gauge canonicalization, exact geometry quotas','isolated_decoder':'v12 best_projector scores restricted to isolated six','leakage_policy':'true labels/projector scores are evaluation-only'}
    if have:
        m9=metrics(C9);mz=metrics(Cz)
        summary.update({'num_states':int(C9.sum()),'hybrid9_accuracy':m9['accuracy'],'hybrid9_macro_f1':m9['macro_f1'],'hybrid9_per_class_f1':{BRANCHES[i]:m9['f1'][i] for i in range(9)},'zero3_accuracy':mz['accuracy'],'zero3_macro_f1':mz['macro_f1'],'zero3_per_class_f1':{'OC':mz['f1'][0],'IC-II':mz['f1'][1],'BULK-0':mz['f1'][2]},'zero_membership_errors':int(sum(r.get('zero_membership_errors',0) for r in rows))})
        write_matrix(out/'confusion_9class.csv',C9,BRANCHES);write_matrix(out/'confusion_zero3.csv',Cz,['OC','IC-II','BULK-0'])
    with open(out/'summary.json','w',encoding='utf-8') as f:json.dump(summary,f,indent=2,ensure_ascii=False)
    print('\n'+'='*112)
    if have:
        print(f"ZERO3 : accuracy={summary['zero3_accuracy']:.9f}  Macro-F1={summary['zero3_macro_f1']:.9f}")
        print("        OC={:.9f}  IC-II={:.9f}  BULK-0={:.9f}".format(summary['zero3_per_class_f1']['OC'],summary['zero3_per_class_f1']['IC-II'],summary['zero3_per_class_f1']['BULK-0']))
        print(f"9-CLASS: accuracy={summary['hybrid9_accuracy']:.9f}  Macro-F1={summary['hybrid9_macro_f1']:.9f}")
        print('zero-membership errors:',summary['zero_membership_errors'])
    print('RESULTS:',out);print('FINAL CLASSIFICATION FINISHED');print('='*112)

if __name__=='__main__':main()
