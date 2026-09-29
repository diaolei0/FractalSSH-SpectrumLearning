# -*- coding: utf-8 -*-
"""Fractal SSH v12 PAPER K1/K2/K3 hierarchy + interference audit/cache precomputation.

This process deliberately does NOT import PyTorch.  NumPy/SciPy operator
construction is isolated from the training process to avoid duplicate OpenMP
runtimes on Windows/Anaconda.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from operator_features_v12 import operator_features_from_generator, OP_NUM_TOKENS, OP_TOKEN_DIM, OP_GLOBAL_DIM
from operator_cache_v12 import CACHE_VERSION

REQUIRED_STATE_FIELDS={'eigenvalues','q_state','branch_scores','branch_index','cluster_id','cluster_size','is_degenerate'}


def sha256_bytes(x:bytes)->str:
    return hashlib.sha256(x).hexdigest()


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def mask_fingerprint(M:np.ndarray)->str:
    M=np.ascontiguousarray(M.astype(np.uint8))
    return sha256_bytes(str(M.shape).encode()+b'|'+M.tobytes())


def load_records(db_root:Path):
    manifest=db_root/'geometry_manifest.csv'
    if not manifest.exists(): raise FileNotFoundError(manifest)
    out=[];missing=0
    with manifest.open('r',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            fid=str(row['family_id']);mask=db_root/str(row['mask_file']);state=db_root/'states'/f'{fid}.npz'
            if not state.exists(): missing+=1;continue
            if not mask.exists(): raise FileNotFoundError(mask)
            out.append((fid,int(row['b']),int(row['N']),mask,state))
    if not out: raise RuntimeError('No completed states/*.npz files are available.')
    print(f'Available completed families: {len(out)} (missing state files in selected scope: {missing})')
    return out,manifest


def audit_state_file(fid:str,state_path:Path,expected_n:int):
    with np.load(state_path,allow_pickle=False) as z:
        miss=REQUIRED_STATE_FIELDS-set(z.files)
        if miss: raise RuntimeError(f'missing fields {sorted(miss)}')
        E=z['eigenvalues'].astype(np.float64);q=z['q_state'];sc=z['branch_scores'];hard=z['branch_index']
        cid=z['cluster_id'];cs=z['cluster_size'];deg=z['is_degenerate']
        n=len(E)
        if n!=expected_n: raise RuntimeError(f'state count {n} != expected {expected_n}')
        for name,a in [('q_state',q),('branch_scores',sc),('branch_index',hard),('cluster_id',cid),('cluster_size',cs),('is_degenerate',deg)]:
            if len(a)!=n: raise RuntimeError(f'{name} length {len(a)} != {n}')
        if sc.shape!=(n,9): raise RuntimeError(f'branch_scores shape {sc.shape} != {(n,9)}')
        if not np.isfinite(E).all() or not np.isfinite(q).all() or not np.isfinite(sc).all(): raise RuntimeError('non-finite state data')
        if np.min(np.diff(E)) < -1e-9: raise RuntimeError('stored eigenvalues are not ordered')
        ch=float(np.max(np.abs(E+E[::-1]))) if n else 0.0
        if ch>1e-7: raise RuntimeError(f'chiral spectrum audit failed: {ch:.3e}')
        rs=sc.sum(1)
        if np.any(rs<=1e-12): raise RuntimeError('branch_scores contains zero-mass row')
        if float(np.max(np.abs(rs-1.0)))>1e-4: raise RuntimeError('branch_scores row normalization error exceeds 1e-4')
        if np.min(sc)<-1e-7: raise RuntimeError('branch_scores contains negative entries')
        if np.any((hard<0)|(hard>=9)): raise RuntimeError('branch_index outside 0..8')
        if np.min(cid)<0 or np.max(cid)>=n: raise RuntimeError('cluster_id outside [0,Nstate-1]; incompatible with vectorized segment aggregation')
        # cluster_size and is_degenerate must agree with the stored cluster grouping.
        unique,counts=np.unique(cid,return_counts=True); cmap={int(c):int(k) for c,k in zip(unique,counts)}
        expected_cs=np.asarray([cmap[int(c)] for c in cid])
        if np.any(np.asarray(cs,dtype=int)!=expected_cs): raise RuntimeError('cluster_size inconsistent with cluster_id')
        if np.any(np.asarray(deg,dtype=bool)!=(expected_cs>1)): raise RuntimeError('is_degenerate inconsistent with cluster_id')
        return {'stored_chiral_maxerr':ch,'branch_rowsum_maxerr':float(np.max(np.abs(rs-1.0)))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db-root',type=Path,required=True);ap.add_argument('--cache-dir',type=Path,required=True)
    ap.add_argument('--max-families',type=int,default=0);ap.add_argument('--force',action='store_true');a=ap.parse_args()
    db=a.db_root.resolve();cache=a.cache_dir.resolve();cache.mkdir(parents=True,exist_ok=True)
    rec,manifest=load_records(db);todo=rec if a.max_families<=0 else rec[:a.max_families]
    bad=[];rows=[];t=time.time();reused=0;recomputed=0;max_ch=0.0;max_rowsum=0.0
    for i,(fid,b,N,mask_path,state_path) in enumerate(todo,1):
        try:
            with np.load(mask_path,allow_pickle=False) as z: M=z['M'].astype(np.uint8)
            if M.shape!=(b,b): raise RuntimeError(f'mask M shape {M.shape} != {(b,b)}')
            if int(M.sum())!=N: raise RuntimeError(f'mask occupancy {int(M.sum())} != manifest N={N}')
            mhash=mask_fingerprint(M); out=cache/f'{fid}.npz';om={};valid_existing=False
            if out.exists() and not a.force:
                try:
                    with np.load(out,allow_pickle=False) as z:
                        valid_existing=(
                            'cache_version' in z.files and str(z['cache_version'].item())==CACHE_VERSION and
                            'mask_sha256' in z.files and str(z['mask_sha256'].item())==mhash and
                            z['operator_tokens'].shape==(OP_NUM_TOKENS,OP_TOKEN_DIM) and
                            z['operator_global'].shape==(OP_GLOBAL_DIM,) and
                            np.isfinite(z['operator_tokens']).all() and np.isfinite(z['operator_global']).all()
                        )
                        if valid_existing and 'meta_json' in z.files: om=json.loads(str(z['meta_json'].item()))
                except Exception:
                    valid_existing=False
            if valid_existing:
                reused+=1
            else:
                tok,glob,om=operator_features_from_generator(M)
                if tok.shape!=(OP_NUM_TOKENS,OP_TOKEN_DIM) or glob.shape!=(OP_GLOBAL_DIM,): raise RuntimeError('operator feature shape failure')
                if not np.isfinite(tok).all() or not np.isfinite(glob).all(): raise RuntimeError('non-finite operator features')
                np.savez_compressed(out,operator_tokens=tok.astype(np.float32),operator_global=glob.astype(np.float32),
                                    cache_version=np.asarray(CACHE_VERSION),mask_sha256=np.asarray(mhash),meta_json=np.asarray(json.dumps(om,sort_keys=True)))
                recomputed+=1
            st=audit_state_file(fid,state_path,4*N*N);max_ch=max(max_ch,st['stored_chiral_maxerr']);max_rowsum=max(max_rowsum,st['branch_rowsum_maxerr'])
            rows.append({'family_id':fid,'b':b,'N':N,'n_states':4*N*N,'cache_file':out.name,'mask_sha256':mhash,**om})
        except Exception as e:
            bad.append((fid,repr(e)))
        if i%50==0 or i==len(todo): print(f'[{i}/{len(todo)}] bad={len(bad)} reused={reused} recomputed={recomputed} elapsed={time.time()-t:.1f}s')
    fields=['family_id','b','N','n_states','cache_file','mask_sha256','zdim_OC','zdim_ICII','zdim_B0','d_OC_ICII','d_OC_B0','d_ICII_B0','K1_mean_block_strength','K2_mean_block_strength','K3_mean_block_strength','Keff3_mean_block_strength','K1K2_mean_alignment','K2K3_mean_alignment','Keff3_trace_OC','Keff3_trace_ICII','Keff3_trace_B0','K2_OC_B0_fro','K3_OC_B0_fro','K2_path_piece_count','K2_path_abs_sum','eta2_OC_B0','cancel2_OC_B0','two_step_path_present','first_nonzero_OC_B0_order']
    with (cache/'operator_cache_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary={'version':'v12-paper-operator-hierarchy-2026-09-13-r1','cache_version':CACHE_VERSION,'families_checked':len(todo),'failures':bad,'cache_dir':str(cache),
             'cache_reused':reused,'cache_recomputed':recomputed,'database_manifest_sha256':sha256_file(manifest),
             'max_stored_energy_chiral_error':max_ch,'max_branch_rowsum_error':max_rowsum,
             'operator_policy':'K1=PVP, K2=-PVRVP, K3=PVRVRVP-1/2{PVP,PVR^2VP}, Keff3=t1*K1+t1^2*K2+t1^3*K3, and eta2/path-order interference descriptors from the t1=0 parent basis; no finite-t1 full eigendecomposition',
             'runtime_policy':'NumPy/SciPy operator construction is isolated from the PyTorch training process.'}
    (cache/'operator_cache_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(summary,indent=2,ensure_ascii=False))
    if bad: raise RuntimeError(f'Operator audit failed for {len(bad)} families')

if __name__=='__main__':main()
