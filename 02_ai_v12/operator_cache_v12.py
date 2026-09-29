# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np

OP_TOKEN_DIM=20
OP_NUM_TOKENS=24
OP_GLOBAL_DIM=24
CACHE_VERSION='v12-operator-hierarchy-opcache-1'

# Global positions mirror operator_features_v12.py without importing SciPy in
# the PyTorch process.
G_DIMFRAC=slice(0,3);G_MEAN_K1=3;G_MEAN_K2=4;G_MEAN_K3=5;G_MEAN_KEFF=6
G_ALIGN12=7;G_ALIGN23=8;G_D01=9;G_D02=10;G_D12=11
G_TRACE_OC=12;G_TRACE_ICII=13;G_TRACE_B0=14
G_ETA2=15;G_CANCEL2=16;G_PATH2_PRESENT=17;G_PATH2_COUNT=18;G_PATH2_ABS=19
G_ORDER_ONEHOT=slice(20,24)


def cache_file(cache_dir:Path,family_id:str)->Path:return Path(cache_dir)/f'{family_id}.npz'


def load_operator_cache(cache_dir:Path,family_id:str):
    p=cache_file(cache_dir,family_id)
    if not p.exists():raise FileNotFoundError(f'Missing operator cache for {family_id}: {p}. Run STEP 1 first.')
    with np.load(p,allow_pickle=False) as z:
        if 'cache_version' not in z.files or str(z['cache_version'].item())!=CACHE_VERSION:
            raise RuntimeError(f'{family_id}: incompatible v12 operator cache; rerun STEP 1')
        tok=z['operator_tokens'].astype(np.float32);glob=z['operator_global'].astype(np.float32)
    if tok.shape!=(OP_NUM_TOKENS,OP_TOKEN_DIM):raise RuntimeError(f'{family_id}: bad operator token shape {tok.shape}')
    if glob.shape!=(OP_GLOBAL_DIM,):raise RuntimeError(f'{family_id}: bad operator global shape {glob.shape}')
    if not np.isfinite(tok).all() or not np.isfinite(glob).all():raise RuntimeError(f'{family_id}: non-finite operator cache')
    return tok,glob


def load_operator_meta(cache_dir:Path,family_id:str):
    p=cache_file(cache_dir,family_id)
    with np.load(p,allow_pickle=False) as z:
        if 'meta_json' not in z.files:return {}
        return json.loads(str(z['meta_json'].item()))


def validate_cache_audit(cache_dir:Path,expected_families:int|None=None,db_manifest:Path|None=None):
    p=Path(cache_dir)/'operator_cache_audit.json'
    if not p.exists():raise FileNotFoundError(f'Missing cache audit: {p}')
    obj=json.loads(p.read_text(encoding='utf-8'))
    if obj.get('cache_version')!=CACHE_VERSION:raise RuntimeError('v12 operator cache audit version mismatch')
    if obj.get('failures'):raise RuntimeError(f'Operator cache audit contains failures: {len(obj["failures"])}')
    if expected_families is not None and int(obj.get('families_checked',-1))!=int(expected_families):raise RuntimeError('Operator cache family count mismatch')
    if db_manifest is not None:
        h=hashlib.sha256(Path(db_manifest).read_bytes()).hexdigest()
        if obj.get('database_manifest_sha256')!=h:raise RuntimeError('Operator cache was built for a different geometry_manifest.csv; rerun STEP 1')
    return obj
