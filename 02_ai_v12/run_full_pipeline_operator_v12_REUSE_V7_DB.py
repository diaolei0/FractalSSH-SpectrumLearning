# -*- coding: utf-8 -*-
"""One-click Fractal SSH v12 PAPER pipeline.

Default configuration
---------------------
* same v9-scale backbone used by the FAST v10 / v11 paper versions;
* factorized zero-manifold projector head;
* K1/K2/K3 + third-order Keff operator tokens;
* eta2 / path-presence / first-nonzero-order hierarchy descriptors;
* vectorized gauge-aware degenerate-cluster supervision.
"""
from __future__ import annotations
import argparse,csv,os,subprocess,sys
from pathlib import Path
from datetime import datetime

HERE=Path(__file__).resolve().parent
PY=Path(sys.executable)
MODEL=HERE/'fractal_ssh_full_spectrum_operator_v12.py'
AUDIT=HERE/'validate_operator_informed_v12.py'
DEFAULT_RUN=HERE/'run_full_spectrum_operator_v12_OPERATOR_HIERARCHY'
LOG=HERE/'one_click_operator_v12.log'
EPOCHS=140
LR=3e-5
NUM_WORKERS=0


def db_is_complete(p:Path)->bool:
    try:
        man=p/'geometry_manifest.csv';states=p/'states'
        if not man.exists() or not states.is_dir():return False
        with man.open('r',encoding='utf-8-sig',newline='') as f:n=sum(1 for _ in csv.DictReader(f))
        return n>=816 and len(list(states.glob('*.npz')))>=816
    except Exception:return False


def find_db(explicit:Path|None=None):
    if explicit is not None:
        p=explicit.expanduser().resolve()
        if db_is_complete(p):return p
        raise FileNotFoundError(f'指定数据库不完整或路径错误: {p}')
    for env in ('V7_DB','V8_DB','FRACTAL_SSH_DB'):
        x=os.environ.get(env,'').strip()
        if x and db_is_complete(Path(x)):return Path(x).expanduser().resolve()
    txt=HERE/'DATABASE_PATH.txt'
    if txt.exists():
        lines=[ln.strip() for ln in txt.read_text(encoding='utf-8-sig').splitlines()]
        x=next((ln.strip('"') for ln in lines if ln and not ln.startswith('#')),'')
        if x and db_is_complete(Path(x)):return Path(x).expanduser().resolve()
    names=('fractal_ssh_topological_database_b3_b7_canonical_v7','fractal_ssh_topological_database_b3_b7_canonical_v8')
    roots=[];p=HERE
    for _ in range(7):
        roots.append(p)
        if p.parent==p:break
        p=p.parent
    seen=set()
    for root in roots:
        for name in names:
            for c in [
                root/name,
                root/'Fractal_SSH_Canonical_v7_FULL_CLEAN_oneclick'/name,
                root/'v10'/'Fractal_SSH_Canonical_v7_FULL_CLEAN_oneclick'/name,
            ]:
                try:r=c.resolve()
                except Exception:continue
                if str(r) in seen:continue
                seen.add(str(r))
                if db_is_complete(r):return r
    project_root=next((r for r in roots if r.name=='fractal_ssh_full_from_scratch_oc_v2'),None)
    if project_root:
        for name in names:
            for c in project_root.glob(f'**/{name}'):
                if db_is_complete(c):return c.resolve()
    raise FileNotFoundError('找不到完整 816-family v7/v8 数据库。可在 DATABASE_PATH.txt 写入路径，或设置 V7_DB。')


def run(cmd,title):
    s=f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {title}'
    print('\n'+'='*110+'\n'+s+'\n'+'='*110)
    with LOG.open('a',encoding='utf-8') as f:
        f.write('\n'+'='*110+'\n'+s+'\n'+'='*110+'\n');f.flush()
        p=subprocess.Popen([str(x) for x in cmd],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in p.stdout:
            print(line,end='');f.write(line);f.flush()
        rc=p.wait()
    if rc:raise RuntimeError(f'{title} failed, rc={rc}')


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--db-root',type=Path,default=None)
    ap.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    ap.add_argument('--operator-cache',type=Path,default=None)
    ap.add_argument('--epochs',type=int,default=EPOCHS)
    ap.add_argument('--progress-every',type=int,default=50)
    ap.add_argument('--force-operator-cache',action='store_true')
    ap.add_argument('--device',default=os.environ.get('V12_DEVICE','').strip())
    ap.add_argument('--operator-mode',choices=['none','k1','k1k2','k1k2k3','full_no_hierarchy','full'],default='full')
    ap.add_argument('--projector-head-mode',choices=['flat','factorized'],default='factorized')
    a=ap.parse_args(argv)

    LOG.write_text('',encoding='utf-8')
    db=find_db(a.db_root)
    run_dir=a.run_dir.resolve()
    op_cache=(a.operator_cache.resolve() if a.operator_cache is not None else run_dir/'operator_cache')
    run_dir.mkdir(parents=True,exist_ok=True)
    print('FRACTAL SSH v12 PAPER — K1/K2/K3 operator hierarchy / eta2 interference / gauge-aware')
    print('DB :',db);print('RUN:',run_dir)
    print('CONFIG: operator=',a.operator_mode,'projector_head=',a.projector_head_mode)

    audit=[PY,AUDIT,'--db-root',db,'--cache-dir',op_cache]
    if a.force_operator_cache:audit.append('--force')
    run(audit,'STEP 1/9 full 816-family database + v12 K1/K2/K3/eta2 audit/cache')

    device_args=(['--device',a.device] if a.device else [])
    common=['--db-root',db,'--run-dir',run_dir,'--operator-cache',op_cache,
            '--operator-mode',a.operator_mode,'--projector-head-mode',a.projector_head_mode,*device_args]
    run([PY,MODEL,'--mode','check',*common],'STEP 2/9 full forward + hierarchy/factorized loss + backward wiring check')

    cmd=[PY,MODEL,'--mode','train',*common,'--epochs',str(a.epochs),'--lr',str(LR),
         '--num-workers',str(NUM_WORKERS),'--grad-accum-steps','4','--progress-every',str(a.progress_every)]
    if (run_dir/'last_model.pt').exists():cmd.append('--resume')
    init=os.environ.get('V12_INIT_CKPT','').strip() or os.environ.get('V11_INIT_CKPT','').strip() or os.environ.get('V9_CKPT','').strip()
    local=HERE/'best_model_v9.pt'
    if not (run_dir/'last_model.pt').exists() and (init or local.exists()):
        cmd += ['--init-v9',str(Path(init).resolve() if init else local.resolve())]
    run(cmd,'STEP 3/9 v12 OPERATOR-HIERARCHY training')

    tests=[
        ('best_model.pt','best_model'),
        ('best_projector_model.pt','best_projector'),
        ('best_zero_parentage_model.pt','best_zero_parentage'),
        ('best_zero_cluster_model.pt','best_zero_cluster'),
        ('best_v9_comparable_zero_model.pt','best_v9_comparable_zero'),
        ('best_cluster_trace_model.pt','best_cluster_trace'),
    ]
    for j,(ck,tag) in enumerate(tests,4):
        run([PY,MODEL,'--mode','test','--db-root',db,'--run-dir',run_dir,'--operator-cache',op_cache,
             '--checkpoint-name',ck,'--test-tag',tag,*device_args],f'STEP {j}/9 test {tag}')
    print('\nV12 OPERATOR HIERARCHY FINISHED:',run_dir)

if __name__=='__main__':main()
