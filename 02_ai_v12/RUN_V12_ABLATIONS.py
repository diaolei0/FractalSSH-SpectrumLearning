# -*- coding: utf-8 -*-
"""Controlled paper ablations for Fractal SSH v12.

Every configuration uses the same family split seed, v9-scale backbone and
operator cache.  This isolates four questions:
  1) Is operator information useful beyond geometry alone?
  2) Does K3 help beyond K1+K2?
  3) Do eta2/path/order hierarchy descriptors help beyond the matrices alone?
  4) Does gauge-factorized supervision help beyond a flat nine-way head?

Use --epochs 80 for screening and --epochs 140 for final paper numbers.
"""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
from run_full_pipeline_operator_v12_REUSE_V7_DB import find_db

HERE=Path(__file__).resolve().parent
PY=Path(sys.executable)
MODEL=HERE/'fractal_ssh_full_spectrum_operator_v12.py'


def run(cmd):
    print('\n>',' '.join(map(str,cmd)))
    subprocess.run([str(x) for x in cmd],cwd=HERE,check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db-root',type=Path,default=None)
    ap.add_argument('--operator-cache',type=Path,default=HERE/'run_full_spectrum_operator_v12_OPERATOR_HIERARCHY'/'operator_cache')
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--device',default='')
    a=ap.parse_args()
    db=find_db(a.db_root);cache=a.operator_cache.resolve()
    if not (cache/'operator_cache_audit.json').exists():
        raise FileNotFoundError(f'Run main v12 STEP 1 first or pass --operator-cache: {cache}')

    configs=[
        # name, operator mode, projector head, extra loss overrides
        ('A_factorized_no_operator','none','factorized',[]),
        ('B_factorized_K1K2','k1k2','factorized',[]),
        ('C_factorized_K1K2K3','k1k2k3','factorized',[]),
        ('D_full_matrices_no_hierarchy','full_no_hierarchy','factorized',[]),
        ('E_full_v12','full','factorized',[]),
        ('F_flat_full_v12','full','flat',[]),
        ('G_full_no_cluster_loss','full','factorized',[
            '--cluster-trace-weight','0','--zero-cluster-parentage-weight','0','--zero-cluster-mass-weight','0'
        ]),
    ]
    device=(['--device',a.device] if a.device else [])
    root=HERE/'ablation_v12';root.mkdir(exist_ok=True)
    for name,op,head,extra in configs:
        rd=root/name;rd.mkdir(exist_ok=True)
        common=['--db-root',db,'--run-dir',rd,'--operator-cache',cache,
                '--operator-mode',op,'--projector-head-mode',head,*device]
        run([PY,MODEL,'--mode','check',*common,*extra])
        train=[PY,MODEL,'--mode','train',*common,'--epochs',str(a.epochs),'--lr','3e-5',
               '--grad-accum-steps','4','--progress-every','100',*extra]
        if (rd/'last_model.pt').exists():train.append('--resume')
        run(train)
        for ck,tag in [
            ('best_v9_comparable_zero_model.pt','v9zero'),
            ('best_zero_cluster_model.pt','zerocluster'),
            ('best_model.pt','topology'),
        ]:
            run([PY,MODEL,'--mode','test','--db-root',db,'--run-dir',rd,'--operator-cache',cache,
                 '--checkpoint-name',ck,'--test-tag',tag,*device])
    print('\nAblations finished. Run summarize_v12_results.py')

if __name__=='__main__':main()
