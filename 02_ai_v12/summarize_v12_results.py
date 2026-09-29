# -*- coding: utf-8 -*-
"""Collect v12 main/ablation JSON results into paper-ready CSV tables."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd


def flatten_summary(path:Path):
    s=json.loads(path.read_text(encoding='utf-8'))
    e=s.get('state_weighted_energy',{});p=s.get('primary_gauge_aware',{});v=s.get('v9_comparable_state_weighted',{})
    return {
        'run':path.parent.name,'tag':path.stem.replace('test_summary_',''),
        'checkpoint':s.get('checkpoint'),'checkpoint_epoch':s.get('checkpoint_epoch'),
        'operator_mode':s.get('operator_mode'),'projector_head_mode':s.get('projector_head_mode'),
        'energy_MAE':e.get('MAE'),'energy_R2':e.get('pooled_R2'),
        'primary_state_projector_MAE':p.get('state_projector_MAE_nondeg_or_highpurity'),
        'primary_state_zeroP_MAE':p.get('state_zero_parentage_MAE_weighted_nondeg_or_highpurity'),
        'cluster_trace_MAE':p.get('degenerate_cluster_normalized_projector_trace_MAE'),
        'cluster_zeroP_MAE':p.get('degenerate_cluster_zero_parentage_MAE'),
        'isolated6_F1':p.get('isolated6_macro_F1'),
        'v9_projector_MAE':v.get('projector_score_MAE'),
        'v9_zeroP_MAE':v.get('zero_parentage_MAE_weighted'),
        'v9_zero_mass_MAE':v.get('zero_mass_MAE'),
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=ap.parse_args()
    paths=[]
    main_run=a.root/'run_full_spectrum_operator_v12_OPERATOR_HIERARCHY'
    if main_run.exists():paths += sorted(main_run.glob('test_summary_*.json'))
    ab=a.root/'ablation_v12'
    if ab.exists():paths += sorted(ab.glob('*/test_summary_*.json'))
    if not paths:raise FileNotFoundError('No v12 test_summary_*.json files found.')
    df=pd.DataFrame([flatten_summary(p) for p in paths])
    out=a.root/'v12_paper_summary.csv';df.to_csv(out,index=False)
    print(df.to_string(index=False));print('\nSaved:',out)

if __name__=='__main__':main()
