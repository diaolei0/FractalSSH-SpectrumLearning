# -*- coding: utf-8 -*-
"""Two-process v12 self test.

SciPy operator construction and PyTorch are deliberately executed in separate
Python processes, matching the Windows/Anaconda production pipeline.
"""
from __future__ import annotations
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent


def run(code):
    subprocess.run([sys.executable,'-c',code],cwd=HERE,check=True)


def main():
    op=r'''
import numpy as np
from operator_features_v12 import operator_features_from_generator,OP_NUM_TOKENS,OP_TOKEN_DIM,OP_GLOBAL_DIM

# Formal b5_F0024_N24: two-step paths exist but K2 and K3 OC-B0 cancel.
M=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1]],np.uint8)
t0,g0,m0=operator_features_from_generator(M)
assert t0.shape==(OP_NUM_TOKENS,OP_TOKEN_DIM)==(24,20)
assert g0.shape==(OP_GLOBAL_DIM,)==(24,)
assert m0['K2_path_piece_count']>0
assert m0['K2_OC_B0_fro']<1e-10 and m0['K3_OC_B0_fro']<1e-10
assert m0['eta2_OC_B0']<1e-10 and m0['first_nonzero_OC_B0_order']=='>K3'
for k in range(4):
 R=np.rot90(M,k)
 for X in (R,np.fliplr(R)):
  t,g,m=operator_features_from_generator(np.ascontiguousarray(X))
  assert np.max(np.abs(t-t0))<1e-6 and np.max(np.abs(g-g0))<1e-6

# Formal b7_F0003_N17: K2 cancels, K3 opens the missing OC-B0 channel.
K=np.array([[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[1,1,1,1,1,1,1],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0]],np.uint8)
_,_,mk=operator_features_from_generator(K)
assert mk['K2_OC_B0_fro']<1e-10
assert mk['K3_OC_B0_fro']>1e-6
assert mk['first_nonzero_OC_B0_order']=='K3'
print('STEP A v12 K1/K2/K3/eta2 + D4 PASS')
'''
    run(op)

    th=r'''
import torch,numpy as np
from fractal_ssh_full_spectrum_operator_v12 import OperatorInformedFullSpectrumModel,vectorized_cluster_terms
B,N=1,64
grid=torch.zeros(B,4,7,7);grid[:,1,:3,:3]=1
gf=torch.randn(B,18);prior=torch.rand(B,9);q=torch.linspace(-1,1,N).view(B,N,1)
ot=torch.randn(B,24,20);og=torch.randn(B,24)
partner=torch.tensor([0,2,1,5,4,3,8,7,6])
for opmode in ['none','k1','k1k2','k1k2k3','full_no_hierarchy','full']:
 for head in ['flat','factorized']:
  m=OperatorInformedFullSpectrumModel(d_model=64,nhead=8,encoder_layers=1,operator_layers=1,query_layers=1,spectral_layers=1,n_latents=8,dim_feedforward=128,cnn_channels=16,operator_mode=opmode,projector_head_mode=head)
  E,pl,_=m(grid,gf,prior,q,ot,og);p=pl.softmax(-1)
  assert torch.isfinite(E).all() and torch.isfinite(pl).all()
  assert float((p.sum(-1)-1).abs().max().detach())<1e-5
  assert float((E+torch.flip(E,[1])).abs().max().detach())<1e-7
  if head=='factorized':assert float((p-torch.flip(p,[1])[...,partner]).abs().max().detach())<2e-6

true=torch.zeros(1,4,9);pred=true.clone()
true[0,0,0]=.8;true[0,0,4]=.2;true[0,1,0]=.2;true[0,1,4]=.8
pred[0,0,0]=pred[0,0,4]=pred[0,1,0]=pred[0,1,4]=.5
for x in (true,pred):x[0,2,1]=1;x[0,3,2]=1
ct,zp,zm=vectorized_cluster_terms(pred,true,torch.tensor([[0,0,2,3]]),torch.tensor([[1,1,0,0]],dtype=torch.bool))
assert max(float(ct),float(zp),float(zm))<1e-7
print('STEP B torch/chiral/factorized/vectorized-ablation PASS')
'''
    run(th)
    print('V12 TWO-PROCESS SELF TEST PASS')

if __name__=='__main__':main()
