# Fractal SSH v12 PAPER — Operator Hierarchy / Interference-Aware

这是面向论文主实验的 v12 完整版。它不改变已经锁定的数据库与物理定义，而是在 v10/v11 已验证稳定的 full-spectrum 主干上加入**三阶有效算符与路径干涉层级信息**。

## 1. 锁定不变的物理设置

- 严格自相似 C4 family，`b=3..7`，总数 816。
- `G=2`，完整 SSH unit-cell cropping。
- 有限耦合固定 `t1=-0.1, t2=-1`。
- 9 个 parent projector：`OC, ED-I, ED-II, IC-I, IC-II, IC-III, BULK-L, BULK-0, BULK-U`。
- zero manifold：`OC + IC-II + BULK-0`。
- family-level train/val/test split，禁止 eigenstate 泄漏。
- mixed exact-degenerate states 以 cluster/subspace projector trace 为主监督，而不是强迫任意 canonical eigenvector 具有唯一 parentage。

## 2. v12 新增的物理输入

在 `t1=0` parent zero subspace 中定义

```text
K1 = P V P
K2 = - P V R V P
K3 = P V R V R V P - 1/2 { P V P , P V R^2 V P }
```

并构造固定耦合三阶有效算符

```text
Keff3 = t1*K1 + t1^2*K2 + t1^3*K3,  t1=-0.1.
```

v12 进一步编码我们在 816 个正式 family 上验证出的二阶干涉效率

```text
eta2 = || sum_c A_c^(2) ||_F / sum_c || A_c^(2) ||_F
```

它把两种物理情况严格区分开：

1. 根本不存在 OC -> BULK0 二步虚路径；
2. 存在许多二步路径，但它们发生 destructive interference，净 K2 接近或等于零。

同时提供：component-graph distance、二阶路径是否存在、路径数量/绝对强度、以及 OC-BULK0 首次非零有效阶的类别 `no BULK0 / K2 / K3 / >K3`。

## 3. 默认网络

默认仍使用已经验证过速度的 v9-scale backbone：

```text
d_model=128
encoder_layers=3
operator_layers=2
query_layers=2
spectral_layers=3
n_latents=40
lr=3e-5
epochs=140
grad_accum=4
```

projector head 默认采用 factorized zero-manifold 参数化：

```text
m0 = total zero-manifold mass
p0 = conditional OC / IC-II / BULK0 parentage
piso = conditional isolated-six distribution

Czero = m0 * p0
Ciso  = (1-m0) * piso
```

并且 projector-level chiral pairing 由结构直接保证。

## 4. Windows 直接运行

解压到一个新的目录，不要覆盖 v10/v11，然后直接运行：

```text
RUN_THIS_OPERATOR_AI_v12.py
```

或：

```text
RUN_V12_WINDOWS.bat
```

程序会自动寻找完整的 v7 数据库。若自动寻找失败，可将数据库绝对路径写入 `DATABASE_PATH.txt`，也可以设置环境变量 `V7_DB`。

默认输出目录：

```text
run_full_spectrum_operator_v12_OPERATOR_HIERARCHY
```

## 5. 一键流程

```text
STEP 1  816-family database + K1/K2/K3/eta2 audit/cache
STEP 2  production forward + loss + backward check
STEP 3  140-epoch training
STEP 4  test best_model
STEP 5  test best_projector
STEP 6  test best_zero_parentage
STEP 7  test best_zero_cluster
STEP 8  test best_v9_comparable_zero
STEP 9  test best_cluster_trace
```

第一次 STEP 1 必须重新计算 v12 cache，因为 v10/v11 cache 不包含 K3/eta2；之后重新运行会复用 cache。v12 的 STEP 1 比 v10 更重，这是正常的，但不会在训练进程中导入 SciPy。

## 6. 正式 ablation

主模型跑通并建立 operator cache 后：

```text
python RUN_V12_ABLATIONS.py --epochs 80
```

筛选后最终论文表建议改成 `--epochs 140`。默认 ablation：

```text
A  factorized + no operator
B  factorized + K1/K2
C  factorized + K1/K2/K3
D  all operator matrices, but hide eta2/path/order hierarchy
E  full v12
F  flat 9-way head + full v12 operator
G  full v12 but remove cluster losses
```

这可以直接回答：增益来自 K3、来自 interference hierarchy、还是来自 gauge-factorized target。

运行后：

```text
python summarize_v12_results.py
```

生成 `v12_paper_summary.csv`。

## 7. 重要 checkpoint

除总 topology checkpoint 外，v12 单独保存：

```text
best_model.pt
best_projector_model.pt
best_zero_parentage_model.pt
best_zero_cluster_model.pt
best_v9_comparable_zero_model.pt
best_cluster_trace_model.pt
last_model.pt
```

其中 `best_v9_comparable_zero_model.pt` 是按原 v9 all-state zero-parentage 口径选出的，便于直接做 baseline table；`best_zero_cluster_model.pt` 是按 gauge-invariant cluster zero-parentage 选出的。

## 8. 中断后恢复

同一个 run directory 再运行一键脚本即可：

- operator cache 自动复用；
- 如果存在 `last_model.pt`，训练自动 `--resume`；
- split/scaler 沿用 run directory 中已经保存的文件。

## 9. 本地已经实际验证的范围

当前 release 已通过：

- `python -m py_compile *.py`；
- 两进程 NumPy/SciPy 与 PyTorch 隔离自检；
- K3 / eta2 的 D4 invariance；
- 正式 `b5_F0024_N24`：二阶路径存在但 K2、K3 OC-B0 精确抵消；
- 正式 `b7_F0003_N17`：K2 被抑制而 K3 首次打开 OC-B0；
- 9 个真实 v7 family exact mini-database：audit/cache -> forward/loss/backward -> train -> resume -> 六套 checkpoint test 全部成功；
- production-size 默认模型 forward/loss/backward 在真实 family 上通过；
- training modules 不导入 SciPy/sklearn/matplotlib。

没有在本环境完成 816-family、140 epoch 的 CUDA 长训练，因此最终性能必须以你机器上的正式 run 为准。
