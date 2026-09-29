# Fractal SSH CanonICAL v7 FULL CLEAN

这是一个**从头开始、一键完整运行**的 v7 版本。它不读取 v5 / v6 的数据库标签，也不读取旧 checkpoint。

## 一键运行

在 PyCharm 里直接运行：

```text
run_full_pipeline_canonical_v7_ONECLICK.py
```

首次正常启动时必须看到：

```text
FRACTAL SSH CANONICAL v7 FULL CLEAN — RANK-CONSISTENT ZERO LABELS + DUAL HEAD
FROM SCRATCH: 首次运行不加载任何 v5 / v6 checkpoint
```

完整流程：

```text
STEP 1  物理 / canonicalization / v7 hard-label benchmark
   ↓
STEP 2  从头生成 816-family v7 数据库
   ↓
STEP 3  数据库 + 双头模型检查
   ↓
STEP 4  随机初始化，全网络训练
   ↓
STEP 5  83 个 unseen test families 正式测试
```

---

## v7 到底改了什么

### 1. Hamiltonian 没改

仍然锁定：

```text
finite    : t1 = -0.1, t2 = -1.0
reference : t1 =  0.0, t2 = -1.0
G = 2
```

几何 family、裁剪方式 `H_family = P H_parent P`、reference projectors、816-family 集合全部保持不变。

### 2. Soft projector target 不再和 hard label 混为一谈

每个有限耦合本征态仍然保存 9 维连续 projector composition：

```text
branch_scores[n, alpha] = <psi_n | P_alpha^ref | psi_n>
```

这仍然满足精确 trace / rank identity：

```text
sum_n branch_scores[n, alpha] = rank(P_alpha^ref)
```

这些连续 projector scores 是物理 soft target，本身不被 v7 hard assignment 修改。

### 3. v7 重新定义 OC / IC-II / BULK-0 的 hard truth

旧版直接用：

```text
argmax(projector score)
```

这会导致 projector rank 与 hard label 数量不一致，特别是 OC。

v7 对三个零能 reference sector：

```text
OC / IC-II / BULK-0
```

定义一个全局 rank-constrained maximum-overlap assignment：

```text
maximize    sum_n C[n, label_n]
subject to  #OC     = rank(P_OC)
            #IC-II  = rank(P_IC-II)
            #BULK-0 = rank(P_BULK-0)
```

候选 zero pool 定义为 reference zero 总 rank 对应的最小 `|E|` 中央谱区；程序还会硬性检查它是否和稳定 projector-argmax 的 zero pool 完全相同。如果不一致，数据库构建直接报错，不会静默继续。

求解使用 `scipy.optimize.linprog(method="highs")` 的 3-class transportation LP；LP 解必须是 integral，容量必须逐类精确满足，否则直接失败。

六个非零分支：

```text
ED-I / ED-II / IC-I / IC-III / BULK-L / BULK-U
```

仍保留稳定 projector argmax，因为它们在当前锁定参数下已经逐 family 与 reference rank 一致。

### 4. 两个输出 head 完全分开

模型现在输出：

```text
energy_head      -> E_n
projector_head   -> 9 维连续 projector composition
hard_branch_head -> 9 类 v7 canonical hard identity
```

soft projector loss 只作用在 projector head；hard CE 只作用在 hard head。

因此不再出现旧版“同一组 logits 一边被要求输出连续 `(0.42,0.45,0.13)`，一边又被 CE 要求输出 one-hot”的直接目标冲突。

### 5. Primary test 不使用 exact-capacity decoder

数据库 hard truth 使用 reference ranks 来**定义标签**，但模型测试时不强制容量。

正式测试：

```text
pred_branch = argmax(hard_branch_head)
```

所以最终 Precision / Recall / F1 仍然是模型自由预测结果，不会因为 decoder 被人为保证 count 正确。

### 6. q embedding 提高分辨率

Fourier frequencies 从原来的：

```text
1, 2, 4, 8, 16, 32
```

扩展到：

```text
1, 2, 4, 8, 16, 32, 64, 128
```

这是为了覆盖此前 zero-sector 中比 `1/64` 更窄的 q 结构。它不是 hard-label 定义的一部分，只是模型分辨率改进。

---

## 数据库新增字段

每个 `states/*.npz` 主要包含：

```text
branch_scores                    连续 projector target
branch_index                     v7 rank-consistent hard label
branch_index_canonical_argmax    原 canonical projector argmax，仅诊断
zero_candidate_mask              v7 zero-sector candidate pool
zero_assignment_changed          v7 assignment 是否改写了原 argmax hard label
assigned_overlap                 最终 hard label 对应的 projector overlap
assigned_counts                  v7 hard count，应与 ref_dims 完全相等
assigned_counts_canonical_argmax 原 argmax count，仅诊断
zero_assignment_regret           capacity 约束相对逐态 argmax 的总 overlap 代价
```

---

## 输出目录

数据库：

```text
fractal_ssh_topological_database_b3_b7_canonical_v7
```

正式训练结果：

```text
run_ms_cnn_transformer_canonical_v7
```

最终重点看：

```text
test_branch_metrics_best_model.csv
test_confusion_matrix_best_model.csv
test_summary_best_model.json
test_predictions_best_model/
```

`test_branch_metrics_best_model.csv` 还会额外给出：

```text
support
predicted
precision
recall
F1
projector_MAE
count_error
count_rel_error
```

其中 `support` 是 v7 rank-consistent truth；`predicted` 是 hard head 的自由预测数量。

---

## 正式默认训练参数

```text
epochs                  = 160
batch size              = 8
queries / family        = 96
learning rate           = 3e-5
uniform query fraction  = 0.65
class-balance power     = 0.25
projector loss weight   = 0.20
hard CE weight          = 0.20
subspace loss weight    = 0.10
projector rank weight   = 0.08
hard-prob rank weight   = 0.04
monotonic weight        = 0.05
```

训练 query 中 65% 保持 uniform；剩余 35% 按 family 内实际存在的 hard classes 近似均衡抽样。只有 uniform 部分参与 energy / rank 的无偏估计。

---

## 续跑

数据库生成支持断点续跑，已有完整 family 自动跳过。

如果同一个 v7 结果目录已经存在：

```text
run_ms_cnn_transformer_canonical_v7/last_model.pt
```

再次运行一键脚本会自动 `--resume`。

第一次运行不会寻找或加载任何 v5/v6 checkpoint。

---

## 本地已完成的验证

在打包前已完成：

1. Sierpiński G1 / G2 / Vicsek G2 严格 physics benchmark。
2. canonical gauge stress test；canonical score error 为 `~1e-15`。
3. v7 hard branch count 与 reference rank 完全一致。
4. v7 zero transportation assignment 的 row-permutation invariance 检查。
5. b=3 全部 family 数据库生成。
6. b=4 全部 family 数据库生成。
7. 一批 b=5 family 数据库生成。
8. 双头模型 `--mode check`。
9. 从随机初始化进行 1 epoch smoke training。
10. `--mode test` 完整 smoke inference / CSV / JSON 输出。

这些只是代码正确性 smoke test，不代表正式 v7 的最终论文指标；正式指标必须等完整 816-family + 160 epoch 训练结束后再评价。
