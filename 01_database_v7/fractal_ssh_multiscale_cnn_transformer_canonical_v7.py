# -*- coding: utf-8 -*-
"""
Fractal SSH MultiScaleCNN-Transformer v7 Dual-Head Predictor FULL CLEAN
===================================================================

Paper-1 forward task
--------------------
    (generator M, b, N, Df, normalized state coordinate q_n)
        -> finite-coupling eigenenergy E_n
        -> 9 reference-projector branch scores

This script is designed for the database produced by:
    build_fractal_ssh_topological_database_b3_b7_canonical_v7.py

Locked physics in that database
--------------------------------
    G = 2
    finite:    t1 = -0.1, t2 = -1.0
    reference: t1 =  0.0, t2 = -1.0
    H_family = P H_parent P

Why this is not the old fixed-length autoregressive decoder
-----------------------------------------------------------
The previous acoustic code predicted a fixed 63-point sequence. Here the
number of SSH eigenstates depends on the fractal generator:

    N_E = 4 N^2,  N = sum(M)

so output length is variable. We therefore use q_n=(n+0.5)/N_E as a query.
The geometry is encoded once; arbitrary state queries then cross-attend to
that geometry memory. Full spectra are reconstructed by evaluating all q_n.

Architecture
------------
1. 7x7 padded generator representation with 4 channels:
       M, valid mask, x coordinate, y coordinate
2. 2D multi-scale CNN with kernels 3x3 / 5x5 / 7x7
3. Transformer geometry encoder
4. Fourier embedding of q_n
5. Transformer-style cross-attention query blocks
6. Geometry-derived strong-coupling reference-rank prior from monomer/dimer/trimer/tetramer motifs
7. Two heads:
       energy head -> E_n
       branch head -> 9 continuous projector-score probabilities
8. Exact degeneracies use projector-purity canonical states and deterministic within-cluster branch ordering; invariant cluster weights remain an audit target

Important data rule
-------------------
Train/validation/test splits are by FAMILY ID, never by individual states.
All states of one Hamiltonian belong to exactly one split.

Modes
-----
check    : verify database/model wiring
train    : train and save best/last checkpoints
test     : evaluate every full spectrum in the test split
predict  : predict one full family spectrum

Examples
--------
python fractal_ssh_multiscale_cnn_transformer_canonical_v7.py \
    --mode check \
    --db-root ./fractal_ssh_topological_database_b3_b7_canonical_v7

python fractal_ssh_multiscale_cnn_transformer_canonical_v7.py \
    --mode train \
    --db-root ./fractal_ssh_topological_database_b3_b7_canonical_v7 \
    --run-dir ./run_ms_cnn_transformer_canonical_v7 \
    --train-backbone \
    --epochs 150

python fractal_ssh_multiscale_cnn_transformer_canonical_v7.py \
    --mode test \
    --db-root ./fractal_ssh_topological_database_b3_b7_canonical_v7 \
    --run-dir ./run_ms_cnn_transformer_canonical_v7

python fractal_ssh_multiscale_cnn_transformer_canonical_v7.py \
    --mode predict \
    --db-root ./fractal_ssh_topological_database_b3_b7_canonical_v7 \
    --run-dir ./run_ms_cnn_transformer_canonical_v7 \
    --family-id b5_F0005_N16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 0. LOCKED LABEL ORDER
# =============================================================================

BRANCH_LABELS = [
    "OC",
    "ED-I",
    "ED-II",
    "IC-I",
    "IC-II",
    "IC-III",
    "BULK-L",
    "BULK-0",
    "BULK-U",
]
NUM_BRANCHES = len(BRANCH_LABELS)
MAX_B = 7

OC_INDEX = BRANCH_LABELS.index("OC")
ZERO_SECTOR_INDICES = (
    BRANCH_LABELS.index("OC"),
    BRANCH_LABELS.index("IC-II"),
    BRANCH_LABELS.index("BULK-0"),
)


# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# 2. DATABASE RECORDS
# =============================================================================

@dataclass(frozen=True)
class FamilyRecord:
    family_id: str
    b: int
    N: int
    Df: float
    n_sites: int
    mask_path: Path
    state_path: Path


def load_available_families(
    db_root: Path,
    only_b: Optional[List[int]] = None,
) -> List[FamilyRecord]:
    manifest_path = db_root / "geometry_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing: {manifest_path}")

    df = pd.read_csv(manifest_path)
    wanted_b = set(only_b) if only_b else None

    records: List[FamilyRecord] = []
    missing_states = 0

    for _, row in df.iterrows():
        b = int(row["b"])
        if wanted_b is not None and b not in wanted_b:
            continue

        fid = str(row["family_id"])
        mask_path = db_root / str(row["mask_file"])
        state_path = db_root / "states" / f"{fid}.npz"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")
        if not state_path.exists():
            missing_states += 1
            continue

        records.append(
            FamilyRecord(
                family_id=fid,
                b=b,
                N=int(row["N"]),
                Df=float(row["Df"]),
                n_sites=int(row["n_sites"]),
                mask_path=mask_path,
                state_path=state_path,
            )
        )

    if not records:
        raise RuntimeError("No completed states/*.npz files are available.")

    print(
        f"Available completed families: {len(records)} "
        f"(missing state files in selected scope: {missing_states})"
    )
    return records


def print_family_stats(name: str, records: List[FamilyRecord]):
    by_b: Dict[int, int] = {}
    for r in records:
        by_b[r.b] = by_b.get(r.b, 0) + 1
    detail = ", ".join(f"b={b}:{by_b[b]}" for b in sorted(by_b))
    print(f"{name}: {len(records)} families | {detail}")


# =============================================================================
# 3. FAMILY-LEVEL TRAIN / VAL / TEST SPLIT
# =============================================================================

def stratified_family_split(
    records: List[FamilyRecord],
    seed: int,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
):
    """Approximate stratification by b, while keeping each family intact."""
    rng = random.Random(seed)
    by_b: Dict[int, List[FamilyRecord]] = {}
    for r in records:
        by_b.setdefault(r.b, []).append(r)

    train: List[FamilyRecord] = []
    val: List[FamilyRecord] = []
    test: List[FamilyRecord] = []

    for b in sorted(by_b):
        group = by_b[b][:]
        rng.shuffle(group)
        n = len(group)

        if n == 1:
            train.extend(group)
            continue
        if n == 2:
            train.append(group[0])
            test.append(group[1])
            continue
        if n == 3:
            train.extend(group[:2])
            test.append(group[2])
            continue

        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_test + n_val >= n:
            n_val = max(0, n - n_test - 1)

        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])

    return train, val, test


def save_split(run_dir: Path, train, val, test):
    obj = {
        "train": [r.family_id for r in train],
        "val": [r.family_id for r in val],
        "test": [r.family_id for r in test],
    }
    with (run_dir / "family_split.json").open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    rows = []
    for split_name, group in (("train", train), ("val", val), ("test", test)):
        for r in group:
            rows.append({
                "family_id": r.family_id,
                "split": split_name,
                "b": r.b,
                "N": r.N,
                "Df": r.Df,
                "n_sites": r.n_sites,
            })
    pd.DataFrame(rows).to_csv(run_dir / "split_manifest.csv", index=False)


def load_split(run_dir: Path, records: List[FamilyRecord]):
    path = run_dir / "family_split.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    table = {r.family_id: r for r in records}
    split_ids = set(obj["train"]) | set(obj["val"]) | set(obj["test"])
    current_ids = set(table)

    if split_ids != current_ids:
        missing_now = sorted(split_ids - current_ids)
        new_now = sorted(current_ids - split_ids)
        raise RuntimeError(
            "Current completed-family set differs from the saved split.\n"
            f"Missing from current DB: {len(missing_now)}\n"
            f"New completed families: {len(new_now)}\n"
            "For a fresh training run use --refresh-split."
        )

    def pick(ids):
        return [table[x] for x in ids]

    return pick(obj["train"]), pick(obj["val"]), pick(obj["test"])


# =============================================================================
# 4. TRAIN-ONLY SCALERS
# =============================================================================

@dataclass
class EnergyScaler:
    mean: float
    std: float

    def transform_np(self, x):
        return (x - self.mean) / self.std

    def inverse_np(self, x):
        return x * self.std + self.mean


@dataclass
class GlobalFeatureScaler:
    mean: List[float]
    std: List[float]

    def transform_np(self, x: np.ndarray):
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (x - mean) / std


def fit_energy_scaler(records: List[FamilyRecord]) -> EnergyScaler:
    n_total = 0
    sum1 = 0.0
    sum2 = 0.0

    for r in records:
        with np.load(r.state_path, allow_pickle=False) as z:
            E = z["eigenvalues"].astype(np.float64)
        n_total += len(E)
        sum1 += float(E.sum())
        sum2 += float(np.square(E).sum())

    mean = sum1 / n_total
    var = max(sum2 / n_total - mean * mean, 1e-12)
    return EnergyScaler(mean=mean, std=math.sqrt(var))


def raw_global_features(r: FamilyRecord):
    return np.asarray([r.b, r.N, r.Df], dtype=np.float32)


def fit_global_scaler(records: List[FamilyRecord]) -> GlobalFeatureScaler:
    X = np.stack([raw_global_features(r) for r in records], axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return GlobalFeatureScaler(mean=mean.tolist(), std=std.tolist())


def save_scalers(run_dir: Path, energy: EnergyScaler, global_scaler: GlobalFeatureScaler):
    with (run_dir / "scalers.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "energy": asdict(energy),
                "global_features": asdict(global_scaler),
            },
            f,
            indent=2,
        )


def load_scalers(run_dir: Path):
    with (run_dir / "scalers.json").open("r", encoding="utf-8") as f:
        x = json.load(f)
    return (
        EnergyScaler(**x["energy"]),
        GlobalFeatureScaler(**x["global_features"]),
    )


def fit_branch_class_weights(
    records: List[FamilyRecord],
    power: float = 0.5,
    max_weight: float = 4.0,
):
    """Training-split-only class weights for canonical hard labels.

    Uses inverse-frequency**power, normalized to mean 1 over present classes.
    No validation/test labels enter this calculation.
    """
    counts = np.zeros(NUM_BRANCHES, dtype=np.float64)
    for r in records:
        with np.load(r.state_path, allow_pickle=False) as z:
            y = z["branch_index"].astype(np.int64)
        counts += np.bincount(y, minlength=NUM_BRANCHES)
    present = counts > 0
    if not np.any(present):
        return np.ones(NUM_BRANCHES, dtype=np.float32), counts
    freq = counts[present] / counts[present].sum()
    w = np.ones(NUM_BRANCHES, dtype=np.float64)
    w[present] = np.power(1.0 / np.clip(freq, 1e-12, None), float(power))
    w[present] /= np.mean(w[present])
    w = np.clip(w, 1.0 / float(max_weight), float(max_weight))
    return w.astype(np.float32), counts


# =============================================================================
# 5. GEOMETRY REPRESENTATION + D4 AUGMENTATION
# =============================================================================

def random_d4(M: np.ndarray, rng: np.random.Generator):
    X = np.rot90(M, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        X = np.fliplr(X)
    return np.ascontiguousarray(X)


def make_geometry_grid(M: np.ndarray, b: int):
    """
    Output [4,7,7]:
      ch0 = M
      ch1 = valid bxb region mask
      ch2 = normalized x coordinate inside bxb
      ch3 = normalized y coordinate inside bxb
    """
    grid = np.zeros((4, MAX_B, MAX_B), dtype=np.float32)
    grid[0, :b, :b] = M.astype(np.float32)
    grid[1, :b, :b] = 1.0

    coords = np.linspace(-1.0, 1.0, b, dtype=np.float32) if b > 1 else np.zeros(1, np.float32)
    xx, yy = np.meshgrid(coords, coords)
    grid[2, :b, :b] = xx
    grid[3, :b, :b] = yy
    return grid


# =============================================================================
# 5B. EXACT STRONG-COUPLING TOPOLOGY PRIOR FROM GEOMETRY
# =============================================================================

def strong_coupling_component_counts_from_generator(M: np.ndarray):
    """
    Derive the t1=0 strong-bond connected-component counts directly from M.

    This is a geometry-only calculation and does NOT use finite-coupling
    eigenvalues/eigenvectors or branch labels.  For G=2,

        G2 = M \\otimes M

    and every retained G2 SSH cell contributes four sites.  Only t2 bonds are
    present in the reference limit.  Connected components of size 1/2/3/4 are
    respectively monomer/dimer/trimer/tetramer motifs.

    Returns
    -------
    counts : dict
        n_monomer, n_dimer, n_trimer, n_tetramer
    ref_dims : np.ndarray, shape (9,)
        Exact reference-projector ranks in BRANCH_LABELS order:
        [n1, n2, n2, n3, n3, n3, n4, 2*n4, n4]
    prior : np.ndarray, shape (9,)
        ref_dims / n_sites; sums to one.
    """
    M = np.asarray(M, dtype=np.uint8)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2D, got shape={M.shape}")

    G2 = np.kron(M, M).astype(np.uint8)
    ys, xs = np.where(G2 > 0)

    # Site index only for retained complete SSH cells.
    index = {}
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        for sy in (0, 1):
            for sx in (0, 1):
                index[(cx, cy, sx, sy)] = len(index)

    nsite = len(index)
    if nsite == 0:
        raise RuntimeError("Degenerate generator: no retained G2 SSH sites.")

    parent = np.arange(nsite, dtype=np.int64)
    size = np.ones(nsite, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    Ny, Nx = G2.shape

    # Horizontal t2 bonds.
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        if cx + 1 < Nx and G2[cy, cx + 1]:
            for sy in (0, 1):
                a = index[(cx, cy, 1, sy)]
                b = index[(cx + 1, cy, 0, sy)]
                union(a, b)

        # Vertical t2 bonds.
        if cy + 1 < Ny and G2[cy + 1, cx]:
            for sx in (0, 1):
                a = index[(cx, cy, sx, 1)]
                b = index[(cx, cy + 1, sx, 0)]
                union(a, b)

    component_sizes = {}
    for i in range(nsite):
        r = find(i)
        component_sizes[r] = component_sizes.get(r, 0) + 1

    hist = {}
    for s in component_sizes.values():
        hist[s] = hist.get(s, 0) + 1

    unexpected = sorted(s for s in hist if s not in (1, 2, 3, 4))
    if unexpected:
        raise RuntimeError(
            "Unexpected t1=0 component size(s) from geometry: "
            f"{unexpected}. This v7 expects the locked b=3..7 strict family."
        )

    n1 = int(hist.get(1, 0))
    n2 = int(hist.get(2, 0))
    n3 = int(hist.get(3, 0))
    n4 = int(hist.get(4, 0))

    counts = {
        "n_monomer": n1,
        "n_dimer": n2,
        "n_trimer": n3,
        "n_tetramer": n4,
    }

    ref_dims = np.asarray(
        [n1, n2, n2, n3, n3, n3, n4, 2 * n4, n4],
        dtype=np.float32,
    )

    if int(round(float(ref_dims.sum()))) != nsite:
        raise RuntimeError(
            "Reference-rank completeness failure: "
            f"sum(ref_dims)={ref_dims.sum()}, nsite={nsite}"
        )

    prior = ref_dims / float(nsite)
    return counts, ref_dims, prior.astype(np.float32)


def topology_prior_from_generator(M: np.ndarray):
    """Return the normalized 9-branch reference-rank prior derived from M."""
    return strong_coupling_component_counts_from_generator(M)[2]


# =============================================================================
# 6. DATASET
# =============================================================================

class FamilyQueryDataset(Dataset):
    """One item = one family + K spectral queries.

    v7 target policy
    ----------------
    ``branch_scores`` is the canonical state-resolved projector target.
      * singleton eigenstate -> ordinary state projector scores;
      * strictly degenerate cluster -> canonicalized eigenvectors obtained from
        the fixed reference-projector gauge operator in the v7 database.
      * invariant cluster compositions are retained for an auxiliary audit loss.

    Training queries contain a uniform subset (for unbiased rank conservation)
    and, when available, a cluster-balanced degenerate subset.
    """

    def __init__(
        self,
        records: List[FamilyRecord],
        energy_scaler: EnergyScaler,
        global_scaler: GlobalFeatureScaler,
        queries_per_family: int,
        training: bool,
        d4_augment: bool,
        seed: int,
        verify_reference_dims: bool = True,
        uniform_query_fraction: float = 0.70,
        zero_focus_power: float = 1.0,  # retained CLI compatibility; unused in v7
    ):
        self.records = records
        self.energy_scaler = energy_scaler
        self.global_scaler = global_scaler
        self.K = int(queries_per_family)
        self.training = training
        self.d4_augment = d4_augment
        self.seed = int(seed)
        self.epoch = 0
        self.uniform_query_fraction = float(uniform_query_fraction)
        if not (0.0 < self.uniform_query_fraction <= 1.0):
            raise ValueError("uniform_query_fraction must be in (0,1].")

        self.topology_prior = {}
        self.ref_dims_geometry = {}
        self.component_counts = {}

        for r in self.records:
            with np.load(r.mask_path, allow_pickle=False) as z:
                M = z["M"].astype(np.uint8)
            counts, ref_dims, prior = strong_coupling_component_counts_from_generator(M)
            self.topology_prior[r.family_id] = prior
            self.ref_dims_geometry[r.family_id] = ref_dims.astype(np.int64)
            self.component_counts[r.family_id] = counts

            if verify_reference_dims:
                with np.load(r.state_path, allow_pickle=False) as z:
                    required = {"branch_scores", "cluster_id", "cluster_size", "is_degenerate", "canonicalized", "branch_index", "branch_index_canonical_argmax", "zero_candidate_mask"}
                    missing = required - set(z.files)
                    if missing:
                        raise RuntimeError(
                            f"{r.family_id} is not a v7 rank-consistent state file; missing {sorted(missing)}"
                        )
                    if "ref_dims" in z.files:
                        ref_dims_db = z["ref_dims"].astype(np.int64)
                        if not np.array_equal(ref_dims_db, ref_dims.astype(np.int64)):
                            raise RuntimeError(
                                f"Geometry-derived ref_dims disagree with database for {r.family_id}: "
                                f"geometry={ref_dims.tolist()}, database={ref_dims_db.tolist()}"
                            )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def _sample_training_indices(self, rng, cluster_id, is_degenerate, hard_branch):
        """Mixed query sampler for v7.

        The uniform part is the only part used for energy/rank estimation. The
        focused part samples present hard classes approximately uniformly so OC
        is actually seen during training without introducing class weights twice.
        """
        nstate = len(cluster_id)
        ku = max(1, int(round(self.K * self.uniform_query_fraction)))
        ku = min(ku, self.K)
        uniform_inds = rng.choice(nstate, size=ku, replace=(nstate < ku)).astype(np.int64)

        kfocus = self.K - ku
        focus = []
        if kfocus > 0:
            present = np.unique(hard_branch).astype(np.int64)
            chosen_classes = rng.choice(present, size=kfocus, replace=True)
            for cls in chosen_classes:
                members = np.flatnonzero(hard_branch == cls)
                focus.append(int(rng.choice(members)))

        inds = np.concatenate([uniform_inds, np.asarray(focus, dtype=np.int64)])
        uniform_mask = np.concatenate([
            np.ones(ku, dtype=np.float32),
            np.zeros(len(inds) - ku, dtype=np.float32),
        ])
        order = np.argsort(inds, kind="stable")
        return inds[order], uniform_mask[order]

    def __getitem__(self, idx):
        r = self.records[idx]
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + 7919 * idx)

        with np.load(r.mask_path, allow_pickle=False) as z:
            M = z["M"].astype(np.uint8)
        if self.training and self.d4_augment:
            M = random_d4(M, rng)

        with np.load(r.state_path, allow_pickle=False) as z:
            E = z["eigenvalues"].astype(np.float64)
            q = z["q_state"].astype(np.float64)
            branch_scores = z["branch_scores"].astype(np.float64)
            hard_branch = z["branch_index"].astype(np.int64)
            cluster_id = z["cluster_id"].astype(np.int64)
            cluster_size = z["cluster_size"].astype(np.int64)
            is_degenerate = z["is_degenerate"].astype(bool)
            canonicalized = z["canonicalized"].astype(bool)

        nstate = len(E)
        if self.training:
            inds, rank_mask = self._sample_training_indices(rng, cluster_id, is_degenerate, hard_branch)
        else:
            inds = np.linspace(0, nstate - 1, self.K, dtype=np.int64)
            rank_mask = np.ones(len(inds), dtype=np.float32)

        target_scores = np.clip(branch_scores[inds], 0.0, None)
        target_scores /= np.clip(target_scores.sum(axis=1, keepdims=True), 1e-12, None)
        gf = self.global_scaler.transform_np(raw_global_features(r))
        E_scaled = self.energy_scaler.transform_np(E[inds])

        # Equalize cluster contribution inside the explicit subspace loss.
        cw = np.ones(len(inds), dtype=np.float32)
        deg = is_degenerate[inds]
        cw[deg] = 1.0 / np.maximum(cluster_size[inds][deg].astype(np.float32), 1.0)

        return {
            "family_id": r.family_id,
            "grid": torch.from_numpy(make_geometry_grid(M, r.b)),
            "global_features": torch.from_numpy(gf.astype(np.float32)),
            "topology_prior": torch.from_numpy(self.topology_prior[r.family_id].astype(np.float32)),
            "q": torch.tensor(q[inds], dtype=torch.float32).unsqueeze(-1),
            "energy": torch.tensor(E_scaled, dtype=torch.float32).unsqueeze(-1),
            "branch_scores": torch.tensor(target_scores, dtype=torch.float32),
            "hard_branch": torch.tensor(hard_branch[inds], dtype=torch.long),
            "rank_mask": torch.tensor(rank_mask, dtype=torch.float32),
            "degenerate_mask": torch.tensor(deg.astype(np.float32), dtype=torch.float32),
            "canonicalized_mask": torch.tensor(canonicalized[inds].astype(np.float32), dtype=torch.float32),
            "cluster_weight": torch.tensor(cw, dtype=torch.float32),
        }


# =============================================================================
# 7. MULTI-SCALE CNN + TRANSFORMER MODEL
# =============================================================================

class MultiScaleCNN2D(nn.Module):
    """2D adaptation of the old 3/5/7 multi-scale CNN idea."""

    def __init__(self, in_channels=4, branch_channels=64, d_model=128, dropout=0.10):
        super().__init__()

        self.conv3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, branch_channels, kernel_size=7, padding=3)

        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        c7 = F.relu(self.conv7(x))
        x = torch.cat([c3, c5, c7], dim=1)
        return self.fuse(x)


class GeometryEncoder(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.10,
        cnn_channels=64,
    ):
        super().__init__()

        self.cnn = MultiScaleCNN2D(
            in_channels=4,
            branch_channels=cnn_channels,
            d_model=d_model,
            dropout=dropout,
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + MAX_B * MAX_B, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, grid, global_features):
        # CNN feature map: [B,D,7,7] -> 49 spatial tokens
        feat = self.cnn(grid)
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B,49,D]

        # global b/N/Df token
        g = self.global_mlp(global_features).unsqueeze(1)  # [B,1,D]
        tokens = torch.cat([g, tokens], dim=1)             # [B,50,D]
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]

        # invalid padded positions in the 7x7 grid are ignored by attention
        valid = grid[:, 1].flatten(1) > 0.5
        global_valid = torch.ones((B, 1), dtype=torch.bool, device=grid.device)
        valid = torch.cat([global_valid, valid], dim=1)
        padding_mask = ~valid

        memory = self.encoder(tokens, src_key_padding_mask=padding_mask)
        memory = self.norm(memory)
        return memory, padding_mask


class FourierQueryEmbedding(nn.Module):
    def __init__(self, d_model=128, frequencies=(1, 2, 4, 8, 16, 32, 64, 128)):
        super().__init__()
        self.frequencies = tuple(frequencies)
        in_dim = 1 + 2 * len(self.frequencies)
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, q):
        feats = [q]
        for f in self.frequencies:
            a = 2.0 * math.pi * float(f) * q
            feats.extend([torch.sin(a), torch.cos(a)])
        return self.net(torch.cat(feats, dim=-1))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens, memory, memory_padding_mask):
        x, _ = self.attn(
            query=q_tokens,
            key=memory,
            value=memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        q_tokens = self.norm1(q_tokens + x)
        q_tokens = self.norm2(q_tokens + self.ff(q_tokens))
        return q_tokens


class FractalSSHPredictor(nn.Module):
    """
    v7 dual-head projector + rank-consistent hard-label model (architecture kept checkpoint-compatible with v3).

    The energy stream is intentionally unchanged from v1.  The exact
    strong-coupling reference-rank prior is injected only into the branch
    stream:

        x_branch = x_query + f_topo(r_ref)

    The final linear layer of f_topo is zero-initialized, so a v1 checkpoint
    the v7 full-clean run is trained from scratch
    before fine-tuning.
    """

    def __init__(
        self,
        d_model=128,
        nhead=8,
        encoder_layers=3,
        query_layers=2,
        dim_feedforward=512,
        dropout=0.10,
        cnn_channels=64,
    ):
        super().__init__()

        self.geometry_encoder = GeometryEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            cnn_channels=cnn_channels,
        )
        self.query_embed = FourierQueryEmbedding(d_model=d_model)
        self.query_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(query_layers)
        ])

        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Head A: continuous projector composition. This is the physical soft
        # target and is the only head used by projector/rank/subspace losses.
        self.branch_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, NUM_BRANCHES),
        )

        # Head B: discrete v7 canonical branch identity. It is intentionally
        # separate from the projector head so hard CE cannot directly distort
        # continuous projector probabilities.
        self.hard_branch_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, NUM_BRANCHES),
        )

        # Exact geometry-derived reference-rank prior -> shared branch feature bias.
        self.topology_prior_mlp = nn.Sequential(
            nn.Linear(NUM_BRANCHES, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        last = self.topology_prior_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def encode_geometry(self, grid, global_features):
        return self.geometry_encoder(grid, global_features)

    def decode_queries(
        self,
        q,
        memory,
        memory_padding_mask,
        topology_prior,
    ):
        x = self.query_embed(q)
        for block in self.query_blocks:
            x = block(x, memory, memory_padding_mask)

        pred_E = self.energy_head(x)

        topo_bias = self.topology_prior_mlp(topology_prior).unsqueeze(1)
        branch_features = x + topo_bias
        projector_logits = self.branch_head(branch_features)
        hard_logits = self.hard_branch_head(branch_features)
        return pred_E, projector_logits, hard_logits

    def forward(self, grid, global_features, topology_prior, q):
        memory, mask = self.encode_geometry(grid, global_features)
        return self.decode_queries(q, memory, mask, topology_prior)


# =============================================================================
# 8. MULTI-TASK LOSS
# =============================================================================

class MultiTaskLoss(nn.Module):
    """v7 dual-head objective.

    Projector head:
        soft projector CE + degenerate-projector L1 + projector-rank loss.
    Hard head:
        class-balanced CE on the rank-consistent canonical labels + a small
        hard-probability rank prior.
    Shared backbone:
        energy loss is estimated only from uniformly sampled queries, avoiding
        bias from the class-focused query subset.
    """

    def __init__(
        self,
        class_weights,
        smooth_l1_beta=0.8,
        branch_weight=0.20,
        hard_ce_weight=0.20,
        subspace_weight=0.10,
        rank_weight=0.08,
        hard_rank_weight=0.04,
        monotonic_weight=0.05,
    ):
        super().__init__()
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.branch_weight = float(branch_weight)
        self.hard_ce_weight = float(hard_ce_weight)
        self.subspace_weight = float(subspace_weight)
        self.rank_weight = float(rank_weight)
        self.hard_rank_weight = float(hard_rank_weight)
        self.monotonic_weight = float(monotonic_weight)

    def forward(
        self, pred_E, true_E, projector_logits, hard_logits, true_scores, hard_branch,
        topology_prior, rank_mask, degenerate_mask, canonicalized_mask, cluster_weight,
    ):
        eps = 1e-7
        rm_state = rank_mask.to(pred_E.dtype)
        eper = F.smooth_l1_loss(pred_E[..., 0], true_E[..., 0], beta=self.smooth_l1_beta, reduction="none")
        energy = torch.sum(eper * rm_state) / torch.sum(rm_state).clamp_min(1.0)

        proj_logp = F.log_softmax(projector_logits, dim=-1)
        proj_probs = torch.softmax(projector_logits, dim=-1)
        projector = -(true_scores * proj_logp).sum(dim=-1).mean()

        hard_ce = F.cross_entropy(
            hard_logits.reshape(-1, NUM_BRANCHES),
            hard_branch.reshape(-1),
            weight=self.class_weights,
            reduction="mean",
        )

        per_state_l1 = torch.mean(torch.abs(proj_probs - true_scores), dim=-1)
        sw = degenerate_mask.to(per_state_l1.dtype) * cluster_weight.to(per_state_l1.dtype)
        if torch.sum(sw) > 0:
            subspace = torch.sum(per_state_l1 * sw) / torch.sum(sw).clamp_min(eps)
        else:
            subspace = pred_E.new_tensor(0.0)

        rm = rank_mask.to(proj_probs.dtype).unsqueeze(-1)
        denom = rm.sum(dim=1).clamp_min(1.0)
        proj_mean = (proj_probs * rm).sum(dim=1) / denom
        rank = torch.mean(torch.sum((proj_mean - topology_prior) ** 2, dim=-1) / NUM_BRANCHES)

        hard_probs = torch.softmax(hard_logits, dim=-1)
        hard_mean = (hard_probs * rm).sum(dim=1) / denom
        hard_rank = torch.mean(torch.sum((hard_mean - topology_prior) ** 2, dim=-1) / NUM_BRANCHES)

        if pred_E.size(1) > 1:
            monotonic = F.relu(pred_E[:, :-1, 0] - pred_E[:, 1:, 0]).mean()
        else:
            monotonic = pred_E.new_tensor(0.0)

        total = (
            energy
            + self.branch_weight * projector
            + self.hard_ce_weight * hard_ce
            + self.subspace_weight * subspace
            + self.rank_weight * rank
            + self.hard_rank_weight * hard_rank
            + self.monotonic_weight * monotonic
        )
        return total, energy, projector, hard_ce, subspace, rank, hard_rank, monotonic


# =============================================================================
# 9. SAMPLER / LOADERS
# =============================================================================

def make_family_sampler(records: List[FamilyRecord], strength=0.50):
    """Mild balancing because b=7 contains most geometries."""
    count_b: Dict[int, int] = {}
    for r in records:
        count_b[r.b] = count_b.get(r.b, 0) + 1

    weights = [1.0 / (count_b[r.b] ** strength) for r in records]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
    )


def make_loaders(args, train_records, val_records, energy_scaler, global_scaler):
    train_ds = FamilyQueryDataset(
        train_records,
        energy_scaler,
        global_scaler,
        queries_per_family=args.queries_per_family,
        training=True,
        d4_augment=not args.no_d4_augment,
        seed=args.seed,
        uniform_query_fraction=args.uniform_query_fraction,
        zero_focus_power=args.zero_focus_power,
    )
    val_ds = FamilyQueryDataset(
        val_records,
        energy_scaler,
        global_scaler,
        queries_per_family=args.queries_per_family,
        training=False,
        d4_augment=False,
        seed=args.seed + 1,
        uniform_query_fraction=1.0,
        zero_focus_power=args.zero_focus_power,
    )

    sampler = make_family_sampler(train_records, strength=args.balance_strength)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    return train_ds, val_ds, train_loader, val_loader


# =============================================================================
# 10. TRAINING UTILITIES
# =============================================================================

def build_model(args):
    return FractalSSHPredictor(
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        query_layers=args.query_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        cnn_channels=args.cnn_channels,
    )


def classification_metrics_from_confusion(confusion: np.ndarray):
    precision, recall, f1 = [], [], []
    for k in range(NUM_BRANCHES):
        tp = float(confusion[k, k])
        support = float(confusion[k, :].sum())
        predicted = float(confusion[:, k].sum())
        p = tp / predicted if predicted > 0 else float("nan")
        r = tp / support if support > 0 else float("nan")
        if np.isfinite(p) and np.isfinite(r) and p + r > 0:
            ff = 2.0 * p * r / (p + r)
        elif support > 0:
            ff = 0.0
        else:
            ff = float("nan")
        precision.append(p); recall.append(r); f1.append(ff)
    valid = [x for x in f1 if np.isfinite(x)]
    return {
        "precision": np.asarray(precision),
        "recall": np.asarray(recall),
        "f1": np.asarray(f1),
        "macro_f1": float(np.mean(valid)) if valid else float("nan"),
    }


def move_batch(batch, device):
    keys = [
        "grid", "global_features", "topology_prior", "q", "energy",
        "branch_scores", "hard_branch", "rank_mask", "degenerate_mask",
        "canonicalized_mask", "cluster_weight",
    ]
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)

    if training and not any(p.requires_grad for p in model.geometry_encoder.parameters()):
        model.geometry_encoder.eval()
        model.query_embed.eval()
        model.query_blocks.eval()
        model.energy_head.eval()
        model.branch_head.train()
        model.hard_branch_head.train()
        model.topology_prior_mlp.train()

    sums = {"loss":0.0, "energy":0.0, "projector":0.0, "hard_ce":0.0, "subspace":0.0, "rank":0.0, "hard_rank":0.0, "monotonic":0.0}
    confusion = np.zeros((NUM_BRANCHES, NUM_BRANCHES), dtype=np.int64)
    proj_abs_sum = 0.0
    proj_count = 0
    deg_abs_sum = 0.0
    deg_count = 0
    nbatch = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            b = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            pred_E, projector_logits, hard_logits = model(
                b["grid"], b["global_features"], b["topology_prior"], b["q"]
            )
            losses = criterion(
                pred_E, b["energy"], projector_logits, hard_logits, b["branch_scores"], b["hard_branch"],
                b["topology_prior"], b["rank_mask"], b["degenerate_mask"], b["canonicalized_mask"], b["cluster_weight"]
            )
            loss, loss_E, loss_P, loss_CE, loss_S, loss_R, loss_HR, loss_M = losses

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                optimizer.step()

            probs = torch.softmax(projector_logits, dim=-1)
            pred_branch = hard_logits.argmax(dim=-1)
            yt = b["hard_branch"].detach().cpu().numpy().reshape(-1)
            yp = pred_branch.detach().cpu().numpy().reshape(-1)
            np.add.at(confusion, (yt, yp), 1)

            abs_err = torch.abs(probs - b["branch_scores"])
            proj_abs_sum += float(abs_err.sum().item())
            proj_count += int(abs_err.numel())
            dm = b["degenerate_mask"].bool()
            if dm.any():
                de = abs_err[dm]
                deg_abs_sum += float(de.sum().item())
                deg_count += int(de.numel())

            sums["loss"] += float(loss.item())
            sums["energy"] += float(loss_E.item())
            sums["projector"] += float(loss_P.item())
            sums["hard_ce"] += float(loss_CE.item())
            sums["subspace"] += float(loss_S.item())
            sums["rank"] += float(loss_R.item())
            sums["hard_rank"] += float(loss_HR.item())
            sums["monotonic"] += float(loss_M.item())
            nbatch += 1

    if nbatch == 0:
        return {k: float("nan") for k in [
            "loss","energy_loss","projector_loss","hard_ce_loss","subspace_loss","rank_loss","hard_rank_loss","monotonic_loss",
            "branch_acc_aux","macro_f1_aux","projector_mae","degenerate_projector_mae"
        ]}

    cls = classification_metrics_from_confusion(confusion)
    return {
        "loss": sums["loss"] / nbatch,
        "energy_loss": sums["energy"] / nbatch,
        "projector_loss": sums["projector"] / nbatch,
        "hard_ce_loss": sums["hard_ce"] / nbatch,
        "subspace_loss": sums["subspace"] / nbatch,
        "rank_loss": sums["rank"] / nbatch,
        "hard_rank_loss": sums["hard_rank"] / nbatch,
        "monotonic_loss": sums["monotonic"] / nbatch,
        "branch_acc_aux": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "macro_f1_aux": cls["macro_f1"],
        "projector_mae": proj_abs_sum / max(proj_count, 1),
        "degenerate_projector_mae": deg_abs_sum / max(deg_count, 1) if deg_count else float("nan"),
    }


def checkpoint_model_args(args):
    return {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "encoder_layers": args.encoder_layers,
        "query_layers": args.query_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "cnn_channels": args.cnn_channels,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val,
    best_oc_f1,
    args,
    best_zero_f1=-float("inf"),
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": int(epoch),
            "best_val": float(best_val),
            "best_oc_f1": float(best_oc_f1),
            "best_zero_f1": float(best_zero_f1),
            "model_args": checkpoint_model_args(args),
            "physics_prior": {
                "name": "geometry-derived strong-coupling reference-rank prior",
                "branch_order": BRANCH_LABELS,
                "zero_sector": [BRANCH_LABELS[i] for i in ZERO_SECTOR_INDICES],
                "oc_index": OC_INDEX,
            },
        },
        path,
    )


def model_from_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    model = FractalSSHPredictor(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_pretrained_weights(model, checkpoint_path: Path, device):
    """
    Optional compatibility loader. The formal v7 one-click run does not use
    older checkpoints; it starts from random initialization.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"]

    incompat = model.load_state_dict(state, strict=False)
    unexpected = list(incompat.unexpected_keys)
    missing = list(incompat.missing_keys)

    allowed_prefixes = (
        "hard_branch_head.",
        "topology_prior_mlp.",
    )
    bad_missing = [
        k for k in missing
        if not k.startswith(allowed_prefixes)
    ]

    if unexpected or bad_missing:
        raise RuntimeError(
            "pretrained -> v7 loading mismatch.\n"
            f"Unexpected keys: {unexpected}\n"
            f"Unexpected missing keys: {bad_missing}"
        )

    print(f"Initialized v7 from checkpoint: {checkpoint_path}")
    print(
        f"source epoch={ckpt.get('epoch')}, "
        f"best_val={ckpt.get('best_val')}, "
        f"best_zero_f1={ckpt.get('best_zero_f1')}"
    )
    print("New v7-compatible keys:", [k for k in missing if k.startswith("hard_branch_head.")])
    return ckpt


def configure_trainable_parameters(model, train_backbone: bool):
    """
    Default fine-tuning strategy freezes the entire energy/backbone stream and trains
    only:
        1) branch_head
        2) topology_prior_mlp
        3) hard_branch_head

    The formal v7 full-clean pipeline uses --train-backbone and random initialization.

    Use --train-backbone only for a later second-stage experiment.
    """
    if train_backbone:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

        for p in model.branch_head.parameters():
            p.requires_grad = True

        for p in model.topology_prior_mlp.parameters():
            p.requires_grad = True

        for p in model.hard_branch_head.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.2f}%)"
    )

    return trainable, total


def build_optimizer(model, args):
    # Physics-specific trainable modules receive the larger LR because the
    # hard_branch_head and topology_prior_mlp are the
    # geometry-derived physics adapter.  branch_head remains at base LR.
    physics_params = [
        p
        for module in (
            model.topology_prior_mlp,
            model.hard_branch_head,
        )
        for p in module.parameters()
        if p.requires_grad
    ]
    physics_ids = {id(p) for p in physics_params}

    base_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in physics_ids
    ]

    groups = []
    if base_params:
        groups.append({
            "params": base_params,
            "lr": args.lr,
            "name": "branch_base",
        })

    if physics_params:
        groups.append({
            "params": physics_params,
            "lr": args.lr * args.prior_lr_multiplier,
            "name": "physics_hard_head",
        })

    if not groups:
        raise RuntimeError("No trainable parameters were selected.")

    return torch.optim.Adam(
        groups,
        weight_decay=args.weight_decay,
    )


# =============================================================================
# 11. FULL-SPECTRUM INFERENCE
# =============================================================================

@torch.no_grad()
def predict_full_family(
    model,
    record: FamilyRecord,
    energy_scaler: EnergyScaler,
    global_scaler: GlobalFeatureScaler,
    device,
    chunk_size=1024,
):
    model.eval()
    with np.load(record.mask_path, allow_pickle=False) as z:
        M = z["M"].astype(np.uint8)

    grid = torch.from_numpy(make_geometry_grid(M, record.b)).unsqueeze(0).to(device)
    gf_np = global_scaler.transform_np(raw_global_features(record)).astype(np.float32)
    gf = torch.from_numpy(gf_np).unsqueeze(0).to(device)
    counts, ref_dims, prior_np = strong_coupling_component_counts_from_generator(M)
    topology_prior = torch.from_numpy(prior_np.astype(np.float32)).unsqueeze(0).to(device)

    with np.load(record.state_path, allow_pickle=False) as z:
        true_E = z["eigenvalues"].astype(np.float64)
        q = z["q_state"].astype(np.float64)
        true_scores = z["branch_scores"].astype(np.float64)
        true_branch = z["branch_index"].astype(np.int64)
        true_branch_argmax = z["branch_index_canonical_argmax"].astype(np.int64) if "branch_index_canonical_argmax" in z.files else true_branch.copy()
        zero_assignment_changed = z["zero_assignment_changed"].astype(bool) if "zero_assignment_changed" in z.files else (true_branch != true_branch_argmax)
        cluster_id = z["cluster_id"].astype(np.int64)
        cluster_size = z["cluster_size"].astype(np.int64)
        cluster_energy = z["cluster_energy"].astype(np.float64)
        is_degenerate = z["is_degenerate"].astype(bool)
        is_zero_cluster = z["is_zero_cluster"].astype(bool) if "is_zero_cluster" in z.files else np.zeros(len(q), bool)
        true_scores_raw = z["branch_scores_raw"].astype(np.float64) if "branch_scores_raw" in z.files else true_scores.copy()
        if "ref_dims" in z.files and not np.array_equal(z["ref_dims"].astype(np.int64), ref_dims.astype(np.int64)):
            raise RuntimeError(f"Geometry-derived ref_dims disagree with DB for {record.family_id}")

    memory, memory_mask = model.encode_geometry(grid, gf)
    E_chunks, score_chunks, hard_chunks = [], [], []
    for start_i in range(0, len(q), chunk_size):
        q_t = torch.tensor(q[start_i:start_i + chunk_size], dtype=torch.float32, device=device).view(1, -1, 1)
        pred_E_scaled, projector_logits, hard_logits = model.decode_queries(q_t, memory, memory_mask, topology_prior)
        E_chunks.append(pred_E_scaled[0, :, 0].cpu().numpy())
        score_chunks.append(torch.softmax(projector_logits, dim=-1)[0].cpu().numpy())
        hard_chunks.append(torch.softmax(hard_logits, dim=-1)[0].cpu().numpy())

    pred_E = energy_scaler.inverse_np(np.concatenate(E_chunks))
    pred_scores = np.concatenate(score_chunks, axis=0)
    pred_hard_probs = np.concatenate(hard_chunks, axis=0)
    pred_branch = pred_hard_probs.argmax(axis=1).astype(np.int64)

    return {
        "q": q,
        "true_E": true_E,
        "pred_E": pred_E,
        "true_scores": true_scores,
        "true_scores_raw_diagnostic": true_scores_raw,
        "pred_scores": pred_scores,
        "pred_hard_probs": pred_hard_probs,
        "true_branch": true_branch,
        "true_branch_argmax_diagnostic": true_branch_argmax,
        "zero_assignment_changed": zero_assignment_changed,
        "pred_branch": pred_branch,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "cluster_energy": cluster_energy,
        "is_degenerate": is_degenerate,
        "is_zero_cluster": is_zero_cluster,
        "topology_prior": prior_np.astype(np.float64),
        "ref_dims_geometry": ref_dims.astype(np.int64),
        "component_counts": np.asarray([
            counts["n_monomer"], counts["n_dimer"], counts["n_trimer"], counts["n_tetramer"]
        ], dtype=np.int64),
    }


def compute_family_metrics(pred):
    y, yp = pred["true_E"], pred["pred_E"]
    mae = mean_absolute_error(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    r2 = r2_score(y, yp) if np.std(y) > 1e-12 else float("nan")

    true_scores = np.asarray(pred["true_scores"], dtype=float)
    pred_scores = np.asarray(pred["pred_scores"], dtype=float)
    score_mae = float(np.mean(np.abs(true_scores - pred_scores)))
    deg = np.asarray(pred["is_degenerate"], dtype=bool)
    nondeg = ~deg
    deg_mae = float(np.mean(np.abs(true_scores[deg] - pred_scores[deg]))) if np.any(deg) else float("nan")
    nondeg_mae = float(np.mean(np.abs(true_scores[nondeg] - pred_scores[nondeg]))) if np.any(nondeg) else float("nan")

    true_branch = np.asarray(pred["true_branch"], dtype=np.int64)
    pred_branch = np.asarray(pred["pred_branch"], dtype=np.int64)
    aux_acc = float(np.mean(true_branch == pred_branch))
    aux_deg_acc = float(np.mean(true_branch[deg] == pred_branch[deg])) if np.any(deg) else float("nan")

    # Equal-weight cluster composition metric, the main subspace diagnostic.
    cids = np.unique(pred["cluster_id"])
    cluster_err = []
    deg_cluster_err = []
    for cid in cids:
        m = pred["cluster_id"] == cid
        pt = np.mean(true_scores[m], axis=0)
        pp = np.mean(pred_scores[m], axis=0)
        e = float(np.mean(np.abs(pt - pp)))
        cluster_err.append(e)
        if np.sum(m) > 1:
            deg_cluster_err.append(e)

    prior = np.asarray(pred["topology_prior"], dtype=np.float64)
    projector_rank_mae = float(np.mean(np.abs(np.mean(pred_scores, axis=0) - prior)))
    hard_probs = np.asarray(pred.get("pred_hard_probs"), dtype=np.float64)
    hard_rank_mae = float(np.mean(np.abs(np.mean(hard_probs, axis=0) - prior)))
    hard_count_frac = np.bincount(pred_branch, minlength=NUM_BRANCHES).astype(np.float64) / max(len(pred_branch), 1)
    hard_count_rank_mae = float(np.mean(np.abs(hard_count_frac - prior)))
    monotonic = float(np.mean(np.diff(yp) < 0.0)) if len(yp) > 1 else 0.0

    return {
        "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
        "projector_score_MAE": score_mae,
        "nondegenerate_projector_MAE": nondeg_mae,
        "degenerate_projector_MAE": deg_mae,
        "cluster_projector_MAE": float(np.mean(cluster_err)) if cluster_err else float("nan"),
        "degenerate_cluster_projector_MAE": float(np.mean(deg_cluster_err)) if deg_cluster_err else float("nan"),
        "hard_argmax_accuracy_aux": aux_acc,
        "degenerate_hard_argmax_accuracy_aux": aux_deg_acc,
        "projector_rank_MAE": projector_rank_mae,
        "hard_probability_rank_MAE": hard_rank_mae,
        "hard_count_rank_MAE": hard_count_rank_mae,
        "zero_assignment_changed_fraction": float(np.mean(np.asarray(pred.get("zero_assignment_changed", np.zeros(len(true_branch), bool)), dtype=bool))),
        "monotonic_violation_rate": monotonic,
        "n_degenerate_states": int(np.sum(deg)),
        "n_degenerate_clusters": int(sum(np.sum(pred["cluster_id"] == cid) > 1 for cid in cids)),
    }


def save_prediction_plot(path: Path, record: FamilyRecord, pred, metrics):
    idx = np.arange(1, len(pred["true_E"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(idx, pred["true_E"], linewidth=1.2, label="True")
    ax.plot(idx, pred["pred_E"], linewidth=1.2, linestyle="--", label="Predicted")
    ax.set_xlabel("State index")
    ax.set_ylabel("Energy")
    ax.set_title(
        f"{record.family_id} | RMSE={metrics['RMSE']:.4f}, "
        f"R2={metrics['R2']:.4f}, Proj MAE={metrics['projector_score_MAE']:.4f}"
    )
    ax.legend()
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 12. MODES
# =============================================================================

def mode_check(args):
    records = load_available_families(args.db_root, args.only_b)
    print_family_stats("available", records)

    energy_scaler = fit_energy_scaler(records)
    global_scaler = fit_global_scaler(records)

    ds = FamilyQueryDataset(
        records,
        energy_scaler,
        global_scaler,
        queries_per_family=min(args.queries_per_family, 32),
        training=False,
        d4_augment=False,
        seed=args.seed,
        verify_reference_dims=True,
    )

    loader = DataLoader(
        ds,
        batch_size=min(2, len(ds)),
        shuffle=False,
    )
    batch = next(iter(loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(args).to(device)
    b = move_batch(batch, device)

    with torch.no_grad():
        pred_E, pred_P, pred_H = model(
            b["grid"],
            b["global_features"],
            b["topology_prior"],
            b["q"],
        )

    print("Device                :", device)
    print("grid                  :", tuple(b["grid"].shape))
    print("global features       :", tuple(b["global_features"].shape))
    print("topology prior        :", tuple(b["topology_prior"].shape))
    print("q                     :", tuple(b["q"].shape))
    print("predicted energy      :", tuple(pred_E.shape))
    print("predicted projector logit:", tuple(pred_P.shape))
    print("predicted hard logit     :", tuple(pred_H.shape))

    prior_sum = b["topology_prior"].sum(dim=-1)
    print("topology prior sum    :", prior_sum.detach().cpu().numpy())

    print("Geometry-derived reference-rank + v7 dual-target check: PASS")
    print("CHECK PASSED")


def seed_new_run_from_previous(args):
    """
    Copy only split/scaler metadata from the original formal run.

    This preserves a fair pretrained-to-v6 comparison:
      - identical 652/81/83 family split
      - identical train-only scalers

    Old checkpoints/results are never overwritten.
    """
    if args.init_from_run is None:
        return

    src_run = Path(args.init_from_run)
    if not src_run.exists():
        raise FileNotFoundError(
            f"--init-from-run does not exist: {src_run}"
        )

    if args.refresh_split:
        print(
            "WARNING: --refresh-split is set, so the old family split will "
            "not be copied."
        )
    else:
        for name in ("family_split.json", "split_manifest.csv"):
            src_path = src_run / name
            dst_path = args.run_dir / name

            if src_path.exists() and not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"Copied {name} from old formal run.")

    scaler_src = src_run / "scalers.json"
    scaler_dst = args.run_dir / "scalers.json"

    if scaler_src.exists() and not scaler_dst.exists():
        shutil.copy2(scaler_src, scaler_dst)
        print("Copied train-only scalers.json from old formal run.")


def prepare_or_load_split(args, records):
    split_path = args.run_dir / "family_split.json"

    if args.refresh_split and split_path.exists():
        split_path.unlink()
        csv_path = args.run_dir / "split_manifest.csv"
        if csv_path.exists():
            csv_path.unlink()

    if split_path.exists():
        return load_split(args.run_dir, records)

    train, val, test = stratified_family_split(
        records,
        seed=args.seed,
    )
    save_split(args.run_dir, train, val, test)
    return train, val, test


def mode_train(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    seed_new_run_from_previous(args)
    records = load_available_families(args.db_root, args.only_b)
    train_records, val_records, test_records = prepare_or_load_split(args, records)
    print_family_stats("train", train_records); print_family_stats("val", val_records); print_family_stats("test", test_records)
    if not val_records:
        raise RuntimeError("Validation split is empty; use the full formal database or refresh the split.")

    if (args.run_dir / "scalers.json").exists():
        energy_scaler, global_scaler = load_scalers(args.run_dir)
    else:
        energy_scaler = fit_energy_scaler(train_records)
        global_scaler = fit_global_scaler(train_records)
        save_scalers(args.run_dir, energy_scaler, global_scaler)

    train_ds, val_ds, train_loader, val_loader = make_loaders(
        args, train_records, val_records, energy_scaler, global_scaler
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if args.init_from_run is None and not args.train_backbone:
        print("No pretrained run supplied: enabling backbone training for a valid from-scratch v7 run.")
        args.train_backbone = True
    model = build_model(args).to(device)
    class_weights_np, class_counts = fit_branch_class_weights(
        train_records, power=args.class_balance_power, max_weight=args.max_class_weight
    )
    print("Training canonical branch counts:", dict(zip(BRANCH_LABELS, class_counts.astype(int).tolist())))
    print("Training hard-CE class weights:", dict(zip(BRANCH_LABELS, np.round(class_weights_np, 4).tolist())))
    criterion = MultiTaskLoss(
        class_weights=class_weights_np,
        smooth_l1_beta=0.8,
        branch_weight=args.branch_weight,
        hard_ce_weight=args.hard_ce_weight,
        subspace_weight=args.subspace_weight,
        rank_weight=args.rank_weight,
        hard_rank_weight=args.hard_rank_weight,
        monotonic_weight=args.monotonic_weight,
    ).to(device)
    # Safety rule: a frozen random backbone is never a valid training setup.
    # If no compatible pretrained run is supplied, automatically train the full backbone.
    effective_train_backbone = bool(args.train_backbone or args.init_from_run is None)
    if args.init_from_run is None and not args.train_backbone:
        print(
            "No --init-from-run supplied: automatically enabling full-backbone training "
            "to avoid freezing a random encoder."
        )
    configure_trainable_parameters(model, train_backbone=effective_train_backbone)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
    )

    start_epoch, best_val, bad_epochs = 1, float("inf"), 0
    history_path = args.run_dir / "training_history.csv"
    history_rows = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    last_path = args.run_dir / "last_model.pt"

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state") is not None: optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None: scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"Resuming v7 from epoch {start_epoch}; best val={best_val:.6g}")
    elif args.init_from_run is not None:
        old_best = Path(args.init_from_run) / args.init_checkpoint_name
        if not old_best.exists(): raise FileNotFoundError(old_best)
        load_pretrained_weights(model, old_best, device)
    else:
        print("WARNING: no --init-from-run supplied; v7 trains from random initialization.")

    config = {
        "model": "MultiScaleCNN-Transformer canonical-basis v7",
        "branch_order": BRANCH_LABELS,
        "target": "dual target: continuous canonical projector scores + v7 rank-consistent hard labels",
        "hard_labels": "OC/IC-II/BULK-0 rank-constrained maximum-overlap truth; six nonzero branches stable argmax; freely predicted at inference; no capacity decoder",
        "class_balance_power": args.class_balance_power,
        "max_class_weight": args.max_class_weight,
        "freeze_backbone": not effective_train_backbone,
        "loss_weights": {
            "projector": args.branch_weight,
            "rank_consistent_hard_ce": args.hard_ce_weight,
            "degenerate_subspace": args.subspace_weight,
            "rank": args.rank_weight,
            "hard_rank": args.hard_rank_weight,
            "monotonic": args.monotonic_weight,
        },
        "uniform_query_fraction": args.uniform_query_fraction,
        "lr": args.lr, "prior_lr_multiplier": args.prior_lr_multiplier,
        "queries_per_family": args.queries_per_family, "seed": args.seed,
    }
    with (args.run_dir / "canonical_v7_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        tr = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, grad_clip=args.grad_clip)
        va = run_epoch(model, val_loader, criterion, device, optimizer=None, grad_clip=args.grad_clip)
        scheduler.step(va["loss"])
        lr_values = [g["lr"] for g in optimizer.param_groups]
        row = {"epoch": epoch, "lr_base": lr_values[0], **{f"train_{k}":v for k,v in tr.items()}, **{f"val_{k}":v for k,v in va.items()}}
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d} | train={tr['loss']:.6f} | val={va['loss']:.6f} | "
            f"val_E={va['energy_loss']:.6f} | val_hardCE={va['hard_ce_loss']:.5f} | val_projMAE={va['projector_mae']:.5f} | "
            f"val_degMAE={va['degenerate_projector_mae']:.5f} | aux_acc={va['branch_acc_aux']:.4f} | lr={lr_values}"
        )
        improve = va["loss"] < best_val - args.min_delta
        if improve:
            best_val = va["loss"]; bad_epochs = 0
            save_checkpoint(args.run_dir / "best_model.pt", model, optimizer, scheduler, epoch, best_val, -1.0, args, best_zero_f1=-1.0)
        else:
            bad_epochs += 1
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val, -1.0, args, best_zero_f1=-1.0)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        if bad_epochs >= args.early_stop_patience:
            print(f"Early stopping after {bad_epochs} epochs without validation-loss improvement.")
            break

    print("Training finished. Best validation loss:", best_val)
    print("Best checkpoint:", args.run_dir / "best_model.pt")


def make_branch_metrics_table(confusion):
    cls = classification_metrics_from_confusion(
        confusion
    )

    rows = []
    for k, label in enumerate(BRANCH_LABELS):
        rows.append({
            "branch": label,
            "support": int(
                confusion[k, :].sum()
            ),
            "predicted": int(
                confusion[:, k].sum()
            ),
            "precision": float(
                cls["precision"][k]
            ),
            "recall": float(
                cls["recall"][k]
            ),
            "F1": float(
                cls["f1"][k]
            ),
        })

    return pd.DataFrame(rows), cls


def mode_test(args):
    records = load_available_families(args.db_root, args.only_b)
    _, _, test_records = load_split(args.run_dir, records)
    energy_scaler, global_scaler = load_scalers(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.run_dir / args.checkpoint_name
    if not checkpoint_path.exists(): raise FileNotFoundError(checkpoint_path)
    model, ckpt = model_from_checkpoint(checkpoint_path, device)
    tag = args.test_tag if args.test_tag else checkpoint_path.stem
    outdir = args.run_dir / f"test_predictions_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows, all_true_E, all_pred_E, all_true_scores, all_pred_scores, all_deg, all_true_branch, all_pred_branch = [], [], [], [], [], [], [], []
    for i, r in enumerate(test_records, start=1):
        pred = predict_full_family(model, r, energy_scaler, global_scaler, device, chunk_size=args.predict_chunk)
        m = compute_family_metrics(pred)
        rows.append({"family_id":r.family_id,"b":r.b,"N":r.N,"Df":r.Df,"n_sites":r.n_sites,**m})
        all_true_E.append(pred["true_E"]); all_pred_E.append(pred["pred_E"])
        all_true_scores.append(pred["true_scores"]); all_pred_scores.append(pred["pred_scores"]); all_deg.append(pred["is_degenerate"]); all_true_branch.append(pred["true_branch"]); all_pred_branch.append(pred["pred_branch"])
        np.savez_compressed(outdir / f"{r.family_id}_prediction.npz", **pred)
        if i <= args.save_test_plots:
            save_prediction_plot(outdir / f"{r.family_id}_spectrum.png", r, pred, m)
        print(
            f"[{i}/{len(test_records)}] {r.family_id} | RMSE={m['RMSE']:.5f}, R2={m['R2']:.5f}, "
            f"projMAE={m['projector_score_MAE']:.5f}, degClusterMAE={m['degenerate_cluster_projector_MAE']:.5f}"
        )

    df = pd.DataFrame(rows)
    family_metrics_path = args.run_dir / f"test_family_metrics_{tag}.csv"
    df.to_csv(family_metrics_path, index=False)
    metric_cols = [c for c in [
        "MAE","RMSE","R2","projector_score_MAE","nondegenerate_projector_MAE",
        "degenerate_projector_MAE","cluster_projector_MAE","degenerate_cluster_projector_MAE",
        "hard_argmax_accuracy_aux","degenerate_hard_argmax_accuracy_aux","projector_rank_MAE","hard_probability_rank_MAE","hard_count_rank_MAE","zero_assignment_changed_fraction",
        "monotonic_violation_rate"
    ] if c in df.columns]
    by_b = df.groupby("b")[metric_cols].mean(); by_b["num_test_families"] = df.groupby("b").size()
    by_b_path = args.run_dir / f"test_metrics_by_b_{tag}.csv"; by_b.to_csv(by_b_path)

    true_E, pred_E = np.concatenate(all_true_E), np.concatenate(all_pred_E)
    true_scores, pred_scores = np.concatenate(all_true_scores,0), np.concatenate(all_pred_scores,0)
    deg = np.concatenate(all_deg).astype(bool)
    true_branch = np.concatenate(all_true_branch).astype(np.int64)
    pred_branch = np.concatenate(all_pred_branch).astype(np.int64)
    confusion = np.zeros((NUM_BRANCHES, NUM_BRANCHES), dtype=np.int64)
    np.add.at(confusion, (true_branch, pred_branch), 1)
    branch_df, branch_cls = make_branch_metrics_table(confusion)
    branch_df["projector_MAE"] = [
        float(np.mean(np.abs(true_scores[:, k] - pred_scores[:, k])))
        for k in range(NUM_BRANCHES)
    ]
    branch_df["count_error"] = branch_df["predicted"] - branch_df["support"]
    branch_df["count_rel_error"] = np.where(
        branch_df["support"] > 0, branch_df["count_error"] / branch_df["support"], np.nan
    )
    branch_metrics_path = args.run_dir / f"test_branch_metrics_{tag}.csv"
    branch_df.to_csv(branch_metrics_path, index=False)
    confusion_path = args.run_dir / f"test_confusion_matrix_{tag}.csv"
    pd.DataFrame(confusion, index=BRANCH_LABELS, columns=BRANCH_LABELS).to_csv(confusion_path)
    ss_res = float(np.sum((true_E-pred_E)**2)); ss_tot=float(np.sum((true_E-np.mean(true_E))**2))
    summary = {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(ckpt["epoch"]),
        "num_test_families": int(len(test_records)), "num_test_states": int(len(true_E)),
        "state_weighted": {
            "MAE": float(np.mean(np.abs(true_E-pred_E))),
            "RMSE": float(np.sqrt(np.mean((true_E-pred_E)**2))),
            "pooled_R2": float(1.0-ss_res/ss_tot),
            "projector_score_MAE": float(np.mean(np.abs(true_scores-pred_scores))),
            "degenerate_projector_MAE": float(np.mean(np.abs(true_scores[deg]-pred_scores[deg]))) if np.any(deg) else None,
            "canonical_hard_accuracy": float(np.mean(true_branch == pred_branch)),
            "canonical_macro_F1": float(branch_cls["macro_f1"]),
            "hard_count_rank_MAE": float(np.mean(np.abs(np.bincount(pred_branch, minlength=NUM_BRANCHES) / len(pred_branch) - np.bincount(true_branch, minlength=NUM_BRANCHES) / len(true_branch)))),
            "hard_label_note": "v7 truth uses rank-constrained maximum-overlap OC/IC-II/BULK-0 assignment; model hard head predicts freely with no capacity decoder in primary metrics",
        },
        "family_mean": {k: float(df[k].mean()) for k in metric_cols},
    }
    summary_path = args.run_dir / f"test_summary_{tag}.json"
    with summary_path.open("w", encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    print("\nMean TEST metrics across families:\n", df[metric_cols].mean())
    print("\nState-weighted TEST metrics:\n", json.dumps(summary["state_weighted"], indent=2))
    print("\nPer-branch canonical metrics:\n", branch_df.to_string(index=False))
    print("\nOutputs:\n ", family_metrics_path, "\n ", by_b_path, "\n ", branch_metrics_path, "\n ", confusion_path, "\n ", summary_path, "\n ", outdir)


def mode_predict(args):
    if not args.family_id:
        raise ValueError(
            "--family-id is required in predict mode"
        )

    records = load_available_families(
        args.db_root,
        args.only_b,
    )

    table = {
        r.family_id: r
        for r in records
    }

    if args.family_id not in table:
        raise KeyError(
            f"Family not found among completed spectra: "
            f"{args.family_id}"
        )

    energy_scaler, global_scaler = load_scalers(
        args.run_dir
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        args.run_dir / args.checkpoint_name
    )

    model, ckpt = model_from_checkpoint(
        checkpoint_path,
        device,
    )

    r = table[args.family_id]

    pred = predict_full_family(
        model,
        r,
        energy_scaler,
        global_scaler,
        device,
        chunk_size=args.predict_chunk,
    )

    m = compute_family_metrics(pred)

    tag = (
        args.test_tag
        if args.test_tag
        else checkpoint_path.stem
    )

    outdir = (
        args.run_dir
        / f"single_predictions_{tag}"
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        outdir
        / f"{r.family_id}_prediction.npz",
        **pred,
    )

    save_prediction_plot(
        outdir
        / f"{r.family_id}_spectrum.png",
        r,
        pred,
        m,
    )

    state_df = pd.DataFrame({
        "state_index": np.arange(
            1,
            len(pred["q"]) + 1,
        ),
        "q": pred["q"],
        "true_energy": pred["true_E"],
        "pred_energy": pred["pred_E"],
        "true_branch": [
            BRANCH_LABELS[i]
            for i in pred["true_branch"]
        ],
        "pred_branch": [
            BRANCH_LABELS[i]
            for i in pred["pred_branch"]
        ],
    })

    state_df.to_csv(
        outdir
        / f"{r.family_id}_states.csv",
        index=False,
    )

    print(
        json.dumps(
            m,
            indent=2,
        )
    )
    print("Prediction output:", outdir)


# =============================================================================
# 13. CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--mode",
        required=True,
        choices=(
            "check",
            "train",
            "test",
            "predict",
        ),
    )

    p.add_argument(
        "--db-root",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "./run_ms_cnn_transformer_canonical_v7"
        ),
    )

    p.add_argument(
        "--init-from-run",
        type=Path,
        default=None,
        help=(
            "Previous compatible v3/v2/v1 run directory. Its split/scalers and checkpoint "
            "can be reused only in optional compatibility experiments."
        ),
    )

    p.add_argument(
        "--init-checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside --init-from-run.",
    )

    p.add_argument(
        "--family-id",
        type=str,
        default=None,
    )

    p.add_argument(
        "--only-b",
        type=int,
        nargs="*",
        default=None,
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    p.add_argument(
        "--seed",
        type=int,
        default=20260907,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--queries-per-family",
        type=int,
        default=96,
    )

    # Fine-tuning LR. A validated v3 checkpoint can be used as initialization.
    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    p.add_argument(
        "--prior-lr-multiplier",
        type=float,
        default=3.0,
        help="LR multiplier for the new topology-prior MLP.",
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    # Physics-aware loss weights
    p.add_argument(
        "--branch-weight",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--hard-ce-weight",
        type=float,
        default=0.25,
        help="Class-balanced CE weight for v7 canonical hard branch labels.",
    )

    p.add_argument(
        "--canonical-ce-boost",
        type=float,
        default=0.50,
        help="Extra multiplicative CE emphasis on states canonicalized inside exact degeneracies.",
    )

    p.add_argument(
        "--class-balance-power",
        type=float,
        default=0.25,
        help="Inverse training-label frequency exponent used for canonical hard CE weights.",
    )

    p.add_argument(
        "--max-class-weight",
        type=float,
        default=4.0,
        help="Clamp for training-split-only canonical hard CE class weights.",
    )

    p.add_argument(
        "--subspace-weight",
        type=float,
        default=0.10,
        help="Extra L1 weight on gauge-invariant targets for degenerate clusters.",
    )

    p.add_argument(
        "--oc-weight",
        type=float,
        default=0.0,
        help="Legacy compatibility option; unused by the v7 loss.",
    )

    p.add_argument(
        "--zero-sector-weight",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--zero-margin-weight",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--zero-margin",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--zero-confidence",
        type=float,
        default=0.60,
    )

    p.add_argument(
        "--zero-min-mass",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--rank-weight",
        type=float,
        default=0.08,
    )

    p.add_argument(
        "--hard-rank-weight",
        type=float,
        default=0.04,
        help="Rank-prior consistency weight for the separate hard-label probability head.",
    )

    p.add_argument(
        "--monotonic-weight",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--oc-focus",
        type=float,
        default=1.0,
        help="Legacy compatibility option; unused by the v7 loss.",
    )

    p.add_argument(
        "--rank-oc-weight",
        type=float,
        default=1.0,
        help="Legacy compatibility option; unused by the v7 loss.",
    )

    p.add_argument(
        "--rank-zero-weight",
        type=float,
        default=2.0,
        help="Equal rank-loss weight for OC, IC-II and BULK-0.",
    )

    p.add_argument(
        "--uniform-query-fraction",
        type=float,
        default=0.65,
        help="Fraction of training q queries kept uniform for unbiased rank estimation.",
    )

    p.add_argument(
        "--zero-focus-power",
        type=float,
        default=1.5,
        help="Projector-score exponent for balanced zero-sector focused sampling.",
    )

    p.add_argument(
        "--balance-strength",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--no-d4-augment",
        action="store_true",
    )

    p.add_argument(
        "--train-backbone",
        action="store_true",
        help=(
            "Second-stage option. By default the original v1 backbone/energy "
            "stream is frozen and only branch_head + topology_prior_mlp are trained."
        ),
    )

    # -----------------------------------------------------------------
    # Model: identical base architecture to formal v1
    # -----------------------------------------------------------------
    p.add_argument(
        "--cnn-channels",
        type=int,
        default=64,
    )
    p.add_argument(
        "--d-model",
        type=int,
        default=128,
    )
    p.add_argument(
        "--nhead",
        type=int,
        default=8,
    )
    p.add_argument(
        "--encoder-layers",
        type=int,
        default=3,
    )
    p.add_argument(
        "--query-layers",
        type=int,
        default=2,
    )
    p.add_argument(
        "--dim-feedforward",
        type=int,
        default=512,
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )

    # -----------------------------------------------------------------
    # Schedule / stopping
    # -----------------------------------------------------------------
    p.add_argument(
        "--lr-patience",
        type=int,
        default=5,
    )

    p.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
    )

    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
    )

    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--oc-min-delta",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--zero-min-delta",
        type=float,
        default=1e-4,
    )

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    p.add_argument(
        "--predict-chunk",
        type=int,
        default=1024,
    )

    p.add_argument(
        "--save-test-plots",
        type=int,
        default=20,
    )

    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help=(
            "Checkpoint inside --run-dir for test/predict, e.g. "
            "best_model.pt or last_model.pt"
        ),
    )

    p.add_argument(
        "--test-tag",
        type=str,
        default=None,
        help=(
            "Output suffix for test/predict. "
            "If omitted, the checkpoint stem is used."
        ),
    )

    # -----------------------------------------------------------------
    # Run management
    # -----------------------------------------------------------------
    p.add_argument(
        "--resume",
        action="store_true",
    )

    p.add_argument(
        "--refresh-split",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 100)
    print(
        "Fractal SSH MultiScaleCNN-Transformer "
        "v7 Dual-Head Projector + Canonical-Branch Predictor FULL CLEAN"
    )
    print("=" * 100)
    print("Mode        :", args.mode)
    print("Database    :", args.db_root)
    print("Run dir     :", args.run_dir)
    print("Init run    :", args.init_from_run)
    print("Branches    :", BRANCH_LABELS)
    print(
        "Zero sector :",
        [
            BRANCH_LABELS[i]
            for i in ZERO_SECTOR_INDICES
        ],
    )
    print()

    if args.mode == "check":
        mode_check(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "test":
        mode_test(args)
    elif args.mode == "predict":
        mode_predict(args)


if __name__ == "__main__":
    main()
