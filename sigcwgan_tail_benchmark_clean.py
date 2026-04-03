#!/usr/bin/env python3
"""
Tail-Aware SigCWGAN Benchmark for BTC Hourly Log Returns
=========================================================
Clean, single-file Kaggle/local runner.

Research: Extending SigCWGAN (Ni et al., 2020) with a tail-risk-aware training
loss for improved extreme event generation in cryptocurrency return paths.

Models compared:
  1. SigCWGAN (with tail loss — your contribution)
  2. AR(1)-GJR-GARCH(1,1) with Skewed Student-t
  3. Filtered Historical Simulation (FHS)
  4. TimeGAN (conditional, GRU-based)
  5. Neural SDE (1D Euler-Maruyama)
  6. Conditional Tail-GAN (Fissler-Ziegel scoring, adapted from Cont et al. 2023)

Design:
  - 6 BTC regime datasets × 3 expanding splits × 3 seeds × 5 models
  - Sensitivity analysis: tail_loss_weight ∈ {0, 2, 5, 8, 12}
  - Evaluation: VaR/ES at 95/99, ACF error, CvM, Kupiec, Christoffersen,
    Friedman test, Diebold-Mariano test, Signature MMD

Author: Mikhail Matyushenko
Based on: Ni, Szpruch, Wiese, Liao, Xiao, Sabate-Vidales (2020)
          "Sig-Wasserstein GANs for Conditional Time Series Generation"
"""

# ============================================================================
# 0. ENVIRONMENT SETUP
# ============================================================================

import os
import sys
import glob
import json
import shutil
import subprocess
import random
import time
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

# --- Detect environment ---
IS_KAGGLE = Path("/kaggle").exists()
IS_COLAB = False
try:
    from google.colab import files as _colab_files  # type: ignore
    IS_COLAB = True
except Exception:
    pass
if IS_KAGGLE:
    IS_COLAB = False

WORK_DIR = "/kaggle/working" if IS_KAGGLE else os.getcwd()
os.makedirs(WORK_DIR, exist_ok=True)


def _pip_install(spec):
    """Install a pip package quietly."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if isinstance(spec, (list, tuple)):
        cmd.extend(list(spec))
    else:
        cmd.append(str(spec))
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def ensure_package(import_name, pip_name=None):
    """Ensure a Python package is importable, installing if needed."""
    if pip_name is None:
        pip_name = import_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"[install] {pip_name}...")
        if _pip_install(pip_name):
            try:
                __import__(import_name)
                return True
            except ImportError:
                pass
        raise RuntimeError(f"Cannot install required package: {pip_name}")


# --- Install dependencies ---
ensure_package("numpy")
ensure_package("pandas")
ensure_package("scipy")
ensure_package("sklearn", "scikit-learn")
ensure_package("statsmodels")
ensure_package("torch")
ensure_package("tqdm")
ensure_package("matplotlib")
ensure_package("arch")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from scipy import stats
from scipy.stats import chi2
from statsmodels.tsa.stattools import acf
from sklearn.manifold import TSNE

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] Python={sys.version_info.major}.{sys.version_info.minor} | "
      f"PyTorch={torch.__version__} | Device={DEVICE}")

# --- Signatory (required for SigCWGAN training) ---
signatory = None
HAS_SIGNATORY = False       # True = signatory available (differentiable, for training)
HAS_IISIGNATURE = False     # True = iisignature available (numpy, for eval metrics only)

try:
    ensure_package("signatory")
    import signatory as _sig_module
    signatory = _sig_module
    HAS_SIGNATORY = True
    print("[ok] signatory loaded (C++ backend — full SigCWGAN support)")
except Exception as e:
    print(f"[warn] signatory not available: {e}")
    print("       SigCWGAN training requires signatory for differentiable signatures.")
    print("       Other models (GARCH, FHS, TimeGAN, Neural SDE) will still run.")

# iisignature is used for evaluation metrics (Signature MMD) when signatory
# is unavailable. It is NOT used for SigCWGAN training.
try:
    ensure_package("iisignature")
    import iisignature
    HAS_IISIGNATURE = True
    if not HAS_SIGNATORY:
        print("[ok] iisignature loaded (for evaluation metrics only, NOT for SigCWGAN training)")
except Exception:
    pass


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """All experiment parameters in one place."""

    # --- Data ---
    p: int = 24                          # Past context window (hours)
    q: int = 6                           # Future generation horizon (hours)
    train_ratios: Tuple[float, ...] = (0.6, 0.7, 0.8)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # --- Models to run ---
    models: Tuple[str, ...] = (
        "sigcwgan",
        "garch_ar1_gjr_skewt",
        "filtered_historical_simulation",
        "timegan",
        "neural_sde",
        "tail_gan",
    )

    # --- SigCWGAN ---
    sig_depth: int = 4
    hidden_dims: Tuple[int, ...] = (50, 50, 50)
    total_steps: int = 1500
    batch_size: int = 64
    mc_size: int = 256
    learning_rate: float = 1e-3
    lr_step_size: int = 100
    lr_gamma: float = 0.9
    sig_scale: float = 0.2
    lag_m: int = 2
    sig_train_windows_cap: int = 8000

    # --- Tail loss ---
    tail_loss_weights: Tuple[float, ...] = (0, 2, 5, 8, 12)  # Sensitivity analysis
    default_tail_loss_weight: float = 8.0
    var_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    # Sigma-based extreme thresholds (on scaled data)
    extreme_sigma_thresholds: Tuple[float, ...] = (1.5, 2.0, 3.0)
    var_weights: Tuple[float, ...] = (1.0, 1.5, 2.0)
    extreme_weights: Tuple[float, ...] = (1.0, 1.5, 2.0)
    tail_over_penalty: float = 15.0    # Heavier penalty for underestimating risk (financially dangerous)
    tail_under_penalty: float = 10.0   # Lighter penalty for overestimating risk (conservative, less harmful)
    # Penalty ratio ablation: list of (over, under) tuples tested at fixed tlw
    penalty_ablation_configs: Tuple[Tuple[float, float], ...] = (
        (10.0, 10.0),   # Symmetric
        (20.0, 5.0),    # Strong asymmetry (risk-averse)
        (5.0, 20.0),    # Reversed asymmetry
        (20.0, 10.0),   # Stronger default-direction
    )
    penalty_ablation_tlw: float = 8.0  # Fixed tail_loss_weight for ablation

    # --- Conditional Tail-GAN (adapted from Cont et al., 2023) ---
    tailgan_hidden_dim: int = 256
    tailgan_latent_dim: int = 64
    tailgan_lr_g: float = 1e-4
    tailgan_lr_d: float = 5e-5
    tailgan_train_steps: int = 1500
    tailgan_batch_size: int = 128
    tailgan_fz_W: float = 10.0         # Scale param for FZ scoring
    tailgan_sort_temp: float = 0.01     # Neural sorting temperature
    tailgan_alphas: Tuple[float, ...] = (0.01, 0.05)  # VaR levels for FZ loss
    tailgan_n_critic: int = 3           # Discriminator steps per generator step

    # --- TimeGAN ---
    timegan_hidden_dim: int = 64
    timegan_context_dim: int = 32
    timegan_latent_dim: int = 8
    timegan_batch_size: int = 64
    timegan_lr: float = 5e-4
    timegan_pretrain_steps: int = 200
    timegan_supervisor_steps: int = 200
    timegan_joint_steps: int = 400

    # --- Neural SDE ---
    neural_sde_hidden_dim: int = 64
    neural_sde_train_steps: int = 1200
    neural_sde_batch_size: int = 256
    neural_sde_lr: float = 1e-3

    # --- Evaluation ---
    n_eval_windows_cap: int = 5000
    acf_max_lag: int = 20
    cvm_bootstrap_iterations: int = 200
    cvm_block_length: int = 24
    cvm_max_points: int = 10000
    mmd_max_paths: int = 512
    mmd_sig_depth: int = 3

    # --- Output ---
    artifact_prefix: str = "tail_sigcwgan_benchmark"
    plots_dir: str = "."
    export_latex: bool = True

    # --- Friedman / Bootstrap ---
    bootstrap_ci_iterations: int = 2000
    dm_hac_lags: int = 3

    device: str = field(default_factory=lambda: DEVICE)


CFG = ExperimentConfig()


# ============================================================================
# 2. DATA UTILITIES
# ============================================================================


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StandardScalerTS:
    """Standard scaler for time series tensors. Fits on axis=(0,1)."""

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor):
        # x shape: [1, T, 1] or [N, T, 1]
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        self.std = x.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
        return self

    def to(self, device):
        """Move scaler parameters to the specified device."""
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


def rolling_window(x: torch.Tensor, window_len: int) -> torch.Tensor:
    """Create rolling windows from a 2D tensor [T, D] -> [N, window_len, D]."""
    T = x.shape[0]
    if T < window_len:
        raise ValueError(f"Series length {T} < window length {window_len}")
    windows = torch.stack([x[t:t + window_len] for t in range(T - window_len + 1)])
    return windows


def load_dataset(csv_path: str, train_ratio: float, cfg: ExperimentConfig):
    """
    Load a BTC dataset CSV and prepare train/test rolling windows.

    Returns dict with all data needed for model training and evaluation.
    """
    df = pd.read_csv(csv_path)
    if "log_return" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'log_return' column")

    log_returns = df["log_return"].dropna().values.astype(np.float64)
    if len(log_returns) < 100:
        raise ValueError(f"Dataset too small: {len(log_returns)} points")

    # Convert to tensor [1, T, 1]
    data_raw = torch.from_numpy(log_returns.reshape(1, -1, 1)).float()
    T = data_raw.shape[1]

    # Train/test split
    split_idx = int(train_ratio * T)
    min_window = cfg.p + cfg.q + 1

    if split_idx < min_window:
        raise ValueError(f"Train segment too short: {split_idx}")
    if (T - split_idx) < min_window:
        raise ValueError(f"Test segment too short: {T - split_idx}")

    # Fit scaler on training data ONLY
    scaler = StandardScalerTS()
    scaler.fit(data_raw[:, :split_idx, :])
    data_scaled = scaler.transform(data_raw)

    # Create rolling windows (on CPU first for inverse_transform)
    train_scaled = data_scaled[0, :split_idx, :]  # [split_idx, 1]
    test_scaled = data_scaled[0, split_idx:, :]    # [T-split_idx, 1]

    x_train = rolling_window(train_scaled, cfg.p + cfg.q)
    x_test = rolling_window(test_scaled, cfg.p + cfg.q)

    # Original-scale versions for evaluation (compute on CPU before moving to GPU)
    x_train_orig = scaler.inverse_transform(x_train).numpy()[..., 0]
    x_test_orig = scaler.inverse_transform(x_test).numpy()[..., 0]

    # Move to GPU
    x_train = x_train.to(cfg.device)
    x_test = x_test.to(cfg.device)
    data_raw = data_raw.to(cfg.device)

    raw_series = log_returns

    return {
        "csv_path": csv_path,
        "scaler": scaler,
        "data_raw": data_raw,
        "x_train": x_train,
        "x_test": x_test,
        "x_train_orig": x_train_orig,
        "x_test_orig": x_test_orig,
        "raw_train": raw_series[:split_idx],
        "raw_test": raw_series[split_idx:],
        "split_idx": split_idx,
        "total_points": T,
        "train_ratio": train_ratio,
    }


def find_dataset_csvs() -> List[str]:
    """Find all valid BTC dataset CSVs in the working directory."""
    csvs = sorted(glob.glob(os.path.join(WORK_DIR, "*.csv")))
    valid = []
    for f in csvs:
        try:
            cols = pd.read_csv(f, nrows=2).columns
            if "log_return" in cols:
                valid.append(f)
        except Exception:
            pass

    # If on Kaggle, copy from /kaggle/input
    if IS_KAGGLE and len(valid) == 0:
        for src in glob.glob("/kaggle/input/**/*.csv", recursive=True):
            try:
                cols = pd.read_csv(src, nrows=2).columns
                if "log_return" in cols:
                    dst = os.path.join(WORK_DIR, os.path.basename(src))
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    valid.append(dst)
            except Exception:
                pass

    # Also search the script's own directory (common when running locally)
    if len(valid) == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir != WORK_DIR:
            for f in sorted(glob.glob(os.path.join(script_dir, "*.csv"))):
                try:
                    cols = pd.read_csv(f, nrows=2).columns
                    if "log_return" in cols:
                        valid.append(f)
                except Exception:
                    pass

    if len(valid) == 0:
        raise FileNotFoundError(
            "No CSV files with 'log_return' column found. "
            "Place your BTC dataset CSVs in the working directory."
        )

    print(f"[data] Found {len(valid)} dataset(s): {[os.path.basename(f) for f in valid]}")
    return valid


def sample_indices(n: int, batch_size: int, device: str = "cpu") -> torch.Tensor:
    """Sample random indices, with replacement if batch_size > n."""
    replace = batch_size > n
    idx = np.random.choice(n, size=batch_size, replace=replace)
    return torch.from_numpy(idx).to(device)


# ============================================================================
# 3. SIGNATURE UTILITIES
# ============================================================================


class Scale:
    """Multiply path by a scalar (prevents signature explosion)."""
    def __init__(self, scale: float = 0.2):
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class Cumsum:
    """Cumulative sum along time axis (returns -> price-like path)."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(dim=1)


class AddLags:
    """Unfold with lag m to create multi-dimensional embedding."""
    def __init__(self, m: int = 2):
        self.m = m

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, dim]
        if x.shape[1] < self.m:
            raise ValueError(f"Path too short for lag {self.m}: length={x.shape[1]}")
        unfolded = x.unfold(dimension=1, size=self.m, step=1)  # [B, T-m+1, D, m]
        B, T_new, D, M = unfolded.shape
        return unfolded.permute(0, 1, 3, 2).contiguous().view(B, T_new, D * M)


class LeadLag:
    """Lead-lag embedding for capturing temporal asymmetries."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
        return torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)


def build_augmentation_pipeline(cfg: ExperimentConfig):
    """Build the signature augmentation pipeline (NO Concat)."""
    return [
        Scale(cfg.sig_scale),
        Cumsum(),
        AddLags(cfg.lag_m),
        LeadLag(),
    ]


def augment_and_compute_signature(
    x: torch.Tensor,
    augmentations: list,
    depth: int,
    require_grad: bool = False,
) -> torch.Tensor:
    """
    Apply augmentations then compute truncated signature.

    Args:
        x: Input tensor [batch, time, dim]
        augmentations: List of augmentation callables
        depth: Signature truncation depth
        require_grad: If True, requires signatory (differentiable).
                      If False, can use iisignature (for eval metrics).
    """
    y = x
    for aug in augmentations:
        y = aug(y)

    # Use signatory (C++, differentiable) if available
    if signatory is not None:
        return signatory.signature(y, depth)

    # Fall back to iisignature (numpy, NOT differentiable)
    if HAS_IISIGNATURE:
        if require_grad:
            raise RuntimeError(
                "Differentiable signature computation requires signatory. "
                "iisignature is numpy-based and does not support autograd."
            )
        import iisignature
        y_np = y.detach().cpu().numpy().astype(np.float64)
        sigs = iisignature.sig(y_np, depth)  # [batch, sig_dim]
        return torch.from_numpy(sigs).float().to(x.device)

    raise RuntimeError("No signature library available (need signatory or iisignature)")


# ============================================================================
# 4. SIGCWGAN MODEL
# ============================================================================


class EnhancedTailLoss(nn.Module):
    """
    Tail-aware training loss for the generator.

    Two components:
    1. VaR quantile penalty: penalises incorrect VaR at 90/95/99%
    2. Extreme event frequency penalty: penalises incorrect frequency of
       extreme moves beyond sigma-based thresholds (1.5σ, 2σ, 3σ)

    Both use asymmetric penalties: underestimating risk is punished more
    heavily than overestimating it (reflecting financial risk priorities).
    """

    def __init__(
        self,
        x_real_future: torch.Tensor,
        var_levels: Tuple[float, ...] = (0.90, 0.95, 0.99),
        extreme_sigma_thresholds: Tuple[float, ...] = (1.5, 2.0, 3.0),
        var_weights: Tuple[float, ...] = (1.0, 1.5, 2.0),
        extreme_weights: Tuple[float, ...] = (1.0, 1.5, 2.0),
        overhit_penalty: float = 15.0,
        underhit_penalty: float = 10.0,
    ):
        super().__init__()

        flat = x_real_future.reshape(-1)
        dev = flat.device

        # VaR quantiles (left tail)
        var_probs = torch.tensor([1.0 - v for v in var_levels], dtype=torch.float32, device=dev)
        self.register_buffer("var_probs", var_probs)
        self.register_buffer("var_targets", torch.quantile(flat, var_probs))

        # Extreme event frequency targets (sigma-based thresholds)
        # Thresholds are multiples of the data's own std, so "1.5" means
        # "beyond 1.5 standard deviations". On scaled data std≈1, so
        # threshold = sigma_t * std ≈ sigma_t directly.
        flat_std = flat.std().clamp(min=1e-8)
        sigma_t = torch.tensor(extreme_sigma_thresholds, dtype=torch.float32, device=dev)
        absolute_thresholds = sigma_t * flat_std
        self.register_buffer("extreme_thresholds", absolute_thresholds)
        targets = [(torch.abs(flat) > th).float().mean() for th in absolute_thresholds]
        self.register_buffer(
            "extreme_freq_targets",
            torch.tensor(targets, dtype=torch.float32, device=dev),
        )

        self.register_buffer(
            "var_weights", torch.tensor(var_weights, dtype=torch.float32, device=dev)
        )
        self.register_buffer(
            "extreme_weights", torch.tensor(extreme_weights, dtype=torch.float32, device=dev)
        )
        self.overhit_penalty = overhit_penalty
        self.underhit_penalty = underhit_penalty

    def forward(self, x_fake_future: torch.Tensor) -> torch.Tensor:
        flat = x_fake_future.reshape(-1)

        # --- VaR component ---
        fake_var = torch.quantile(flat, self.var_probs.to(flat.device))
        diff = fake_var - self.var_targets.to(flat.device)
        # Negative diff = fake tails thinner than real = underestimating risk
        var_underest = torch.clamp(-diff, min=0.0)   # Fake less extreme → underestimates risk
        var_overest = torch.clamp(diff, min=0.0)      # Fake more extreme → overestimates risk
        var_loss = self.var_weights.to(flat.device) * (
            self.overhit_penalty * var_underest + self.underhit_penalty * var_overest
        )

        # --- Extreme frequency component ---
        extreme_freq = torch.stack([
            (torch.abs(flat) > t).float().mean()
            for t in self.extreme_thresholds.to(flat.device)
        ])
        ext_diff = extreme_freq - self.extreme_freq_targets.to(flat.device)
        ext_overest = torch.clamp(ext_diff, min=0.0)    # Too many extremes → overestimates risk
        ext_underest = torch.clamp(-ext_diff, min=0.0)  # Too few extremes → underestimates risk
        ext_loss = self.extreme_weights.to(flat.device) * (
            self.overhit_penalty * ext_underest + self.underhit_penalty * ext_overest
        )

        return torch.cat([var_loss, ext_loss]).mean()


class ResidualBlock(nn.Module):
    """Residual block with PReLU activation."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU()
        self.has_residual = (input_dim == output_dim)

    def forward(self, x):
        y = self.activation(self.linear(x))
        return (x + y) if self.has_residual else y


class ResFNN(nn.Module):
    """Residual feedforward network."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(d, h))
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ARGenerator(nn.Module):
    """
    Autoregressive generator (AR-FNN from Ni et al., Section 5.3).

    Generates future path one step at a time, feeding each generated
    step back as input for the next.
    """
    def __init__(self, p: int, data_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = ResFNN(
            input_dim=p * data_dim + latent_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
        )

    def forward(self, z: torch.Tensor, x_past: torch.Tensor):
        """
        z: [batch, q, latent_dim]  — noise
        x_past: [batch, p, dim]    — conditioning context
        """
        generated = []
        for t in range(z.shape[1]):
            z_t = z[:, t:t+1, :]                           # [B, 1, latent]
            x_flat = x_past.reshape(x_past.shape[0], 1, -1)  # [B, 1, p*dim]
            inp = torch.cat([z_t, x_flat], dim=-1)          # [B, 1, p*dim+latent]
            x_gen = self.network(inp)                        # [B, 1, dim]
            x_past = torch.cat([x_past[:, 1:], x_gen], dim=1)
            generated.append(x_gen)
        return torch.cat(generated, dim=1)  # [B, q, dim]

    def sample(self, q: int, x_past: torch.Tensor):
        """Generate q future steps given x_past."""
        z = torch.randn(x_past.size(0), q, self.latent_dim, device=x_past.device)
        return self.forward(z, x_past)


class SigCWGAN:
    """
    Conditional Sig-Wasserstein GAN with optional tail loss.

    Training procedure:
    1. Compute signatures of past and future training windows
    2. Fit linear regression: E[S_future | S_past] = L(S_past)
    3. Train generator to minimise:
       loss = ||L(S_past) - E_z[S(G(x_past, z))]||₂ + w * tail_loss
    """

    def __init__(
        self,
        x_train: torch.Tensor,      # [N, p+q, 1]
        cfg: ExperimentConfig,
        tail_loss_weight: float,
        seed: int,
    ):
        if not HAS_SIGNATORY:
            raise RuntimeError("SigCWGAN requires signatory")

        set_global_seed(seed)
        self.cfg = cfg
        self.device = cfg.device
        self.tail_loss_weight = tail_loss_weight

        # Limit training windows
        cap = cfg.sig_train_windows_cap
        if cap and x_train.shape[0] > cap:
            x_train = x_train[:cap]
        self.x_train = x_train

        self.x_past = x_train[:, :cfg.p, :]    # [N, p, 1]
        self.x_future = x_train[:, cfg.p:, :]  # [N, q, 1]

        # Build augmentation pipeline
        self.augmentations = build_augmentation_pipeline(cfg)

        # Compute signatures
        print("  Computing training signatures...")
        with torch.no_grad():
            sigs_past = augment_and_compute_signature(
                self.x_past, self.augmentations, cfg.sig_depth
            )
            sigs_future = augment_and_compute_signature(
                self.x_future, self.augmentations, cfg.sig_depth
            )

        # Linear regression: E[S_future | S_past]
        print("  Calibrating linear regression...")
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        lm.fit(sigs_past.cpu().numpy(), sigs_future.cpu().numpy())
        self.sigs_pred = torch.from_numpy(
            lm.predict(sigs_past.cpu().numpy())
        ).float().to(self.device)

        # Generator
        self.G = ARGenerator(
            p=cfg.p,
            data_dim=1,
            hidden_dims=cfg.hidden_dims,
            latent_dim=1,
        ).to(self.device)

        # Tail loss
        self.tail_loss_fn = None
        if tail_loss_weight > 0:
            self.tail_loss_fn = EnhancedTailLoss(
                self.x_future,
                var_levels=cfg.var_levels,
                extreme_sigma_thresholds=cfg.extreme_sigma_thresholds,
                var_weights=cfg.var_weights,
                extreme_weights=cfg.extreme_weights,
                overhit_penalty=cfg.tail_over_penalty,
                underhit_penalty=cfg.tail_under_penalty,
            ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.G.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )

        self.batch_size = min(cfg.batch_size, x_train.shape[0])
        self.loss_history = {"total": [], "sig": [], "tail": []}

    def _sample_batch(self):
        idx = sample_indices(self.sigs_pred.shape[0], self.batch_size, self.device)
        return self.sigs_pred[idx], self.x_past[idx]

    def _compute_fake_sigs(self, x_past_batch):
        """Monte Carlo estimate of E_z[S(G(x_past, z))]."""
        x_past_mc = x_past_batch.repeat(self.cfg.mc_size, 1, 1)
        x_fake = self.G.sample(self.cfg.q, x_past_mc)

        sigs_fake = augment_and_compute_signature(
            x_fake, self.augmentations, self.cfg.sig_depth
        )
        # Average over MC samples: reshape [mc*B, sig_dim] -> [mc, B, sig_dim] -> mean over mc
        sigs_fake = sigs_fake.reshape(
            self.cfg.mc_size, x_past_batch.shape[0], -1
        ).mean(0)

        return sigs_fake, x_fake

    def train_step(self):
        self.G.train()
        self.optimizer.zero_grad()

        sigs_pred_batch, x_past_batch = self._sample_batch()
        sigs_fake, x_fake = self._compute_fake_sigs(x_past_batch)

        # Signature W1 loss (Algorithm 1, Ni et al. 2020)
        # Gradients flow through signatory's differentiable signature computation
        sig_loss = torch.norm(sigs_pred_batch - sigs_fake, p=2, dim=1).mean()

        # Tail loss (our contribution — operates directly on generated paths)
        tail_loss_val = torch.tensor(0.0, device=self.device)
        if self.tail_loss_fn is not None and self.tail_loss_weight > 0:
            tail_loss_val = self.tail_loss_fn(x_fake)

        # Total: Sig-W1 loss + lambda * tail loss
        total_loss = sig_loss + self.tail_loss_weight * tail_loss_val

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        self.loss_history["total"].append(float(total_loss.item()))
        self.loss_history["sig"].append(float(sig_loss.item()))
        self.loss_history["tail"].append(float(tail_loss_val.item()))

    def fit(self):
        for _ in tqdm(range(self.cfg.total_steps), ncols=80, desc="SigCWGAN"):
            self.train_step()

    def generate(self, x_past_test: torch.Tensor) -> torch.Tensor:
        """Generate future paths for given past contexts."""
        self.G.eval()
        with torch.no_grad():
            return self.G.sample(self.cfg.q, x_past_test)


def run_sigcwgan(bundle: dict, seed: int, cfg: ExperimentConfig,
                 tail_loss_weight: float,
                 over_penalty: float = None,
                 under_penalty: float = None) -> dict:
    """Train SigCWGAN and generate test paths."""
    if not HAS_SIGNATORY:
        raise RuntimeError(
            "SigCWGAN requires signatory (C++ backend) for differentiable "
            "signature computation (Algorithm 1, Ni et al. 2020). "
            "iisignature is numpy-based and does NOT support autograd.\n"
            "Install: pip install signatory==X.X.X.X.X.X "
            "(version must match your PyTorch version).\n"
            "On Kaggle/Amazon with GPU, signatory installs correctly."
        )

    # Override penalty ratios if provided (for ablation study)
    orig_over = cfg.tail_over_penalty
    orig_under = cfg.tail_under_penalty
    if over_penalty is not None:
        cfg.tail_over_penalty = over_penalty
    if under_penalty is not None:
        cfg.tail_under_penalty = under_penalty

    start = time.perf_counter()

    try:
        model = SigCWGAN(bundle["x_train"], cfg, tail_loss_weight, seed)
        model.fit()

        n_eval = min(cfg.n_eval_windows_cap, bundle["x_test"].shape[0])
        x_past_test = bundle["x_test"][:n_eval, :cfg.p, :]
        x_fake_scaled = model.generate(x_past_test)

        # Inverse transform to original scale
        scaler = bundle["scaler"]
        x_fake_orig = scaler.inverse_transform(x_fake_scaled.cpu()).numpy()[..., 0]
        x_real_orig = scaler.inverse_transform(
            bundle["x_test"][:n_eval, cfg.p:, :].cpu()
        ).numpy()[..., 0]

        runtime = time.perf_counter() - start
    finally:
        # Always restore original penalty values, even if training crashes
        cfg.tail_over_penalty = orig_over
        cfg.tail_under_penalty = orig_under

    return {
        "fake_paths": x_fake_orig,
        "real_paths": x_real_orig,
        "n_eval": n_eval,
        "runtime_sec": runtime,
        "final_loss": model.loss_history["total"][-1] if model.loss_history["total"] else np.nan,
        "tail_loss_weight": tail_loss_weight,
    }


# ============================================================================
# 5. BASELINE MODELS
# ============================================================================


# --- 5.1 GARCH Family ---

def _fit_ar1_gjr_skewt(train_returns_pct: np.ndarray):
    """Fit AR(1)-GJR-GARCH(1,1) with skewed Student-t innovations."""
    from arch import arch_model
    am = arch_model(
        train_returns_pct,
        mean="AR", lags=1,
        vol="GARCH", p=1, o=1, q=1,
        dist="skewstudent",
        rescale=False,
    )
    return am.fit(disp="off", show_warning=False, update_freq=0)


def _extract_garch_params(params) -> dict:
    """Extract GARCH parameters from fitted model."""
    def _get(key, default=0.0):
        if key in params.index:
            return float(params[key])
        for idx in params.index:
            if key.lower() in idx.lower():
                return float(params[idx])
        return default

    # Find AR(1) coefficient
    phi = 0.0
    for idx in params.index:
        if "[1]" in idx and not any(k in idx.lower() for k in
                ["alpha", "beta", "gamma", "omega", "eta", "lambda", "nu", "sigma"]):
            phi = float(params[idx])
            break

    return {
        "const": _get("Const", _get("mu", 0.0)),
        "phi": phi,
        "omega": max(_get("omega", 1e-6), 1e-10),
        "alpha": max(_get("alpha[1]", 0.05), 0.0),
        "beta": max(_get("beta[1]", 0.9), 0.0),
        "gamma": max(_get("gamma[1]", 0.0), 0.0),
        "eta": max(_get("eta", 8.0), 2.05),
        "skew": float(np.clip(_get("lambda", 0.0), -0.99, 0.99)),
    }


def _simulate_garch_paths(
    past_contexts: np.ndarray,
    q: int,
    gp: dict,
    innovation_mode: str,
    resid_pool: Optional[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Simulate future return paths from GARCH model."""
    rng = np.random.default_rng(seed)
    n_eval = past_contexts.shape[0]
    fake_paths = np.zeros((n_eval, q), dtype=np.float64)

    for i in range(n_eval):
        ctx_pct = past_contexts[i] * 100.0
        prev_r = float(ctx_pct[-1])
        prev_r_lag = float(ctx_pct[-2]) if ctx_pct.size >= 2 else prev_r
        eps_prev = prev_r - (gp["const"] + gp["phi"] * prev_r_lag)
        sigma2_prev = float(np.var(ctx_pct))
        if sigma2_prev <= 1e-12:
            denom = max(1e-6, 1.0 - gp["alpha"] - gp["beta"] - 0.5 * gp["gamma"])
            sigma2_prev = max(gp["omega"] / denom, 1e-6)

        for t in range(q):
            ind_neg = 1.0 if eps_prev < 0 else 0.0
            sigma2_t = (gp["omega"]
                        + gp["alpha"] * eps_prev**2
                        + gp["beta"] * sigma2_prev
                        + gp["gamma"] * ind_neg * eps_prev**2)
            sigma2_t = max(float(sigma2_t), 1e-10)

            if innovation_mode == "skewstudent":
                from arch.univariate import SkewStudent
                dist = SkewStudent()
                u = np.clip(rng.random(), 1e-8, 1 - 1e-8)
                z_t = float(dist.ppf(np.array([u]),
                            parameters=np.array([gp["eta"], gp["skew"]]))[0])
            elif innovation_mode == "bootstrap":
                z_t = float(rng.choice(resid_pool, size=1)[0])
            else:
                z_t = float(rng.standard_normal())

            eps_t = np.sqrt(sigma2_t) * z_t
            r_t = gp["const"] + gp["phi"] * prev_r + eps_t
            fake_paths[i, t] = r_t / 100.0

            prev_r = r_t
            eps_prev = eps_t
            sigma2_prev = sigma2_t

    return fake_paths


def run_garch(bundle: dict, seed: int, cfg: ExperimentConfig) -> dict:
    """Run AR(1)-GJR-GARCH(1,1) with skewed Student-t innovations."""
    n_eval = min(cfg.n_eval_windows_cap, bundle["x_test"].shape[0])
    train_pct = bundle["raw_train"] * 100.0
    test_orig = bundle["x_test_orig"][:n_eval]
    past_ctx = test_orig[:, :cfg.p]
    real_paths = test_orig[:, cfg.p:cfg.p + cfg.q]

    start = time.perf_counter()
    fit_res = _fit_ar1_gjr_skewt(train_pct)
    gp = _extract_garch_params(fit_res.params)
    runtime = time.perf_counter() - start

    fake_paths = _simulate_garch_paths(
        past_ctx, cfg.q, gp, "skewstudent", None, seed
    )

    return {
        "fake_paths": fake_paths,
        "real_paths": real_paths,
        "n_eval": n_eval,
        "runtime_sec": runtime,
        "final_loss": np.nan,
    }


def run_fhs(bundle: dict, seed: int, cfg: ExperimentConfig) -> dict:
    """Run Filtered Historical Simulation."""
    n_eval = min(cfg.n_eval_windows_cap, bundle["x_test"].shape[0])
    train_pct = bundle["raw_train"] * 100.0
    test_orig = bundle["x_test_orig"][:n_eval]
    past_ctx = test_orig[:, :cfg.p]
    real_paths = test_orig[:, cfg.p:cfg.p + cfg.q]

    start = time.perf_counter()
    fit_res = _fit_ar1_gjr_skewt(train_pct)
    gp = _extract_garch_params(fit_res.params)

    # Get standardised residuals for bootstrapping
    resid_pool = np.asarray(fit_res.std_resid, dtype=np.float64)
    resid_pool = resid_pool[np.isfinite(resid_pool) & (np.abs(resid_pool) < 20.0)]
    if resid_pool.size < 50:
        raise RuntimeError("FHS: not enough valid residuals")
    resid_pool = (resid_pool - resid_pool.mean()) / max(resid_pool.std(), 1e-8)
    runtime = time.perf_counter() - start

    fake_paths = _simulate_garch_paths(
        past_ctx, cfg.q, gp, "bootstrap", resid_pool, seed
    )

    return {
        "fake_paths": fake_paths,
        "real_paths": real_paths,
        "n_eval": n_eval,
        "runtime_sec": runtime,
        "final_loss": np.nan,
    }


# --- 5.2 TimeGAN ---

class _TGContextEncoder(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, context_dim)

    def forward(self, x_past):
        _, h = self.gru(x_past)
        return self.proj(h[-1])


class _TGEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        h, _ = self.gru(x)
        return h


class _TGRecovery(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        y, _ = self.gru(h)
        return self.out(y)


class _TGGenerator(nn.Module):
    def __init__(self, latent_dim, context_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=latent_dim + context_dim,
                          hidden_size=hidden_dim, batch_first=True)

    def forward(self, z, c):
        c_rep = c.unsqueeze(1).repeat(1, z.shape[1], 1)
        return self.gru(torch.cat([z, c_rep], dim=-1))[0]


class _TGSupervisor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, h):
        return self.gru(h)[0]


class _TGDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        out, _ = self.gru(h)
        return self.head(out[:, -1])


def run_timegan(bundle: dict, seed: int, cfg: ExperimentConfig) -> dict:
    """Run conditional TimeGAN baseline."""
    set_global_seed(seed)
    device = cfg.device
    p, q = cfg.p, cfg.q

    x_train = bundle["x_train"]
    x_test = bundle["x_test"]
    scaler = bundle["scaler"]

    x_past = x_train[:, :p, :1].to(device)
    x_future = x_train[:, p:p+q, :1].to(device)

    hd = cfg.timegan_hidden_dim
    cd = cfg.timegan_context_dim
    ld = cfg.timegan_latent_dim
    bs = min(cfg.timegan_batch_size, x_train.shape[0])

    ctx_enc = _TGContextEncoder(hd, cd).to(device)
    embedder = _TGEmbedder(hd).to(device)
    recovery = _TGRecovery(hd).to(device)
    generator = _TGGenerator(ld, cd, hd).to(device)
    supervisor = _TGSupervisor(hd).to(device)
    discriminator = _TGDiscriminator(hd).to(device)

    ae_opt = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=cfg.timegan_lr)
    sup_opt = optim.Adam(supervisor.parameters(), lr=cfg.timegan_lr)
    d_opt = optim.Adam(discriminator.parameters(), lr=cfg.timegan_lr)
    g_opt = optim.Adam(
        list(generator.parameters()) + list(supervisor.parameters()) +
        list(ctx_enc.parameters()) + list(embedder.parameters()) +
        list(recovery.parameters()),
        lr=cfg.timegan_lr,
    )

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    def _batch():
        idx = sample_indices(x_future.shape[0], bs, device)
        return x_past[idx], x_future[idx]

    start = time.perf_counter()

    # Phase 1: Autoencoder pretrain
    for _ in tqdm(range(cfg.timegan_pretrain_steps), ncols=80, desc="TG-AE"):
        pb, fb = _batch()
        ae_opt.zero_grad()
        mse(recovery(embedder(fb)), fb).backward()
        ae_opt.step()

    # Phase 2: Supervisor pretrain
    for _ in tqdm(range(cfg.timegan_supervisor_steps), ncols=80, desc="TG-SUP"):
        _, fb = _batch()
        sup_opt.zero_grad()
        with torch.no_grad():
            h_real = embedder(fb)
        h_sup = supervisor(h_real)
        (mse(h_sup[:, :-1], h_real[:, 1:]) if h_real.shape[1] > 1
         else mse(h_sup, h_real)).backward()
        sup_opt.step()

    # Phase 3: Joint training
    final_g_loss = np.nan
    for _ in tqdm(range(cfg.timegan_joint_steps), ncols=80, desc="TG-JOINT"):
        pb, fb = _batch()
        c = ctx_enc(pb)
        z = torch.randn(fb.shape[0], q, ld, device=device)

        # Discriminator step
        with torch.no_grad():
            h_real = embedder(fb)
            h_fake = supervisor(generator(z, c))
        d_opt.zero_grad()
        d_loss = (bce(discriminator(h_real), torch.ones(bs, 1, device=device)) +
                  bce(discriminator(h_fake), torch.zeros(bs, 1, device=device)))
        d_loss.backward()
        d_opt.step()

        # Generator step
        g_opt.zero_grad()
        h_real2 = embedder(fb)
        h_gen = generator(z, c)
        h_hat = supervisor(h_gen)
        f_hat = recovery(h_hat)

        g_adv = bce(discriminator(h_hat), torch.ones(bs, 1, device=device))
        g_sup = (mse(h_hat[:, :-1], h_gen[:, 1:]) if h_hat.shape[1] > 1
                 else mse(h_hat, h_gen))
        g_mom = (torch.abs(f_hat.mean(dim=(0, 1)) - fb.mean(dim=(0, 1))).mean() +
                 torch.abs(f_hat.std(dim=(0, 1)) - fb.std(dim=(0, 1))).mean())
        recon = mse(recovery(h_real2), fb)

        g_loss = g_adv + 50 * g_sup + 10 * g_mom + 5 * recon
        g_loss.backward()
        g_opt.step()
        final_g_loss = float(g_loss.item())

    runtime = time.perf_counter() - start

    # Inference
    n_eval = min(cfg.n_eval_windows_cap, x_test.shape[0])
    x_past_test = x_test[:n_eval, :p, :1].to(device)

    for m in [ctx_enc, embedder, recovery, generator, supervisor]:
        m.eval()

    with torch.no_grad():
        c = ctx_enc(x_past_test)
        z = torch.randn(n_eval, q, ld, device=device)
        x_fake_scaled = recovery(supervisor(generator(z, c)))

    x_fake_orig = scaler.inverse_transform(x_fake_scaled.cpu()).numpy()[..., 0]
    x_real_orig = scaler.inverse_transform(
        x_test[:n_eval, p:p+q, :1].cpu()
    ).numpy()[..., 0]

    return {
        "fake_paths": x_fake_orig,
        "real_paths": x_real_orig,
        "n_eval": n_eval,
        "runtime_sec": runtime,
        "final_loss": final_g_loss,
    }


# --- 5.3 Neural SDE ---

class NeuralSDE1D(nn.Module):
    """1D Neural SDE with learned drift and diffusion."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.drift = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.diff = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Softplus(),
        )

    def forward_drift(self, x):
        return self.drift(x)

    def forward_diff(self, x):
        return self.diff(x) + 1e-4


def run_neural_sde(bundle: dict, seed: int, cfg: ExperimentConfig) -> dict:
    """Run Neural SDE baseline."""
    set_global_seed(seed)
    device = cfg.device
    p, q = cfg.p, cfg.q

    scaler = bundle["scaler"]
    data_scaled = scaler.transform(bundle["data_raw"])
    scaled_train = data_scaled[0, :bundle["split_idx"], 0].to(device)

    model = NeuralSDE1D(cfg.neural_sde_hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.neural_sde_lr)
    bs = min(cfg.neural_sde_batch_size, int(scaled_train.shape[0] - 1))

    start = time.perf_counter()

    for _ in tqdm(range(cfg.neural_sde_train_steps), ncols=80, desc="NeuralSDE"):
        idx = sample_indices(int(scaled_train.shape[0] - 1), bs, device)
        x_t = scaled_train[idx].unsqueeze(-1)
        x_next = scaled_train[idx + 1].unsqueeze(-1)

        mu = x_t + model.forward_drift(x_t)
        sigma = model.forward_diff(x_t)
        nll = 0.5 * (((x_next - mu) / sigma) ** 2 + 2 * torch.log(sigma))

        opt.zero_grad()
        nll.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    runtime = time.perf_counter() - start

    # Inference: Euler-Maruyama from last context value
    x_test = bundle["x_test"]
    n_eval = min(cfg.n_eval_windows_cap, x_test.shape[0])
    x_context = x_test[:n_eval, :p, :1].to(device)

    model.eval()
    with torch.no_grad():
        x_cur = x_context[:, -1:, :]
        steps = []
        for _ in range(q):
            mu = x_cur + model.forward_drift(x_cur)
            sigma = model.forward_diff(x_cur)
            x_next = mu + sigma * torch.randn_like(x_cur)
            steps.append(x_next)
            x_cur = x_next
        x_fake_scaled = torch.cat(steps, dim=1)

    x_fake_orig = scaler.inverse_transform(x_fake_scaled.cpu()).numpy()[..., 0]
    x_real_orig = scaler.inverse_transform(
        x_test[:n_eval, p:p+q, :1].cpu()
    ).numpy()[..., 0]

    return {
        "fake_paths": x_fake_orig,
        "real_paths": x_real_orig,
        "n_eval": n_eval,
        "runtime_sec": runtime,
        "final_loss": np.nan,
    }


# --- 5.4 Conditional Tail-GAN (adapted from Cont et al., Management Science 2023) ---
# Uses the Fissler-Ziegel (FZ) joint scoring function for VaR and ES as the
# adversarial loss. Adapted from the unconditional Tail-GAN to our conditional
# benchmark setup (past window -> future window).
# Reference: Cont, Cucuringu, Xu, Zhang (2023). "Tail-GAN: Learning to
#   Simulate Tail Risk Scenarios." Management Science.
# FZ scoring: Fissler & Ziegel (2016). "Higher order elicitability and
#   Osband's principle." Annals of Statistics 44, 1680-1707.


class TailGANGenerator(nn.Module):
    """Conditional generator: takes [past_window, noise] -> future_window."""
    def __init__(self, p: int, q: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p + latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, q),
        )
        self.latent_dim = latent_dim

    def forward(self, x_past: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_past, z], dim=-1)
        return self.net(inp)


class TailGANDiscriminator(nn.Module):
    """Discriminator that predicts (VaR, ES) at each alpha level."""
    def __init__(self, q: int, n_alphas: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 2 * n_alphas),  # (VaR, ES) per alpha
        )
        self.n_alphas = n_alphas

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        # Split into VaR and ES estimates
        var_es = out.view(-1, self.n_alphas, 2)
        var = var_es[:, :, 0]   # (batch, n_alphas)
        es = var_es[:, :, 1]    # (batch, n_alphas)
        # ES must be <= VaR (for left tail), enforce with softplus offset
        es = var - F.softplus(var - es)
        return var, es


def fissler_ziegel_score(
    var_pred: torch.Tensor,
    es_pred: torch.Tensor,
    y: torch.Tensor,
    alphas: torch.Tensor,
    W: float = 10.0,
) -> torch.Tensor:
    """
    Fissler-Ziegel consistent scoring function for joint (VaR, ES).

    Uses parameterisation from Cont et al. (2023):
      G1(v, W) = -W * v^2 / 2
      G2(e, alpha) = alpha * e
      zeta2(e, alpha) = alpha * e^2 / 2

    Args:
        var_pred: (batch, n_alphas) VaR predictions
        es_pred:  (batch, n_alphas) ES predictions
        y:        (batch,) or (batch, 1) observed PnL values
        alphas:   (n_alphas,) quantile levels (e.g. [0.01, 0.05])
        W:        scale parameter for quantile term
    Returns:
        scalar score (lower = better for real data)
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)  # (batch, 1)

    v = var_pred   # (batch, n_alphas)
    e = es_pred    # (batch, n_alphas)
    alpha = alphas.unsqueeze(0)  # (1, n_alphas)

    hit = (y <= v).float()  # indicator 1{y <= v}

    # Term 1: quantile scoring via G1
    term1 = (alpha - hit) * (-W * v * v / 2) + hit * (-W * y * y / 2)

    # Term 2: ES scoring via G2
    term2 = alpha * e * (e - v + hit * (v - y) / alpha.clamp(min=1e-8))

    # Term 3: calibration term -zeta2(e)
    term3 = -alpha * e * e / 2

    score = (term1 + term2 + term3).mean()
    return score


def run_tail_gan(bundle: dict, seed: int, cfg: ExperimentConfig) -> dict:
    """
    Run Conditional Tail-GAN baseline.

    Adapted from Cont, Cucuringu, Xu, Zhang (2023) to our conditional
    setup. Generator produces future returns given past; discriminator
    predicts (VaR, ES) using Fissler-Ziegel scoring.
    """
    set_global_seed(seed)
    device = cfg.device
    p, q = cfg.p, cfg.q

    x_train = bundle["x_train"].to(device)
    x_test = bundle["x_test"].to(device)
    scaler = bundle["scaler"]
    n_train = x_train.shape[0]

    # Extract past/future windows (scaled)
    train_past = x_train[:, :p, 0]    # (N, p)
    train_future = x_train[:, p:p+q, 0]  # (N, q)

    n_alphas = len(cfg.tailgan_alphas)
    alphas = torch.tensor(cfg.tailgan_alphas, dtype=torch.float32, device=device)

    gen = TailGANGenerator(p, q, cfg.tailgan_latent_dim, cfg.tailgan_hidden_dim).to(device)
    disc = TailGANDiscriminator(q, n_alphas, cfg.tailgan_hidden_dim // 2).to(device)

    opt_g = optim.Adam(gen.parameters(), lr=cfg.tailgan_lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=cfg.tailgan_lr_d, betas=(0.5, 0.999))

    bs = min(cfg.tailgan_batch_size, n_train)

    start = time.perf_counter()

    for step in tqdm(range(cfg.tailgan_train_steps), ncols=80, desc="TailGAN"):
        # --- Train Discriminator ---
        for _ in range(cfg.tailgan_n_critic):
            idx = sample_indices(n_train, bs, device)
            x_p = train_past[idx]
            x_f_real = train_future[idx]

            z = torch.randn(bs, cfg.tailgan_latent_dim, device=device)
            with torch.no_grad():
                x_f_fake = gen(x_p, z)

            # Discriminator predicts (VaR, ES) on real and fake
            # We compute portfolio PnL as the mean return of the future window
            pnl_real = x_f_real.mean(dim=-1)  # (bs,)
            pnl_fake = x_f_fake.mean(dim=-1)  # (bs,)

            var_r, es_r = disc(x_f_real)
            var_f, es_f = disc(x_f_fake)

            # FZ score: disc maximises on real, minimises on fake
            score_real = fissler_ziegel_score(var_r, es_r, pnl_real, alphas, cfg.tailgan_fz_W)
            score_fake = fissler_ziegel_score(var_f, es_f, pnl_fake, alphas, cfg.tailgan_fz_W)

            loss_d = score_real - score_fake  # Disc minimises FZ on real (good calibration)

            opt_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
            opt_d.step()

        # --- Train Generator ---
        idx = sample_indices(n_train, bs, device)
        x_p = train_past[idx]

        z = torch.randn(bs, cfg.tailgan_latent_dim, device=device)
        x_f_fake = gen(x_p, z)
        pnl_fake = x_f_fake.mean(dim=-1)
        var_f, es_f = disc(x_f_fake)

        # Generator wants disc to score fake like real (minimize FZ score)
        loss_g = fissler_ziegel_score(var_f, es_f, pnl_fake, alphas, cfg.tailgan_fz_W)

        # Also add simple MSE on moments (mean, std) to stabilise training
        real_mean = train_future.mean()
        real_std = train_future.std()
        moment_loss = (x_f_fake.mean() - real_mean) ** 2 + (x_f_fake.std() - real_std) ** 2
        loss_g = loss_g + 0.1 * moment_loss

        opt_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt_g.step()

    runtime = time.perf_counter() - start

    # --- Inference ---
    n_eval = min(cfg.n_eval_windows_cap, x_test.shape[0])
    test_past = x_test[:n_eval, :p, 0]

    gen.eval()
    with torch.no_grad():
        z = torch.randn(n_eval, cfg.tailgan_latent_dim, device=device)
        x_fake_scaled = gen(test_past, z).unsqueeze(-1)  # (n_eval, q, 1)

    x_fake_orig = scaler.inverse_transform(x_fake_scaled.cpu()).numpy()[..., 0]
    x_real_orig = scaler.inverse_transform(
        x_test[:n_eval, p:p+q, :1].cpu()
    ).numpy()[..., 0]

    return {
        "fake_paths": x_fake_orig,
        "real_paths": x_real_orig,
        "n_eval": n_eval,
        "runtime_sec": runtime,
        "final_loss": float(loss_g.item()),
    }


# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================


def _safe_log(x, eps=1e-12):
    return float(np.log(max(x, eps)))


def kupiec_test(violations: np.ndarray, expected_prob: float) -> Tuple[float, float]:
    """Kupiec POF test for VaR backtesting."""
    v = np.asarray(violations).astype(int)
    n = v.size
    x = v.sum()
    if n == 0:
        return np.nan, np.nan

    p = np.clip(expected_prob, 1e-12, 1 - 1e-12)
    pi_hat = np.clip(x / n, 1e-12, 1 - 1e-12)

    ll_h0 = (n - x) * _safe_log(1 - p) + x * _safe_log(p)
    ll_h1 = (n - x) * _safe_log(1 - pi_hat) + x * _safe_log(pi_hat)
    lr = max(0.0, -2 * (ll_h0 - ll_h1))
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen_test(violations: np.ndarray, expected_prob: float):
    """Christoffersen independence + conditional coverage test."""
    v = np.asarray(violations).astype(int)
    n = v.size
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, cur = int(v[i-1]), int(v[i])
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    d01 = n00 + n01
    d11 = n10 + n11
    pi01 = n01 / d01 if d01 > 0 else 0.0
    pi11 = n11 / d11 if d11 > 0 else 0.0
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total > 0 else 0.0

    ll_h0 = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_h1 = (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
             + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11))
    lr_ind = max(0.0, -2 * (ll_h0 - ll_h1))

    lr_pof, _ = kupiec_test(v, expected_prob)
    lr_cc = lr_pof + lr_ind

    return (float(lr_ind), float(1 - chi2.cdf(lr_ind, df=1)),
            float(lr_cc), float(1 - chi2.cdf(lr_cc, df=2)))


def cvm_with_bootstrap(real: np.ndarray, fake: np.ndarray, cfg: ExperimentConfig):
    """Cramér-von Mises statistic with stationary block bootstrap p-value."""
    real = real[np.isfinite(real)]
    fake = fake[np.isfinite(fake)]
    if real.size < 10 or fake.size < 10:
        return np.nan, np.nan

    rng = np.random.default_rng(123)
    if cfg.cvm_max_points and real.size > cfg.cvm_max_points:
        real = real[rng.choice(real.size, cfg.cvm_max_points, replace=False)]
    if cfg.cvm_max_points and fake.size > cfg.cvm_max_points:
        fake = fake[rng.choice(fake.size, cfg.cvm_max_points, replace=False)]

    obs = float(stats.cramervonmises_2samp(real, fake).statistic)
    pooled = np.concatenate([real, fake])

    boot_stats = np.empty(cfg.cvm_bootstrap_iterations)
    p_new = 1.0 / max(cfg.cvm_block_length, 1)

    for b in range(cfg.cvm_bootstrap_iterations):
        # Stationary block bootstrap
        boot_r = np.empty(real.size)
        boot_f = np.empty(fake.size)
        for arr in [boot_r, boot_f]:
            idx = int(rng.integers(0, pooled.size))
            for t in range(arr.size):
                if t == 0 or rng.random() < p_new:
                    idx = int(rng.integers(0, pooled.size))
                else:
                    idx = (idx + 1) % pooled.size
                arr[t] = pooled[idx]
        boot_stats[b] = float(stats.cramervonmises_2samp(boot_r, boot_f).statistic)

    pval = (1 + np.sum(boot_stats >= obs)) / (cfg.cvm_bootstrap_iterations + 1)
    return obs, float(pval)


def compute_signature_mmd(real_paths: np.ndarray, fake_paths: np.ndarray,
                          cfg: ExperimentConfig) -> float:
    """Signature-RBF MMD between real and fake path sets."""
    if not HAS_SIGNATORY and not HAS_IISIGNATURE:
        return np.nan

    n = min(real_paths.shape[0], fake_paths.shape[0], cfg.mmd_max_paths)
    if n < 4:
        return np.nan

    rng = np.random.default_rng(1234)
    idx_r = rng.choice(real_paths.shape[0], n, replace=False)
    idx_f = rng.choice(fake_paths.shape[0], n, replace=False)

    mmd_device = cfg.device
    real_t = torch.tensor(real_paths[idx_r], dtype=torch.float32, device=mmd_device).unsqueeze(-1)
    fake_t = torch.tensor(fake_paths[idx_f], dtype=torch.float32, device=mmd_device).unsqueeze(-1)

    augs = build_augmentation_pipeline(cfg)
    with torch.no_grad():
        x = augment_and_compute_signature(real_t, augs, cfg.mmd_sig_depth).float()
        y = augment_and_compute_signature(fake_t, augs, cfg.mmd_sig_depth).float()

    # Median heuristic bandwidth
    z = torch.cat([x, y])
    z2 = (z * z).sum(1, keepdim=True)
    d2 = torch.clamp(z2 + z2.T - 2 * (z @ z.T), min=0.0)
    triu_mask = torch.triu(torch.ones_like(d2), diagonal=1).bool()
    sigma2 = torch.median(d2[triu_mask])
    sigma2 = sigma2 if sigma2 > 1e-12 else torch.tensor(1.0, device=z.device)

    def rbf(a, b):
        a2 = (a * a).sum(1, keepdim=True)
        b2 = (b * b).sum(1, keepdim=True).T
        return torch.exp(-torch.clamp(a2 + b2 - 2 * a @ b.T, min=0.0) / (2 * sigma2))

    k_xx = rbf(x, x)
    k_yy = rbf(y, y)
    k_xy = rbf(x, y)
    m, n2 = x.shape[0], y.shape[0]

    mmd2 = ((k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
            + (k_yy.sum() - k_yy.diag().sum()) / (n2 * (n2 - 1))
            - 2 * k_xy.mean())

    return float(max(0.0, mmd2.item()))


def compute_metrics(real_paths: np.ndarray, fake_paths: np.ndarray,
                    cfg: ExperimentConfig) -> Dict[str, float]:
    """Compute all evaluation metrics between real and fake return paths."""
    real_flat = real_paths.reshape(-1)
    fake_flat = fake_paths.reshape(-1)
    m = {}

    # --- Distribution moments ---
    m["mean_diff"] = abs(real_flat.mean() - fake_flat.mean())
    m["std_diff"] = abs(real_flat.std() - fake_flat.std())
    m["skew_diff"] = abs(stats.skew(real_flat) - stats.skew(fake_flat))
    m["kurtosis_diff"] = abs(stats.kurtosis(real_flat) - stats.kurtosis(fake_flat))

    # --- KS and CvM ---
    m["ks_stat"] = float(stats.ks_2samp(real_flat, fake_flat).statistic)
    cvm_val, cvm_p = cvm_with_bootstrap(real_flat, fake_flat, cfg)
    m["cvm_stat"] = cvm_val
    m["cvm_pvalue_sbb"] = cvm_p

    # --- ACF ---
    real_acf = acf(real_flat, nlags=cfg.acf_max_lag, fft=False)
    fake_acf = acf(fake_flat, nlags=cfg.acf_max_lag, fft=False)
    m["acf_mean_abs_diff"] = float(np.mean(np.abs(real_acf[1:] - fake_acf[1:])))

    # --- VaR and ES at each level ---
    var_rel_errors = []
    es_rel_errors = []
    extreme_errors = []

    for level in cfg.var_levels:
        key = int(level * 100)
        tail_prob = 1.0 - level

        real_var = np.percentile(real_flat, tail_prob * 100)
        fake_var = np.percentile(fake_flat, tail_prob * 100)

        var_abs = abs(real_var - fake_var)
        var_rel = var_abs / max(abs(real_var), 1e-12) * 100

        m[f"var_real_{key}"] = real_var
        m[f"var_fake_{key}"] = fake_var
        m[f"var_abs_err_{key}"] = var_abs
        m[f"var_rel_err_{key}_pct"] = var_rel
        var_rel_errors.append(var_rel)

        # Pinball loss
        residual = real_flat - fake_var
        pinball = np.where(residual >= 0, tail_prob * residual,
                           (tail_prob - 1) * residual).mean()
        m[f"pinball_loss_{key}"] = pinball

        # Expected Shortfall
        real_tail = real_flat[real_flat <= real_var]
        fake_tail = fake_flat[fake_flat <= fake_var]
        real_es = real_tail.mean() if real_tail.size > 0 else real_var
        fake_es = fake_tail.mean() if fake_tail.size > 0 else fake_var

        es_abs = abs(real_es - fake_es)
        es_rel = es_abs / (1 + abs(real_es)) * 100

        m[f"es_real_{key}"] = real_es
        m[f"es_fake_{key}"] = fake_es
        m[f"es_abs_err_{key}"] = es_abs
        m[f"es_rel_err_{key}_pct"] = es_rel
        es_rel_errors.append(es_rel)

        # VaR backtests
        violations = (real_flat < fake_var).astype(int)
        lr_pof, p_pof = kupiec_test(violations, tail_prob)
        lr_ind, p_ind, lr_cc, p_cc = christoffersen_test(violations, tail_prob)

        m[f"kupiec_lr_{key}"] = lr_pof
        m[f"kupiec_pvalue_{key}"] = p_pof
        m[f"chr_ind_lr_{key}"] = lr_ind
        m[f"chr_ind_pvalue_{key}"] = p_ind
        m[f"chr_cc_lr_{key}"] = lr_cc
        m[f"chr_cc_pvalue_{key}"] = p_cc

    # Extreme event frequency (sigma-based)
    # Same approach as EnhancedTailLoss: threshold = sigma_t × data_std
    # This measures "fraction of returns beyond k standard deviations"
    # Both training loss and evaluation use this consistently.
    for sigma_t in cfg.extreme_sigma_thresholds:
        key = f"{sigma_t:.1f}s"
        real_std = max(real_flat.std(), 1e-12)
        threshold = sigma_t * real_std

        real_freq = (np.abs(real_flat) > threshold).mean() * 100
        fake_freq = (np.abs(fake_flat) > threshold).mean() * 100
        abs_err = abs(real_freq - fake_freq)
        rel_err = abs_err / (1 + real_freq)

        m[f"extreme_real_{key}_pct"] = real_freq
        m[f"extreme_fake_{key}_pct"] = fake_freq
        m[f"extreme_abs_err_{key}_pct"] = abs_err
        m[f"extreme_rel_err_{key}"] = rel_err
        extreme_errors.append(rel_err)

    # --- Aggregate scores ---
    m["var_rel_err_mean_pct"] = np.mean(var_rel_errors)
    m["es_rel_err_mean_pct"] = np.mean(es_rel_errors)
    m["extreme_rel_err_mean"] = np.mean(extreme_errors)
    m["primary_tail_score"] = 0.5 * m["var_rel_err_mean_pct"] + 0.5 * m["extreme_rel_err_mean"]

    # Signature MMD
    m["mmd_signature_rbf"] = compute_signature_mmd(real_paths, fake_paths, cfg)

    # --- Economic evaluation metrics ---
    # These translate statistical tail accuracy into dollar-equivalent terms.

    # 1. Regulatory Capital Ratio (Basel III inspired)
    # Under Basel III/IV, required capital ~ multiplier × ES_99.
    # We compute: |ES_fake_99 / ES_real_99|. Closer to 1.0 = better calibrated.
    # Ratio > 1 means model overestimates risk (excess capital), < 1 means underestimates.
    real_es99 = m.get("es_real_99", None)
    fake_es99 = m.get("es_fake_99", None)
    if real_es99 is not None and fake_es99 is not None and abs(real_es99) > 1e-12:
        m["capital_ratio_99"] = fake_es99 / real_es99  # Should be ~1.0
        m["capital_ratio_err_99"] = abs(1.0 - m["capital_ratio_99"]) * 100  # % deviation from ideal
    else:
        m["capital_ratio_99"] = np.nan
        m["capital_ratio_err_99"] = np.nan

    # Also at 95% level
    real_es95 = m.get("es_real_95", None)
    fake_es95 = m.get("es_fake_95", None)
    if real_es95 is not None and fake_es95 is not None and abs(real_es95) > 1e-12:
        m["capital_ratio_95"] = fake_es95 / real_es95
        m["capital_ratio_err_95"] = abs(1.0 - m["capital_ratio_95"]) * 100
    else:
        m["capital_ratio_95"] = np.nan
        m["capital_ratio_err_95"] = np.nan

    # 2. VaR Violation Cost (Expected Exceedance)
    # When actual loss exceeds predicted VaR, the exceedance is the "cost".
    # Lower = better (fewer/smaller violations).
    for level in cfg.var_levels:
        key = int(level * 100)
        fake_var = m.get(f"var_fake_{key}", None)
        if fake_var is not None:
            violations_mask = real_flat < fake_var  # Losses worse than VaR
            if violations_mask.any():
                exceedances = fake_var - real_flat[violations_mask]  # Positive = how much worse
                m[f"var_violation_cost_{key}"] = float(exceedances.mean())
                m[f"var_violation_count_{key}"] = int(violations_mask.sum())
                m[f"var_violation_rate_{key}"] = float(violations_mask.mean()) * 100
            else:
                m[f"var_violation_cost_{key}"] = 0.0
                m[f"var_violation_count_{key}"] = 0
                m[f"var_violation_rate_{key}"] = 0.0

    # 3. Capital Efficiency Score
    # Combines accuracy and conservatism: penalise both under- and over-estimation,
    # but penalise under-estimation more (regulatory penalty).
    # Score = |capital_ratio - 1| × (3.0 if ratio < 1 else 1.0)
    cr99 = m.get("capital_ratio_99", np.nan)
    if not np.isnan(cr99):
        under_penalty = 3.0 if cr99 < 1.0 else 1.0  # Regulatory asymmetry
        m["capital_efficiency_99"] = abs(1.0 - cr99) * under_penalty * 100
    else:
        m["capital_efficiency_99"] = np.nan

    return m


# ============================================================================
# 7. STATISTICAL TESTS
# ============================================================================


def friedman_tests(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Run Friedman test across models for each metric."""
    rows = []
    task_cols = ["dataset", "split_id", "seed"]
    for metric in metrics:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index=task_cols, columns="model", values=metric, aggfunc="mean")
        pivot = pivot.dropna(how="any")
        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            continue
        stat, pval = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        rows.append({
            "metric": metric,
            "n_tasks": pivot.shape[0],
            "n_models": pivot.shape[1],
            "chi2": float(stat),
            "p_value": float(pval),
            "significant": pval < 0.05,
        })
    return pd.DataFrame(rows)


def diebold_mariano(df: pd.DataFrame, metric: str, ref_model: str,
                    comp_models: List[str], hac_lags: int = 3) -> pd.DataFrame:
    """Diebold-Mariano test comparing reference model to each competitor."""
    rows = []
    task_cols = ["dataset", "split_id", "seed"]
    if metric not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(index=task_cols, columns="model", values=metric, aggfunc="mean")

    for comp in comp_models:
        if ref_model not in pivot.columns or comp not in pivot.columns:
            continue
        pair = pivot[[ref_model, comp]].dropna(how="any")
        if pair.shape[0] < 3:
            continue

        d = pair[ref_model].values - pair[comp].values
        mean_d = d.mean()
        t = d.size

        # HAC long-run variance (Bartlett kernel)
        d_c = d - mean_d
        lrv = float(np.dot(d_c, d_c) / t)
        for lag in range(1, min(hac_lags, t - 1) + 1):
            w = 1 - lag / (hac_lags + 1)
            lrv += 2 * w * float(np.dot(d_c[lag:], d_c[:-lag]) / t)

        if lrv <= 1e-14:
            continue

        dm_stat = mean_d / np.sqrt(lrv / t)
        pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        rows.append({
            "metric": metric,
            "reference": ref_model,
            "competitor": comp,
            "n_tasks": t,
            "dm_stat": float(dm_stat),
            "p_value": float(pval),
            "mean_diff": float(mean_d),
            "ref_better": mean_d < 0,
            "significant": pval < 0.05,
        })

    return pd.DataFrame(rows)


def bootstrap_ci(df: pd.DataFrame, metrics: List[str],
                 n_boot: int = 2000) -> pd.DataFrame:
    """Bootstrap 95% confidence intervals for model means."""
    rows = []
    task_cols = ["dataset", "split_id", "seed"]
    rng = np.random.default_rng(123)

    for metric in metrics:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index=task_cols, columns="model", values=metric, aggfunc="mean")
        pivot = pivot.dropna(how="any")
        if pivot.shape[0] < 2:
            continue

        for model in pivot.columns:
            vals = pivot[model].values
            boot = np.empty(n_boot)
            for b in range(n_boot):
                boot[b] = vals[rng.integers(0, len(vals), len(vals))].mean()

            rows.append({
                "metric": metric,
                "model": model,
                "mean": vals.mean(),
                "ci_low": np.percentile(boot, 2.5),
                "ci_high": np.percentile(boot, 97.5),
                "n_tasks": len(vals),
            })

    return pd.DataFrame(rows)


# ============================================================================
# 8. MODEL DISPATCH AND MAIN BENCHMARK LOOP
# ============================================================================


def run_model(model_name: str, bundle: dict, seed: int,
              cfg: ExperimentConfig, **kwargs) -> dict:
    """Dispatch to the correct model runner."""
    if model_name == "sigcwgan":
        tlw = kwargs.get("tail_loss_weight", cfg.default_tail_loss_weight)
        over_pen = kwargs.get("over_penalty", cfg.tail_over_penalty)
        under_pen = kwargs.get("under_penalty", cfg.tail_under_penalty)
        return run_sigcwgan(bundle, seed, cfg, tail_loss_weight=tlw,
                            over_penalty=over_pen, under_penalty=under_pen)
    elif model_name == "garch_ar1_gjr_skewt":
        return run_garch(bundle, seed, cfg)
    elif model_name == "filtered_historical_simulation":
        return run_fhs(bundle, seed, cfg)
    elif model_name == "timegan":
        return run_timegan(bundle, seed, cfg)
    elif model_name == "neural_sde":
        return run_neural_sde(bundle, seed, cfg)
    elif model_name == "tail_gan":
        return run_tail_gan(bundle, seed, cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_benchmark(cfg: ExperimentConfig, dataset_csvs: List[str]):
    """
    Main benchmark loop.

    Runs all models × datasets × splits × seeds, computes metrics,
    and exports results.
    """
    all_results = []
    run_artifacts = {}  # For t-SNE plotting later

    # Build task list
    # Each task: (csv_path, ds_name, ratio, seed, model, tlw, over_pen, under_pen)
    tasks = []
    for csv_path in dataset_csvs:
        ds_name = Path(csv_path).stem
        for ratio in cfg.train_ratios:
            for seed in cfg.seeds:
                for model in cfg.models:
                    if model == "sigcwgan":
                        # Sensitivity analysis: run all tail_loss_weights
                        for tlw in cfg.tail_loss_weights:
                            tasks.append((csv_path, ds_name, ratio, seed, model, tlw, None, None))
                        # Penalty ratio ablation: fixed tlw, varying (over, under)
                        for (over_p, under_p) in cfg.penalty_ablation_configs:
                            tasks.append((csv_path, ds_name, ratio, seed, model,
                                          cfg.penalty_ablation_tlw, over_p, under_p))
                    else:
                        tasks.append((csv_path, ds_name, ratio, seed, model, None, None, None))

    total_tasks = len(tasks)
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {total_tasks} tasks")
    print(f"  Datasets: {len(dataset_csvs)}")
    print(f"  Splits: {cfg.train_ratios}")
    print(f"  Seeds: {cfg.seeds}")
    print(f"  Models: {cfg.models}")
    print(f"  SigCWGAN tail weights: {cfg.tail_loss_weights}")
    print(f"  Penalty ablation configs: {cfg.penalty_ablation_configs}")
    print(f"  Device: {cfg.device}")
    print(f"{'='*60}\n")

    # Create per-run output directory
    base_dir = os.path.dirname(cfg.artifact_prefix) or "."
    per_run_dir = os.path.join(base_dir, "per_run")
    os.makedirs(per_run_dir, exist_ok=True)

    for i, (csv_path, ds_name, ratio, seed, model, tlw, over_pen, under_pen) in enumerate(tasks):
        split_id = f"r{ratio:.1f}"
        # Build model label
        if tlw is None:
            model_label = model
        elif over_pen is not None:
            model_label = f"{model}_tlw{tlw}_pen{int(over_pen)}_{int(under_pen)}"
        else:
            model_label = f"{model}_tlw{tlw}"

        print(f"\n[{i+1}/{total_tasks}] {ds_name} | {split_id} | seed={seed} | {model_label}")

        try:
            bundle = load_dataset(csv_path, ratio, cfg)
            result = run_model(model, bundle, seed, cfg, tail_loss_weight=tlw,
                               over_penalty=over_pen, under_penalty=under_pen)

            metrics = compute_metrics(result["real_paths"], result["fake_paths"], cfg)
            metrics["dataset"] = ds_name
            metrics["split_id"] = split_id
            metrics["seed"] = seed
            metrics["model"] = model_label
            metrics["base_model"] = model
            metrics["tail_loss_weight"] = tlw if tlw is not None else np.nan
            metrics["over_penalty"] = over_pen if over_pen is not None else np.nan
            metrics["under_penalty"] = under_pen if under_pen is not None else np.nan
            metrics["n_eval"] = result["n_eval"]
            metrics["fit_runtime_sec"] = result["runtime_sec"]
            metrics["final_loss"] = result.get("final_loss", np.nan)
            metrics["status"] = "ok"

            all_results.append(metrics)

            # Store artifacts for plotting
            key = (model_label, ds_name, split_id, seed)
            run_artifacts[key] = {
                "fake_paths": result["fake_paths"],
                "real_paths": result["real_paths"],
            }

            # === Save per-run outputs immediately ===
            run_subdir = os.path.join(per_run_dir, ds_name, f"{model_label}_{split_id}_seed{seed}")
            os.makedirs(run_subdir, exist_ok=True)
            _save_per_run_outputs(
                run_subdir, metrics, result["real_paths"], result["fake_paths"],
                ds_name, model_label, split_id, seed, cfg,
            )

            print(f"  -> primary_tail_score={metrics['primary_tail_score']:.4f} | "
                  f"runtime={result['runtime_sec']:.1f}s")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            all_results.append({
                "dataset": ds_name,
                "split_id": split_id,
                "seed": seed,
                "model": model_label,
                "base_model": model,
                "tail_loss_weight": tlw if tlw is not None else np.nan,
                "over_penalty": over_pen if over_pen is not None else np.nan,
                "under_penalty": under_pen if under_pen is not None else np.nan,
                "status": f"error: {e}",
            })

    return pd.DataFrame(all_results), run_artifacts


# ============================================================================
# 9. EXPORT AND REPORTING
# ============================================================================


def _save_per_run_outputs(
    run_dir: str,
    metrics: dict,
    real_paths: np.ndarray,
    fake_paths: np.ndarray,
    ds_name: str,
    model_label: str,
    split_id: str,
    seed: int,
    cfg: ExperimentConfig,
):
    """
    Save outputs for a SINGLE run (one dataset × model × split × seed).

    Creates in run_dir/:
      - metrics.csv          — all evaluation metrics for this run
      - tsne.png             — t-SNE of real vs generated paths
      - qq_plot.png          — QQ plot (real vs generated quantiles)
      - distribution.png     — histogram overlay (real vs generated)
      - acf.png              — ACF comparison (returns + squared returns)
      - rolling_vol.png      — rolling volatility comparison
      - var_backtest.png     — VaR 99% backtest with violations
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 1. Save metrics CSV ---
    pd.DataFrame([metrics]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    real_flat = real_paths.reshape(-1)
    fake_flat = fake_paths.reshape(-1)
    title_tag = f"{ds_name} | {model_label} | {split_id} | seed={seed}"

    # --- 2. t-SNE for this run ---
    if real_paths.shape[0] >= 10 and fake_paths.shape[0] >= 10:
        max_per = min(300, real_paths.shape[0], fake_paths.shape[0])
        rng = np.random.default_rng(42)
        idx_r = rng.choice(real_paths.shape[0], max_per, replace=False)
        idx_f = rng.choice(fake_paths.shape[0], max_per, replace=False)
        X = np.vstack([real_paths[idx_r], fake_paths[idx_f]])
        labels = ["Real"] * max_per + ["Generated"] * max_per
        if X.shape[0] >= 20:
            perp = min(30, max(5, X.shape[0] // 15))
            emb = TSNE(n_components=2, perplexity=perp, random_state=42,
                       init="pca", learning_rate="auto").fit_transform(X)
            fig, ax = plt.subplots(figsize=(7, 5))
            mask_r = np.array(labels) == "Real"
            ax.scatter(emb[mask_r, 0], emb[mask_r, 1], s=12, alpha=0.5,
                       c="steelblue", marker="o", label="Real")
            ax.scatter(emb[~mask_r, 0], emb[~mask_r, 1], s=12, alpha=0.5,
                       c="coral", marker="x", label="Generated")
            ax.set_title(f"t-SNE: {title_tag}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
            fig.savefig(os.path.join(run_dir, "tsne.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # --- 3. QQ Plot ---
    n_pts = min(500, len(real_flat), len(fake_flat))
    if n_pts >= 10:
        probs = np.linspace(0.5 / n_pts, 1 - 0.5 / n_pts, n_pts)
        real_q = np.quantile(real_flat, probs)
        fake_q = np.quantile(fake_flat, probs)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(real_q, fake_q, s=6, alpha=0.5, c="steelblue")
        lo, hi = min(real_q.min(), fake_q.min()), max(real_q.max(), fake_q.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Perfect match")
        ax.set_xlabel("Real quantiles")
        ax.set_ylabel("Generated quantiles")
        ax.set_title(f"QQ: {title_tag}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.savefig(os.path.join(run_dir, "qq_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- 4. Distribution overlay ---
    fig, ax = plt.subplots(figsize=(7, 4))
    clip_lo = np.percentile(real_flat, 0.5)
    clip_hi = np.percentile(real_flat, 99.5)
    bins = np.linspace(clip_lo, clip_hi, 80)
    ax.hist(real_flat, bins=bins, density=True, alpha=0.5, color="steelblue", label="Real")
    ax.hist(fake_flat, bins=bins, density=True, alpha=0.5, color="coral", label="Generated")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution: {title_tag}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(run_dir, "distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 5. ACF comparison ---
    n_lags = min(cfg.acf_max_lag, 30)
    if len(real_flat) > n_lags + 5 and len(fake_flat) > n_lags + 5:
        real_acf_vals = acf(real_flat, nlags=n_lags, fft=False)
        fake_acf_vals = acf(fake_flat, nlags=n_lags, fft=False)
        real_sq_acf = acf(real_flat ** 2, nlags=n_lags, fft=False)
        fake_sq_acf = acf(fake_flat ** 2, nlags=n_lags, fft=False)
        lags = np.arange(1, n_lags + 1)
        conf = 1.96 / np.sqrt(len(real_flat))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.bar(lags - 0.15, real_acf_vals[1:], width=0.3, alpha=0.7, color="steelblue", label="Real")
        ax1.bar(lags + 0.15, fake_acf_vals[1:], width=0.3, alpha=0.7, color="coral", label="Generated")
        ax1.axhline(conf, ls="--", c="gray", lw=0.8)
        ax1.axhline(-conf, ls="--", c="gray", lw=0.8)
        ax1.set_title("Returns ACF", fontsize=9)
        ax1.set_xlabel("Lag")
        ax1.legend(fontsize=6)
        ax1.grid(alpha=0.2)

        ax2.bar(lags - 0.15, real_sq_acf[1:], width=0.3, alpha=0.7, color="steelblue", label="Real")
        ax2.bar(lags + 0.15, fake_sq_acf[1:], width=0.3, alpha=0.7, color="coral", label="Generated")
        ax2.axhline(conf, ls="--", c="gray", lw=0.8)
        ax2.axhline(-conf, ls="--", c="gray", lw=0.8)
        ax2.set_title("Squared Returns ACF (Vol Clustering)", fontsize=9)
        ax2.set_xlabel("Lag")
        ax2.legend(fontsize=6)
        ax2.grid(alpha=0.2)

        fig.suptitle(f"ACF: {title_tag}", fontsize=10, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(run_dir, "acf.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- 6. Rolling volatility (uses flattened series for enough length) ---
    window = min(24, max(3, len(real_flat) // 10))
    if len(real_flat) > window + 5 and len(fake_flat) > window + 5:
        real_vol = pd.Series(real_flat).rolling(window).std().dropna()
        fake_vol = pd.Series(fake_flat).rolling(window).std().dropna()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(real_vol.values, lw=1, alpha=0.8, color="steelblue", label="Real")
        ax.plot(fake_vol.values[:len(real_vol)], lw=1, alpha=0.8, color="coral", label="Generated")
        ax.set_xlabel("Time step")
        ax.set_ylabel(f"Rolling σ ({window}-step)")
        ax.set_title(f"Rolling Vol: {title_tag}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.savefig(os.path.join(run_dir, "rolling_vol.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- 7. VaR backtest ---
    level = 0.99
    tail_prob = 1.0 - level
    var_estimate = np.percentile(fake_flat, tail_prob * 100)
    violations = real_flat < var_estimate
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(real_flat, lw=0.5, alpha=0.7, color="steelblue", label="Real returns")
    ax.axhline(var_estimate, color="red", ls="--", lw=1.2,
               label=f"VaR 99% = {var_estimate:.4f}")
    viol_idx = np.where(violations)[0]
    if len(viol_idx) > 0:
        ax.scatter(viol_idx, real_flat[viol_idx], c="red", s=10, zorder=5, marker="v",
                   label=f"Violations ({violations.sum()}/{len(real_flat)})")
    actual_pct = violations.mean() * 100
    ax.set_title(f"VaR 99% Backtest: {title_tag} (exp 1.0%, actual {actual_pct:.1f}%)", fontsize=9)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Log return")
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(run_dir, "var_backtest.png"), dpi=150, bbox_inches="tight")
    plt.close()

    n_files = len([f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f))])
    print(f"    [per-run] saved {n_files} outputs to {run_dir}/")


def _export_per_dataset(df: pd.DataFrame, run_artifacts: dict, cfg: ExperimentConfig):
    """
    Save per-dataset aggregated results: CSV + charts.

    Creates per_dataset/{dataset_name}/ with:
      - summary.csv           — all runs for this dataset
      - model_comparison.csv  — model means for this dataset
      - overview.png          — bar chart of model scores
      - tsne.png              — t-SNE with all models overlaid
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    base_dir = os.path.dirname(cfg.artifact_prefix) or "."
    ds_dir_root = os.path.join(base_dir, "per_dataset")
    os.makedirs(ds_dir_root, exist_ok=True)

    for ds_name, ds_df in ok.groupby("dataset"):
        ds_dir = os.path.join(ds_dir_root, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        # --- CSV: all runs for this dataset ---
        ds_df.to_csv(os.path.join(ds_dir, "summary.csv"), index=False)

        # --- CSV: model comparison means ---
        metric_cols = [c for c in [
            "primary_tail_score", "var_rel_err_mean_pct", "es_rel_err_mean_pct",
            "extreme_rel_err_mean", "acf_mean_abs_diff", "ks_stat", "cvm_stat",
            "mmd_signature_rbf", "fit_runtime_sec",
        ] if c in ds_df.columns]
        model_comp = ds_df.groupby("model")[metric_cols].agg(["mean", "std"])
        model_comp.to_csv(os.path.join(ds_dir, "model_comparison.csv"))

        # --- Bar chart: model scores for this dataset ---
        fig, ax = plt.subplots(figsize=(10, 5))
        means = ds_df.groupby("model")["primary_tail_score"].mean().sort_values()
        stds = ds_df.groupby("model")["primary_tail_score"].std().fillna(0).reindex(means.index)
        x = np.arange(len(means))
        ax.bar(x, means.values, yerr=stds.values, capsize=5, alpha=0.85,
               color=plt.cm.Set2(np.linspace(0, 1, len(means))))
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Primary Tail Score (lower = better)")
        ax.set_title(f"Model Comparison: {ds_name}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(os.path.join(ds_dir, "overview.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # --- t-SNE: all models on this dataset ---
        ds_artifacts = {k: v for k, v in run_artifacts.items() if k[1] == ds_name}
        if ds_artifacts:
            model_samples = defaultdict(list)
            real_samples = []
            for key, art in ds_artifacts.items():
                mname = key[0]
                if art["fake_paths"].shape[0] > 0:
                    model_samples[mname].append(art["fake_paths"])
                if art["real_paths"].shape[0] > 0:
                    real_samples.append(art["real_paths"])

            if model_samples:
                max_per = 300
                rng_t = np.random.default_rng(42)
                points, labels = [], []

                if real_samples:
                    cat = np.concatenate(real_samples)
                    n_pts = min(max_per, cat.shape[0])
                    points.append(cat[rng_t.choice(cat.shape[0], n_pts, replace=False)])
                    labels.extend(["Real"] * n_pts)

                for mname, arrs in sorted(model_samples.items()):
                    cat = np.concatenate(arrs)
                    n_pts = min(max_per, cat.shape[0])
                    points.append(cat[rng_t.choice(cat.shape[0], n_pts, replace=False)])
                    labels.extend([mname] * n_pts)

                X = np.concatenate(points)
                if X.shape[0] >= 20:
                    perp = min(30, max(5, X.shape[0] // 15))
                    emb = TSNE(n_components=2, perplexity=perp, random_state=42,
                               init="pca", learning_rate="auto").fit_transform(X)
                    fig = plt.figure(figsize=(9, 7))
                    uniq = sorted(set(labels))
                    cmap = plt.cm.get_cmap("tab10", len(uniq))
                    for idx, lbl in enumerate(uniq):
                        mask = np.array(labels) == lbl
                        marker = "o" if lbl == "Real" else "x"
                        alpha = 0.5 if lbl == "Real" else 0.7
                        plt.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=alpha,
                                    c=[cmap(idx)], marker=marker, label=lbl)
                    plt.title(f"t-SNE: {ds_name}", fontsize=12, fontweight="bold")
                    plt.xlabel("t-SNE 1")
                    plt.ylabel("t-SNE 2")
                    plt.legend(loc="best", fontsize=7)
                    plt.grid(alpha=0.2)
                    fig.savefig(os.path.join(ds_dir, "tsne.png"), dpi=200, bbox_inches="tight")
                    plt.close()

        print(f"  [per-dataset] saved outputs to {ds_dir}/")


def export_results(df: pd.DataFrame, run_artifacts: dict, cfg: ExperimentConfig):
    """Export all result tables, plots, and optionally LaTeX."""
    prefix = cfg.artifact_prefix
    ok = df[df["status"] == "ok"].copy()

    if ok.empty:
        print("No successful runs to export.")
        return

    # --- 9.1 Raw results ---
    df.to_csv(f"{prefix}_all_runs.csv", index=False)
    print(f"Saved: {prefix}_all_runs.csv")

    # --- 9.2 Model summary ---
    summary_metrics = [
        "primary_tail_score", "var_rel_err_mean_pct", "es_rel_err_mean_pct",
        "extreme_rel_err_mean", "acf_mean_abs_diff", "ks_stat", "cvm_stat",
        "mmd_signature_rbf", "fit_runtime_sec",
        "capital_ratio_err_99", "capital_efficiency_99",
        "var_violation_cost_99", "var_violation_rate_99",
    ]
    summary = ok.groupby("model")[
        [c for c in summary_metrics if c in ok.columns]
    ].agg(["mean", "std"])
    summary.to_csv(f"{prefix}_model_summary.csv")
    print(f"Saved: {prefix}_model_summary.csv")

    # --- 9.3 Dataset × Model breakdown ---
    ds_model = ok.groupby(["dataset", "model"])[
        [c for c in ["primary_tail_score", "var_rel_err_mean_pct", "es_rel_err_mean_pct",
                      "acf_mean_abs_diff", "ks_stat", "cvm_stat"] if c in ok.columns]
    ].mean()
    ds_model.to_csv(f"{prefix}_dataset_model.csv")
    print(f"Saved: {prefix}_dataset_model.csv")

    # --- 9.4 Friedman tests ---
    friedman_metrics = [c for c in [
        "primary_tail_score", "var_rel_err_mean_pct", "es_rel_err_mean_pct",
        "extreme_rel_err_mean", "acf_mean_abs_diff", "ks_stat", "cvm_stat",
        "mmd_signature_rbf",
    ] if c in ok.columns]
    ft = friedman_tests(ok, friedman_metrics)
    ft.to_csv(f"{prefix}_friedman.csv", index=False)
    print(f"Saved: {prefix}_friedman.csv")

    # --- 9.5 Diebold-Mariano tests ---
    dm_rows = []
    for metric in ["primary_tail_score", "var_rel_err_mean_pct"]:
        if metric not in ok.columns:
            continue
        # Compare each SigCWGAN variant against each baseline
        sigcwgan_models = [m for m in ok["model"].unique() if m.startswith("sigcwgan")]
        baseline_models = [m for m in ok["model"].unique() if not m.startswith("sigcwgan")]
        for sig_m in sigcwgan_models:
            dm = diebold_mariano(ok, metric, sig_m, baseline_models, cfg.dm_hac_lags)
            if not dm.empty:
                dm_rows.append(dm)
    if dm_rows:
        dm_all = pd.concat(dm_rows, ignore_index=True)
        dm_all.to_csv(f"{prefix}_diebold_mariano.csv", index=False)
        print(f"Saved: {prefix}_diebold_mariano.csv")

    # --- 9.6 Bootstrap CIs ---
    ci = bootstrap_ci(ok, friedman_metrics, cfg.bootstrap_ci_iterations)
    ci.to_csv(f"{prefix}_bootstrap_ci.csv", index=False)
    print(f"Saved: {prefix}_bootstrap_ci.csv")

    # --- 9.7 Sensitivity analysis table ---
    sig_runs = ok[ok["base_model"] == "sigcwgan"].copy()
    if not sig_runs.empty and "tail_loss_weight" in sig_runs.columns:
        sens = sig_runs.groupby("tail_loss_weight")[
            [c for c in ["primary_tail_score", "var_rel_err_mean_pct",
                          "es_rel_err_mean_pct", "extreme_rel_err_mean",
                          "acf_mean_abs_diff", "ks_stat", "cvm_stat"] if c in sig_runs.columns]
        ].agg(["mean", "std"])
        sens.to_csv(f"{prefix}_sensitivity.csv")
        print(f"Saved: {prefix}_sensitivity.csv")

    # --- 9.8 Penalty ablation table ---
    pen_runs = ok[ok["model"].str.contains("_pen", na=False)].copy()
    if not pen_runs.empty:
        pen_metrics = [c for c in ["primary_tail_score", "var_rel_err_mean_pct",
                                    "es_rel_err_mean_pct", "ks_stat", "cvm_stat",
                                    "capital_ratio_err_99", "capital_efficiency_99"]
                       if c in pen_runs.columns]
        pen_summary = pen_runs.groupby("model")[pen_metrics].agg(["mean", "std"])
        pen_summary.to_csv(f"{prefix}_penalty_ablation.csv")
        print(f"Saved: {prefix}_penalty_ablation.csv")

    # --- 9.9 Economic evaluation summary ---
    econ_metrics = [c for c in ["capital_ratio_99", "capital_ratio_err_99",
                                 "capital_ratio_95", "capital_ratio_err_95",
                                 "capital_efficiency_99",
                                 "var_violation_cost_99", "var_violation_rate_99",
                                 "var_violation_cost_95", "var_violation_rate_95"]
                    if c in ok.columns]
    if econ_metrics:
        econ = ok.groupby("model")[econ_metrics].agg(["mean", "std"])
        econ.to_csv(f"{prefix}_economic_evaluation.csv")
        print(f"Saved: {prefix}_economic_evaluation.csv")

    # --- 9.10 Plots (aggregate level) ---
    _make_plots(ok, run_artifacts, cfg)

    # --- 9.11 Per-dataset summaries (CSV + charts + t-SNE per dataset) ---
    _export_per_dataset(df, run_artifacts, cfg)

    # --- 9.12 LaTeX tables ---
    if cfg.export_latex:
        _export_latex_tables(ok, cfg)


def _make_plots(ok: pd.DataFrame, run_artifacts: dict, cfg: ExperimentConfig):
    """Generate publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = cfg.artifact_prefix

    # --- Plot 1: Overall tail score by model ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Bar chart
    ax = axes[0, 0]
    overall = ok.groupby("model")["primary_tail_score"].agg(["mean", "std"]).sort_values("mean")
    x = np.arange(len(overall))
    ax.bar(x, overall["mean"], yerr=overall["std"].fillna(0), capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(overall.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Primary Tail Score (lower = better)")
    ax.set_title("Overall Tail Score by Model")
    ax.grid(axis="y", alpha=0.25)

    # Heatmap: dataset × model
    ax = axes[0, 1]
    heat = ok.groupby(["dataset", "model"])["primary_tail_score"].mean().unstack("model")
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title("Tail Score by Dataset × Model")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Boxplot
    ax = axes[1, 0]
    models_sorted = overall.index.tolist()
    box_data = [ok.loc[ok["model"] == m, "primary_tail_score"].values for m in models_sorted]
    ax.boxplot(box_data, labels=models_sorted, patch_artist=True)
    ax.set_xticklabels(models_sorted, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Primary Tail Score")
    ax.set_title("Score Distribution Across Runs")
    ax.grid(axis="y", alpha=0.25)

    # VaR vs Extreme error scatter
    ax = axes[1, 1]
    scatter = ok.groupby("model")[["var_rel_err_mean_pct", "extreme_rel_err_mean"]].mean()
    for model_name, row in scatter.iterrows():
        ax.scatter(row["var_rel_err_mean_pct"], row["extreme_rel_err_mean"], s=120)
        ax.annotate(f" {model_name}", (row["var_rel_err_mean_pct"], row["extreme_rel_err_mean"]),
                    fontsize=7)
    ax.set_xlabel("VaR Relative Error (%)")
    ax.set_ylabel("Extreme Event Error")
    ax.set_title("VaR vs Extreme Error Trade-off")
    ax.grid(alpha=0.25)

    plt.suptitle("Tail-Aware SigCWGAN Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{prefix}_overview.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {prefix}_overview.png")
    plt.close()

    # --- Plot 2: Sensitivity analysis ---
    sig_runs = ok[ok["base_model"] == "sigcwgan"].copy()
    if not sig_runs.empty and "tail_loss_weight" in sig_runs.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        plot_idx = 0
        for i, metric in enumerate(["primary_tail_score", "var_rel_err_mean_pct", "ks_stat"]):
            if metric not in sig_runs.columns:
                continue
            ax = axes[plot_idx]
            plot_idx += 1
            sens = sig_runs.groupby("tail_loss_weight")[metric].agg(["mean", "std"])
            ax.errorbar(sens.index, sens["mean"], yerr=sens["std"], marker="o", capsize=5)
            ax.set_xlabel("Tail Loss Weight")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"Sensitivity: {metric}")
            ax.grid(alpha=0.25)

        plt.suptitle("Tail Loss Weight Sensitivity", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(f"{prefix}_sensitivity.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {prefix}_sensitivity.png")
        plt.close()

    # --- Plot 3: t-SNE ---
    if run_artifacts:
        _make_tsne_plot(run_artifacts, cfg)

    # --- Plot 4: QQ plots (real vs normal, fake vs normal) ---
    if run_artifacts:
        _make_qq_plots(run_artifacts, cfg)

    # --- Plot 5: Return distribution overlay (real vs fake histograms) ---
    if run_artifacts:
        _make_distribution_overlay(run_artifacts, cfg)

    # --- Plot 6: ACF comparison (real vs fake with confidence bands) ---
    if run_artifacts:
        _make_acf_comparison(run_artifacts, cfg)

    # --- Plot 7: Rolling volatility (real vs fake) ---
    if run_artifacts:
        _make_rolling_volatility(run_artifacts, cfg)

    # --- Plot 8: VaR backtest timeline (returns with VaR line + violations) ---
    if run_artifacts:
        _make_var_backtest_plot(run_artifacts, ok, cfg)

    # --- Plot 9: Model radar chart (multi-metric comparison) ---
    _make_radar_chart(ok, cfg)

    # --- Plot 10: ES comparison across models at multiple levels ---
    _make_es_comparison(ok, cfg)

    # --- Plot 11: PIT histogram (probability integral transform) ---
    if run_artifacts:
        _make_pit_histogram(run_artifacts, cfg)

    # --- Plot 12: Pareto frontier (tail score vs KS stat) ---
    _make_pareto_frontier(ok, cfg)

    # --- Plot 13: Penalty ablation bar chart ---
    _make_penalty_ablation_plot(ok, cfg)

    # --- Plot 14: Economic evaluation comparison ---
    _make_economic_plot(ok, cfg)


# ---------------------------------------------------------------------------
# CHART 12: Pareto Frontier (Tail Score vs Distributional Fidelity)
# ---------------------------------------------------------------------------

def _make_pareto_frontier(ok: pd.DataFrame, cfg: ExperimentConfig):
    """
    Plot the Pareto frontier: primary_tail_score vs ks_stat across
    SigCWGAN lambda values, with baselines as reference points.
    This reframes the distributional fidelity trade-off as a principled
    multi-objective design choice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = cfg.artifact_prefix

    if "primary_tail_score" not in ok.columns or "ks_stat" not in ok.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # SigCWGAN variants: connected line showing the frontier
    sig_runs = ok[ok["base_model"] == "sigcwgan"].copy()
    # Exclude penalty ablation runs (they have _pen in model label)
    sig_runs = sig_runs[~sig_runs["model"].str.contains("_pen", na=False)]

    if not sig_runs.empty and "tail_loss_weight" in sig_runs.columns:
        frontier = sig_runs.groupby("tail_loss_weight")[
            ["primary_tail_score", "ks_stat"]
        ].agg(["mean", "std"])

        xs = frontier[("primary_tail_score", "mean")].values
        ys = frontier[("ks_stat", "mean")].values
        x_err = frontier[("primary_tail_score", "std")].values
        y_err = frontier[("ks_stat", "std")].values
        labels = [f"$\\lambda$={w:.0f}" for w in frontier.index]

        # Sort by tail score for connected line
        order = np.argsort(xs)
        ax.errorbar(xs[order], ys[order], xerr=x_err[order], yerr=y_err[order],
                     fmt="o-", color="#2E75B6", markersize=10, linewidth=2,
                     capsize=5, label="SigCWGAN (Pareto frontier)", zorder=5)

        for i in range(len(labels)):
            ax.annotate(labels[i], (xs[i], ys[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=10, fontweight="bold", color="#2E75B6")

    # Baselines as reference points
    baseline_colors = {
        "garch_ar1_gjr_skewt": ("#2CA02C", "s", "GARCH"),
        "filtered_historical_simulation": ("#FF7F0E", "D", "FHS"),
        "timegan": ("#D62728", "^", "TimeGAN"),
        "neural_sde": ("#9467BD", "v", "Neural SDE"),
        "tail_gan": ("#8C564B", "P", "Tail-GAN"),
    }

    for model_name, (color, marker, label) in baseline_colors.items():
        m_runs = ok[ok["model"] == model_name]
        if m_runs.empty:
            continue
        mx = m_runs["primary_tail_score"].mean()
        my = m_runs["ks_stat"].mean()
        mx_e = m_runs["primary_tail_score"].std()
        my_e = m_runs["ks_stat"].std()
        ax.errorbar(mx, my, xerr=mx_e, yerr=my_e, fmt=marker, color=color,
                     markersize=12, capsize=4, label=label, zorder=4)

    ax.set_xlabel("Primary Tail Score (lower = better tail calibration)", fontsize=12)
    ax.set_ylabel("KS Statistic (lower = better distributional fidelity)", fontsize=12)
    ax.set_title("Pareto Frontier: Tail Calibration vs Distributional Fidelity",
                  fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)

    # Add annotation arrow
    ax.annotate("Better tail risk\n(lower score)", xy=(0.15, 0.5),
                xycoords="axes fraction", fontsize=9, color="gray",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="gray"),
                xytext=(0.3, 0.5))
    ax.annotate("Better distribution\n(lower KS)", xy=(0.5, 0.15),
                xycoords="axes fraction", fontsize=9, color="gray",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="gray"),
                xytext=(0.5, 0.3))

    plt.tight_layout()
    fig.savefig(f"{prefix}_pareto_frontier.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {prefix}_pareto_frontier.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 13: Penalty Ablation Comparison
# ---------------------------------------------------------------------------

def _make_penalty_ablation_plot(ok: pd.DataFrame, cfg: ExperimentConfig):
    """Bar chart comparing different penalty ratio configurations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = cfg.artifact_prefix

    pen_runs = ok[ok["model"].str.contains("_pen", na=False)].copy()
    # Also include the default config (tlw8 without pen override) for reference
    default_runs = ok[ok["model"] == f"sigcwgan_tlw{int(cfg.penalty_ablation_tlw)}"].copy()

    if pen_runs.empty:
        return

    all_ablation = pd.concat([default_runs, pen_runs], ignore_index=True)

    metrics_to_plot = ["primary_tail_score", "ks_stat", "capital_efficiency_99"]
    metrics_to_plot = [m for m in metrics_to_plot if m in all_ablation.columns]

    if not metrics_to_plot:
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        grp = all_ablation.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean")
        x = np.arange(len(grp))
        colors = ["#2E75B6" if "pen" not in idx else "#FF7F0E" for idx in grp.index]
        ax.bar(x, grp["mean"], yerr=grp["std"].fillna(0), capsize=4, alpha=0.85, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(grp.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Penalty Ablation: {metric}")
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("Penalty Ratio Ablation Study", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{prefix}_penalty_ablation.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {prefix}_penalty_ablation.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 14: Economic Evaluation
# ---------------------------------------------------------------------------

def _make_economic_plot(ok: pd.DataFrame, cfg: ExperimentConfig):
    """Bar charts for economic evaluation metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = cfg.artifact_prefix

    econ_metrics = ["capital_ratio_err_99", "var_violation_cost_99", "capital_efficiency_99"]
    econ_metrics = [m for m in econ_metrics if m in ok.columns]
    if not econ_metrics:
        return

    n = len(econ_metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, metric in enumerate(econ_metrics):
        ax = axes[i]
        grp = ok.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean")
        # Exclude penalty ablation for clarity
        grp = grp[~grp.index.str.contains("_pen", na=False)]
        x = np.arange(len(grp))
        ax.bar(x, grp["mean"], yerr=grp["std"].fillna(0), capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(grp.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Economic: {metric}")
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("Economic Evaluation Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{prefix}_economic.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {prefix}_economic.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 4: QQ Plots
# ---------------------------------------------------------------------------

def _make_qq_plots(run_artifacts: dict, cfg: ExperimentConfig):
    """QQ plots comparing real and fake return distributions to normal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate across all runs per model
    model_data = defaultdict(lambda: {"real": [], "fake": []})
    for key, art in run_artifacts.items():
        model_name = key[0]
        model_data[model_name]["real"].append(art["real_paths"].reshape(-1))
        model_data[model_name]["fake"].append(art["fake_paths"].reshape(-1))

    models = sorted(model_data.keys())
    if not models:
        return

    n = len(models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for i, model_name in enumerate(models):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        real_flat = np.concatenate(model_data[model_name]["real"])
        fake_flat = np.concatenate(model_data[model_name]["fake"])

        # QQ: sort quantiles of real and fake, plot against each other
        n_pts = min(500, len(real_flat), len(fake_flat))
        probs = np.linspace(0.5 / n_pts, 1 - 0.5 / n_pts, n_pts)
        real_q = np.quantile(real_flat, probs)
        fake_q = np.quantile(fake_flat, probs)

        ax.scatter(real_q, fake_q, s=6, alpha=0.5, c="steelblue")
        lo = min(real_q.min(), fake_q.min())
        hi = max(real_q.max(), fake_q.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Perfect match")
        ax.set_xlabel("Real quantiles")
        ax.set_ylabel("Generated quantiles")
        ax.set_title(model_name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    # Hide unused axes
    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("QQ Plot: Real vs Generated Return Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_qq_plots.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_qq_plots.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 5: Return Distribution Overlay (histogram + KDE)
# ---------------------------------------------------------------------------

def _make_distribution_overlay(run_artifacts: dict, cfg: ExperimentConfig):
    """Overlay histograms of real vs generated returns per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_data = defaultdict(lambda: {"real": [], "fake": []})
    for key, art in run_artifacts.items():
        model_name = key[0]
        model_data[model_name]["real"].append(art["real_paths"].reshape(-1))
        model_data[model_name]["fake"].append(art["fake_paths"].reshape(-1))

    models = sorted(model_data.keys())
    if not models:
        return

    n = len(models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for i, model_name in enumerate(models):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        real_flat = np.concatenate(model_data[model_name]["real"])
        fake_flat = np.concatenate(model_data[model_name]["fake"])

        # Clip extreme outliers for cleaner visualization
        clip_lo = np.percentile(real_flat, 0.5)
        clip_hi = np.percentile(real_flat, 99.5)
        bins = np.linspace(clip_lo, clip_hi, 80)

        ax.hist(real_flat, bins=bins, density=True, alpha=0.5, color="steelblue", label="Real")
        ax.hist(fake_flat, bins=bins, density=True, alpha=0.5, color="coral", label="Generated")
        ax.set_xlabel("Log return")
        ax.set_ylabel("Density")
        ax.set_title(model_name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Return Distribution: Real vs Generated", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_distributions.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_distributions.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 6: ACF Comparison
# ---------------------------------------------------------------------------

def _make_acf_comparison(run_artifacts: dict, cfg: ExperimentConfig):
    """ACF plots for real vs generated returns (raw and squared)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_data = defaultdict(lambda: {"real": [], "fake": []})
    for key, art in run_artifacts.items():
        model_name = key[0]
        model_data[model_name]["real"].append(art["real_paths"].reshape(-1))
        model_data[model_name]["fake"].append(art["fake_paths"].reshape(-1))

    models = sorted(model_data.keys())
    if not models:
        return

    n_lags = min(cfg.acf_max_lag, 30)
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)

    for i, model_name in enumerate(models):
        real_flat = np.concatenate(model_data[model_name]["real"])
        fake_flat = np.concatenate(model_data[model_name]["fake"])

        real_acf_vals = acf(real_flat, nlags=n_lags, fft=False)
        fake_acf_vals = acf(fake_flat, nlags=n_lags, fft=False)
        real_sq_acf = acf(real_flat ** 2, nlags=n_lags, fft=False)
        fake_sq_acf = acf(fake_flat ** 2, nlags=n_lags, fft=False)

        lags = np.arange(1, n_lags + 1)
        conf = 1.96 / np.sqrt(len(real_flat))

        # Raw returns ACF
        ax = axes[i][0]
        ax.bar(lags - 0.15, real_acf_vals[1:], width=0.3, alpha=0.7, color="steelblue", label="Real")
        ax.bar(lags + 0.15, fake_acf_vals[1:], width=0.3, alpha=0.7, color="coral", label="Generated")
        ax.axhline(conf, ls="--", c="gray", lw=0.8, label="95% CI")
        ax.axhline(-conf, ls="--", c="gray", lw=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(f"{model_name} — Returns ACF", fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)

        # Squared returns ACF (volatility clustering)
        ax = axes[i][1]
        ax.bar(lags - 0.15, real_sq_acf[1:], width=0.3, alpha=0.7, color="steelblue", label="Real")
        ax.bar(lags + 0.15, fake_sq_acf[1:], width=0.3, alpha=0.7, color="coral", label="Generated")
        ax.axhline(conf, ls="--", c="gray", lw=0.8, label="95% CI")
        ax.axhline(-conf, ls="--", c="gray", lw=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF(r²)")
        ax.set_title(f"{model_name} — Squared Returns ACF (Volatility Clustering)", fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)

    fig.suptitle("Autocorrelation: Real vs Generated", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_acf_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_acf_comparison.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 7: Rolling Volatility
# ---------------------------------------------------------------------------

def _make_rolling_volatility(run_artifacts: dict, cfg: ExperimentConfig):
    """Rolling volatility comparison: real vs generated returns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick one representative run per model (first available)
    model_runs = {}
    for key, art in run_artifacts.items():
        model_name = key[0]
        if model_name not in model_runs:
            model_runs[model_name] = art

    models = sorted(model_runs.keys())
    if not models:
        return

    window = 24  # 24-hour rolling window for hourly data

    n = len(models)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows), squeeze=False)

    for i, model_name in enumerate(models):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        art = model_runs[model_name]

        # Use first path from the batch for time-series view
        real_series = art["real_paths"][0] if art["real_paths"].ndim > 1 else art["real_paths"]
        fake_series = art["fake_paths"][0] if art["fake_paths"].ndim > 1 else art["fake_paths"]

        if len(real_series) < window + 1 or len(fake_series) < window + 1:
            ax.text(0.5, 0.5, "Paths too short", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_name, fontsize=10)
            continue

        real_vol = pd.Series(real_series).rolling(window).std().dropna()
        fake_vol = pd.Series(fake_series).rolling(window).std().dropna()

        ax.plot(real_vol.values, lw=1, alpha=0.8, color="steelblue", label="Real")
        ax.plot(fake_vol.values[:len(real_vol)], lw=1, alpha=0.8, color="coral", label="Generated")
        ax.set_xlabel("Time step")
        ax.set_ylabel(f"Rolling σ ({window}h)")
        ax.set_title(model_name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Rolling Volatility: Real vs Generated", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_rolling_vol.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_rolling_vol.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 8: VaR Backtest Timeline
# ---------------------------------------------------------------------------

def _make_var_backtest_plot(run_artifacts: dict, ok: pd.DataFrame, cfg: ExperimentConfig):
    """VaR backtest: returns with VaR threshold and violation markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick one representative run per model
    model_runs = {}
    for key, art in run_artifacts.items():
        model_name = key[0]
        if model_name not in model_runs:
            model_runs[model_name] = art

    models = sorted(model_runs.keys())
    if not models:
        return

    level = 0.99  # Show 99% VaR backtest
    tail_prob = 1.0 - level

    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)

    for i, model_name in enumerate(models):
        ax = axes[i][0]
        art = model_runs[model_name]
        real_flat = art["real_paths"].reshape(-1)
        fake_flat = art["fake_paths"].reshape(-1)

        # VaR estimate from the generated distribution
        var_estimate = np.percentile(fake_flat, tail_prob * 100)

        # Plot real returns
        ax.plot(real_flat, lw=0.5, alpha=0.7, color="steelblue", label="Real returns")
        ax.axhline(var_estimate, color="red", ls="--", lw=1.2,
                    label=f"VaR {int(level*100)}% = {var_estimate:.4f}")

        # Mark violations
        violations = real_flat < var_estimate
        viol_idx = np.where(violations)[0]
        if len(viol_idx) > 0:
            ax.scatter(viol_idx, real_flat[viol_idx], c="red", s=12, zorder=5,
                       marker="v", label=f"Violations ({violations.sum()}/{len(real_flat)})")

        expected_pct = tail_prob * 100
        actual_pct = violations.mean() * 100
        ax.set_title(f"{model_name} — VaR {int(level*100)}% Backtest "
                     f"(expected {expected_pct:.1f}%, actual {actual_pct:.1f}%)", fontsize=9)
        ax.set_xlabel("Observation")
        ax.set_ylabel("Log return")
        ax.legend(fontsize=6, loc="lower left")
        ax.grid(alpha=0.2)

    fig.suptitle("VaR Backtesting: Return Series with Violation Markers",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_var_backtest.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_var_backtest.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 9: Model Radar Chart
# ---------------------------------------------------------------------------

def _make_radar_chart(ok: pd.DataFrame, cfg: ExperimentConfig):
    """Multi-metric radar chart comparing all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["var_rel_err_mean_pct", "es_rel_err_mean_pct", "extreme_rel_err_mean",
               "acf_mean_abs_diff", "ks_stat", "cvm_stat"]
    metrics = [m for m in metrics if m in ok.columns]
    if len(metrics) < 3:
        return

    model_means = ok.groupby("model")[metrics].mean()
    if model_means.empty:
        return

    # Normalize each metric to [0, 1] range (lower = better, so invert)
    norm = model_means.copy()
    for col in metrics:
        col_min = model_means[col].min()
        col_max = model_means[col].max()
        if col_max - col_min > 1e-12:
            norm[col] = (model_means[col] - col_min) / (col_max - col_min)
        else:
            norm[col] = 0.5

    labels = [m.replace("_", "\n").replace("pct", "%") for m in metrics]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap = plt.cm.get_cmap("tab10", len(norm))

    for i, (model_name, row) in enumerate(norm.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", lw=1.5, markersize=4, label=model_name, color=cmap(i))
        ax.fill(angles, values, alpha=0.1, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison Radar (0 = best, 1 = worst)", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_radar.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_radar.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 10: ES Comparison
# ---------------------------------------------------------------------------

def _make_es_comparison(ok: pd.DataFrame, cfg: ExperimentConfig):
    """Expected Shortfall comparison across models at multiple VaR levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Gather ES relative errors at each level
    es_cols = [c for c in ok.columns if c.startswith("es_rel_err_") and c.endswith("_pct")]
    if not es_cols:
        return

    fig, axes = plt.subplots(1, len(es_cols), figsize=(6 * len(es_cols), 5), squeeze=False)

    models = sorted(ok["model"].unique())
    x = np.arange(len(models))

    for j, col in enumerate(es_cols):
        ax = axes[0][j]
        level_str = col.replace("es_rel_err_", "").replace("_pct", "")
        means = ok.groupby("model")[col].mean().reindex(models)
        stds = ok.groupby("model")[col].std().fillna(0).reindex(models)

        bars = ax.bar(x, means.values, yerr=stds.values, capsize=4, alpha=0.8,
                      color=plt.cm.Set2(np.linspace(0, 1, len(models))))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("ES Relative Error (%)")
        ax.set_title(f"ES Error at {level_str}% Level", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Expected Shortfall (ES) Relative Error by Model",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_es_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_es_comparison.png")
    plt.close()


# ---------------------------------------------------------------------------
# CHART 11: PIT Histogram
# ---------------------------------------------------------------------------

def _make_pit_histogram(run_artifacts: dict, cfg: ExperimentConfig):
    """Probability Integral Transform histogram for calibration check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_data = defaultdict(lambda: {"real": [], "fake": []})
    for key, art in run_artifacts.items():
        model_name = key[0]
        model_data[model_name]["real"].append(art["real_paths"].reshape(-1))
        model_data[model_name]["fake"].append(art["fake_paths"].reshape(-1))

    models = sorted(model_data.keys())
    if not models:
        return

    n = len(models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, model_name in enumerate(models):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        fake_flat = np.concatenate(model_data[model_name]["fake"])
        real_flat = np.concatenate(model_data[model_name]["real"])

        # PIT: for each real observation, compute its rank in the fake distribution
        # (empirical CDF of the fake evaluated at each real point)
        fake_sorted = np.sort(fake_flat)
        pit_values = np.searchsorted(fake_sorted, real_flat) / len(fake_sorted)

        ax.hist(pit_values, bins=20, density=True, alpha=0.7, color="steelblue",
                edgecolor="white", label="PIT")
        ax.axhline(1.0, ls="--", c="red", lw=1.2, label="Ideal (uniform)")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title(model_name, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Probability Integral Transform (PIT) — Calibration Check",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{cfg.artifact_prefix}_pit_histogram.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {cfg.artifact_prefix}_pit_histogram.png")
    plt.close()


# ---------------------------------------------------------------------------
# t-SNE (CHART 3 — already existed, kept here)
# ---------------------------------------------------------------------------

def _make_tsne_plot(run_artifacts: dict, cfg: ExperimentConfig):
    """t-SNE visualization of real vs fake paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_per = 300
    rng = np.random.default_rng(42)

    model_samples = defaultdict(list)
    real_samples = []

    for key, art in run_artifacts.items():
        model_name = key[0]
        if art["fake_paths"].shape[0] > 0:
            model_samples[model_name].append(art["fake_paths"])
        if art["real_paths"].shape[0] > 0:
            real_samples.append(art["real_paths"])

    if not model_samples:
        return

    points, labels = [], []

    if real_samples:
        cat = np.concatenate(real_samples)
        n = min(max_per, cat.shape[0])
        points.append(cat[rng.choice(cat.shape[0], n, replace=False)])
        labels.extend(["Real"] * n)

    for name, arrs in sorted(model_samples.items()):
        cat = np.concatenate(arrs)
        n = min(max_per, cat.shape[0])
        points.append(cat[rng.choice(cat.shape[0], n, replace=False)])
        labels.extend([name] * n)

    X = np.concatenate(points)
    if X.shape[0] < 20:
        return

    perp = min(30, max(5, X.shape[0] // 15))
    emb = TSNE(n_components=2, perplexity=perp, random_state=42,
               init="pca", learning_rate="auto").fit_transform(X)

    fig = plt.figure(figsize=(11, 8))
    uniq = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(uniq))

    for i, lbl in enumerate(uniq):
        mask = np.array(labels) == lbl
        marker = "o" if lbl == "Real" else "x"
        alpha = 0.6 if lbl == "Real" else 0.8
        plt.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=alpha,
                    c=[cmap(i)], marker=marker, label=lbl)

    plt.title("t-SNE: Real vs Generated Paths")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", fontsize=7)
    plt.grid(alpha=0.2)

    prefix = cfg.artifact_prefix
    fig.savefig(f"{prefix}_tsne.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {prefix}_tsne.png")
    plt.close()


def _export_latex_tables(ok: pd.DataFrame, cfg: ExperimentConfig):
    """Export key results as LaTeX tables."""
    prefix = cfg.artifact_prefix

    # Model comparison table
    cols = ["primary_tail_score", "var_rel_err_mean_pct", "es_rel_err_mean_pct",
            "acf_mean_abs_diff", "ks_stat", "cvm_stat"]
    cols = [c for c in cols if c in ok.columns]

    summary = ok.groupby("model")[cols].mean().sort_values(cols[0])

    try:
        latex = summary.to_latex(
            float_format="%.3f",
            caption="Model comparison: mean metrics across all tasks (lower is better for all).",
            label="tab:model_comparison",
        )
    except ImportError:
        # Fallback: manual LaTeX generation if jinja2 is too old
        lines = ["\\begin{table}[htbp]", "\\centering",
                 "\\caption{Model comparison: mean metrics across all tasks (lower is better for all).}",
                 "\\label{tab:model_comparison}"]
        col_fmt = "l" + "r" * len(cols)
        lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
        lines.append("\\toprule")
        header = "Model & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        for model_name, row in summary.iterrows():
            vals = " & ".join(f"{row[c]:.3f}" for c in cols)
            escaped_name = model_name.replace('_', '\\_')
            lines.append(f"{escaped_name} & {vals} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        latex = "\n".join(lines)

    with open(f"{prefix}_model_comparison.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {prefix}_model_comparison.tex")


# ============================================================================
# 10. MAIN ENTRY POINT
# ============================================================================


def make_smoke_config() -> ExperimentConfig:
    """Create a reduced config for quick smoke testing (~5 min)."""
    return ExperimentConfig(
        train_ratios=(0.7,),
        seeds=(0,),
        total_steps=50,
        mc_size=32,
        batch_size=32,
        sig_train_windows_cap=500,
        tail_loss_weights=(0, 8),
        penalty_ablation_configs=((10.0, 10.0),),  # Just one for smoke
        timegan_pretrain_steps=20,
        timegan_supervisor_steps=20,
        timegan_joint_steps=50,
        neural_sde_train_steps=100,
        tailgan_train_steps=50,
        n_eval_windows_cap=200,
        cvm_bootstrap_iterations=50,
        cvm_max_points=2000,
        mmd_max_paths=128,
        bootstrap_ci_iterations=200,
        artifact_prefix="smoke_test",
    )


def main():
    """Run the full benchmark."""
    import argparse
    parser = argparse.ArgumentParser(description="Tail-Aware SigCWGAN Benchmark")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a quick smoke test with reduced parameters")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset CSV paths to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files")
    args = parser.parse_args()

    if args.smoke:
        print("=" * 60)
        print("SMOKE TEST MODE (reduced parameters)")
        print("=" * 60)
        cfg = make_smoke_config()
    else:
        cfg = ExperimentConfig()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        cfg.artifact_prefix = os.path.join(args.output_dir, cfg.artifact_prefix)
        cfg.plots_dir = args.output_dir

    # Find datasets
    if args.datasets:
        dataset_csvs = args.datasets
    else:
        dataset_csvs = find_dataset_csvs()

    # Run benchmark
    results_df, run_artifacts = run_benchmark(cfg, dataset_csvs)

    # Export everything
    export_results(results_df, run_artifacts, cfg)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"Total runs: {len(results_df)}")
    ok = results_df[results_df["status"] == "ok"]
    print(f"Successful: {len(ok)}")
    print(f"Failed: {len(results_df) - len(ok)}")
    print(f"{'='*60}")


import sys
sys.argv = ["benchmark", "--output-dir", "/kaggle/working/results"]
main()
