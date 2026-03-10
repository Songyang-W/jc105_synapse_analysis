#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_overlap_synapse_utils.py
----------------------------------
Reusable helpers for analysis_overlap_synapse.py.

Sections
--------
1. Data loading and validation
2. Overlap matrix cleaning and reshaping
3. Pair-level synapse aggregation
4. Branch-code parsing
5. Lookup joins
6. Statistical helpers
7. Plotting helpers

All root IDs are kept as strings throughout. No CAVE / network access.
Overlap values are in µm (despite legacy column name 'overlap_length_nm').
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.special import expit  # logistic sigmoid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOXEL_RES_NM = np.array([7.5, 7.5, 50.0])  # nm per voxel [x, y, z]
OVERLAP_COL = "overlap_length_um"            # canonical column name in this analysis
OVERLAP_EPS = 1e-3                           # µm — floor for log-transform


# ===========================================================================
# 1. Data loading and validation
# ===========================================================================

def load_synapse_table(path: str) -> pd.DataFrame:
    """
    Load cleaned_synapse_table.csv with root IDs as strings.

    Returns
    -------
    pd.DataFrame with 'pre_pt_root_id' and 'post_pt_root_id' as str.
    """
    df = pd.read_csv(
        path,
        dtype={"pre_pt_root_id": str, "post_pt_root_id": str},
        low_memory=False,
    )
    df["pre_pt_root_id"] = df["pre_pt_root_id"].str.strip()
    df["post_pt_root_id"] = df["post_pt_root_id"].str.strip()
    print(f"[load_synapse_table] {len(df):,} synapses loaded from {path}")
    return df


def load_overlap_table(path: str) -> pd.DataFrame:
    """
    Load axon_dend_overlap_table.csv (long format).

    Renames the legacy 'overlap_length_nm' column to 'overlap_length_um'
    because values are actually in µm (known naming bug in source data).

    Returns
    -------
    pd.DataFrame with columns: pre_pt_root_id, post_pt_root_id, overlap_length_um.
    """
    df = pd.read_csv(
        path,
        dtype={"pre_pt_root_id": str, "post_pt_root_id": str},
        low_memory=False,
    )
    df["pre_pt_root_id"] = df["pre_pt_root_id"].str.strip()
    df["post_pt_root_id"] = df["post_pt_root_id"].str.strip()

    # Fix legacy column name
    if "overlap_length_nm" in df.columns and OVERLAP_COL not in df.columns:
        df = df.rename(columns={"overlap_length_nm": OVERLAP_COL})
        warnings.warn(
            "Renamed 'overlap_length_nm' → 'overlap_length_um'. "
            "Values are in µm despite the original column name.",
            UserWarning,
            stacklevel=2,
        )

    # Drop duplicates; keep first row per directed pair
    n_before = len(df)
    df = df.drop_duplicates(subset=["pre_pt_root_id", "post_pt_root_id"], keep="first")
    if len(df) < n_before:
        print(f"[load_overlap_table] Dropped {n_before - len(df)} duplicate rows.")

    print(
        f"[load_overlap_table] {len(df):,} directed pairs, "
        f"overlap range [{df[OVERLAP_COL].min():.2f}, {df[OVERLAP_COL].max():.1f}] µm"
    )
    return df


def load_overlap_matrix(path: str) -> pd.DataFrame:
    """
    Load axon_dend_overlap_matrix_5.csv as a DataFrame.

    - Row index = pre (axon donor) root IDs (strings).
    - Column names = post (dendrite target) root IDs (strings).
    - Values in µm.
    - Deduplicates both axes by keeping first occurrence.

    Returns
    -------
    pd.DataFrame (N_pre × N_post), index and columns are str root IDs.
    """
    mat = pd.read_csv(path, index_col=0, dtype=str)

    # Strip whitespace from all labels
    mat.index = mat.index.str.strip()
    mat.columns = mat.columns.str.strip()

    # Convert values to float
    mat = mat.astype(float)

    # Deduplicate rows and columns (keep first)
    n_rows_before = mat.shape[0]
    n_cols_before = mat.shape[1]
    mat = mat[~mat.index.duplicated(keep="first")]
    mat = mat.loc[:, ~mat.columns.duplicated(keep="first")]

    rows_dropped = n_rows_before - mat.shape[0]
    cols_dropped = n_cols_before - mat.shape[1]
    if rows_dropped or cols_dropped:
        print(
            f"[load_overlap_matrix] Dropped {rows_dropped} duplicate rows, "
            f"{cols_dropped} duplicate columns."
        )

    print(
        f"[load_overlap_matrix] Matrix shape after dedup: {mat.shape}  "
        f"(pre={mat.shape[0]}, post={mat.shape[1]})"
    )
    return mat


def load_lookup_table(path: str, sheet: str = "MASTER_LIST") -> pd.DataFrame:
    """
    Load LOOKUP_TABLE.xlsx MASTER_LIST sheet.

    Returns
    -------
    pd.DataFrame with 'FINAL NEURON ID' as str, deduplicated.
    """
    df = pd.read_excel(
        path,
        sheet_name=sheet,
        dtype={"SHARD ID": str, "FINAL NEURON ID": str, "EM NAME (Nuclear)": str},
    )
    df["FINAL NEURON ID"] = df["FINAL NEURON ID"].astype(str).str.strip()
    n_before = len(df)
    df = df.drop_duplicates(subset=["FINAL NEURON ID"], keep="first")
    if len(df) < n_before:
        print(f"[load_lookup_table] Dropped {n_before - len(df)} duplicate FINAL NEURON ID rows.")
    print(f"[load_lookup_table] {len(df)} neurons in lookup table.")
    return df


def validate_id_column(series: pd.Series, name: str = "column") -> None:
    """Warn if any IDs look like scientific notation or are suspiciously short."""
    sci_mask = series.astype(str).str.contains(r"e\+|e-|E\+|E-", na=False, regex=True)
    if sci_mask.any():
        warnings.warn(
            f"[validate_id_column] {sci_mask.sum()} values in '{name}' look like "
            "scientific notation — root IDs may have been corrupted.",
            UserWarning,
            stacklevel=2,
        )


# ===========================================================================
# 2. Overlap matrix cleaning and reshaping
# ===========================================================================

def matrix_to_pairs(mat: pd.DataFrame, min_overlap: float = 0.0) -> pd.DataFrame:
    """
    Convert the overlap matrix to a long-format directed pair table.

    Parameters
    ----------
    mat : pd.DataFrame
        Square-ish overlap matrix (rows = pre, cols = post), values in µm.
    min_overlap : float
        Only keep pairs with overlap >= this value. Default 0.0 keeps all.

    Returns
    -------
    pd.DataFrame with columns: pre_pt_root_id, post_pt_root_id, overlap_length_um.
    Diagonal (self-pairs) is always excluded.
    """
    # Stack, excluding NaN
    stacked = (
        mat.stack()
        .reset_index()
        .rename(columns={
            mat.index.name if mat.index.name else "level_0": "pre_pt_root_id",
            mat.columns.name if mat.columns.name else "level_1": "post_pt_root_id",
            0: OVERLAP_COL,
        })
    )
    stacked.columns = ["pre_pt_root_id", "post_pt_root_id", OVERLAP_COL]

    # Remove self-pairs
    stacked = stacked[stacked["pre_pt_root_id"] != stacked["post_pt_root_id"]].copy()

    # Apply floor
    if min_overlap > 0.0:
        stacked = stacked[stacked[OVERLAP_COL] >= min_overlap].copy()

    stacked = stacked.reset_index(drop=True)
    print(f"[matrix_to_pairs] {len(stacked):,} directed pairs (min_overlap={min_overlap} µm)")
    return stacked


# ===========================================================================
# 3. Pair-level synapse aggregation
# ===========================================================================

def aggregate_synapses_to_pairs(syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate synapse table to one row per directed (pre, post) pair.

    Returns
    -------
    pd.DataFrame with columns:
        pre_pt_root_id, post_pt_root_id, n_synapses,
        mean_pre_soma_dist_nm, mean_post_soma_dist_nm
    """
    agg = (
        syn_df
        .groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False)
        .agg(
            n_synapses=("id", "count"),
            mean_pre_soma_dist_nm=("pre_pt_to_soma_distance_nm", "mean"),
            mean_post_soma_dist_nm=("post_pt_to_soma_distance_nm", "mean"),
        )
        .reset_index()
    )
    print(f"[aggregate_synapses_to_pairs] {len(agg):,} connected directed pairs")
    return agg


# ===========================================================================
# 4. Branch-code parsing
# ===========================================================================

# Branch codes look like: "A1C2B3B4" or "Y1D2A3A4A5A6A7B8B9A10"
# Format: (letter)(number) pairs, where letter = branch identity, number = branch index
# Missing/special values: NaN, "soma", "unknown", "axon", "dendrite"

_BRANCH_PATTERN = re.compile(r"([A-Z]+)(\d+)")


def parse_branch_code(code) -> dict:
    """
    Parse a branch-tree code string into structured features.

    Parameters
    ----------
    code : str or NaN

    Returns
    -------
    dict with keys:
        tree_letter     : str or None  — first letter sequence (e.g. 'A')
        branch_depth    : int          — number of (letter, number) steps
        last_child      : int or None  — the final number in the code
        is_soma         : bool
        is_axon_label   : bool         — code starts with a tree letter (axon branch)
        is_missing      : bool         — NaN or empty
    """
    if pd.isna(code) or str(code).strip() == "" or str(code).lower() == "nan":
        return {
            "tree_letter": None, "branch_depth": 0, "last_child": None,
            "is_soma": False, "is_axon_label": False, "is_missing": True,
        }

    code_str = str(code).strip()

    if code_str.lower() == "soma":
        return {
            "tree_letter": None, "branch_depth": 0, "last_child": None,
            "is_soma": True, "is_axon_label": False, "is_missing": False,
        }

    steps = _BRANCH_PATTERN.findall(code_str)

    if not steps:
        # Unrecognized format
        return {
            "tree_letter": code_str, "branch_depth": 0, "last_child": None,
            "is_soma": False, "is_axon_label": False, "is_missing": False,
        }

    tree_letter = steps[0][0]
    branch_depth = len(steps)
    last_child = int(steps[-1][1])

    return {
        "tree_letter": tree_letter,
        "branch_depth": branch_depth,
        "last_child": last_child,
        "is_soma": False,
        "is_axon_label": True,
        "is_missing": False,
    }


def add_branch_features(df: pd.DataFrame, loc_col: str, prefix: str) -> pd.DataFrame:
    """
    Add parsed branch-code features to a DataFrame as new columns.

    Parameters
    ----------
    df      : DataFrame containing loc_col.
    loc_col : name of the branch-code column.
    prefix  : column prefix, e.g. 'pre' or 'post'.

    Adds columns: {prefix}_tree_letter, {prefix}_branch_depth,
                  {prefix}_last_child, {prefix}_is_soma, {prefix}_is_missing.
    """
    parsed = df[loc_col].apply(parse_branch_code).apply(pd.Series)
    for col in ["tree_letter", "branch_depth", "last_child", "is_soma", "is_missing"]:
        df[f"{prefix}_{col}"] = parsed[col].values
    return df


# ===========================================================================
# 5. Lookup joins
# ===========================================================================

def _make_lookup_side(lookup_df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Prepare a lookup sub-table for one side (pre or post) of a join.
    Selects a standard set of columns and renames with _{side} suffix.
    """
    want = [
        "FINAL NEURON ID",
        "Cell Type",
        "finer label",
        "Functional Category",
        "2P NAME",
    ]
    cols = [c for c in want if c in lookup_df.columns]
    sub = lookup_df[cols].rename(
        columns={c: f"{c}_{side}" for c in cols if c != "FINAL NEURON ID"}
    )
    sub = sub.rename(columns={"FINAL NEURON ID": f"root_id_{side}"})
    return sub


def join_lookup(
    df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    pre_col: str = "pre_pt_root_id",
    post_col: str = "post_pt_root_id",
) -> pd.DataFrame:
    """
    Left-join lookup metadata onto a pair-level DataFrame for both pre and post.

    Adds columns suffixed with _pre and _post:
        Cell Type_pre / _post
        finer label_pre / _post
        Functional Category_pre / _post
        2P NAME_pre / _post  (if present)

    Does not change the number of rows.
    """
    n_in = len(df)

    pre_meta = _make_lookup_side(lookup_df, "pre")
    post_meta = _make_lookup_side(lookup_df, "post")

    df = df.merge(pre_meta, left_on=pre_col, right_on="root_id_pre", how="left")
    df = df.drop(columns=["root_id_pre"], errors="ignore")

    df = df.merge(post_meta, left_on=post_col, right_on="root_id_post", how="left")
    df = df.drop(columns=["root_id_post"], errors="ignore")

    assert len(df) == n_in, "join_lookup changed row count — check for duplicates in lookup_df"

    coverage = df["Cell Type_pre"].notna().mean()
    print(
        f"[join_lookup] Lookup coverage (pre): {coverage:.1%} of {n_in:,} rows"
    )
    return df


# ===========================================================================
# 6. Statistical helpers
# ===========================================================================

def log_overlap(series: pd.Series, eps: float = OVERLAP_EPS) -> pd.Series:
    """Return log(overlap + eps), safe for zero-overlap pairs."""
    return np.log(series + eps)


def spearman_with_n(x: pd.Series, y: pd.Series):
    """
    Spearman correlation with sample size, drops NaN pairs.

    Returns
    -------
    r, p, n
    """
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 5:
        return np.nan, np.nan, len(x_clean)
    r, p = stats.spearmanr(x_clean, y_clean)
    return float(r), float(p), int(mask.sum())


def fit_logistic(X: np.ndarray, y: np.ndarray):
    """
    Fit a simple logistic regression using scipy.optimize (no sklearn required).

    Parameters
    ----------
    X : (n, p) design matrix (should include intercept column if desired)
    y : (n,) binary outcome

    Returns
    -------
    dict with 'coef', 'intercept' (first column treated as intercept),
    'y_pred_prob', 'auc'.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        Xm, ym = X[mask], y[mask]
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(Xm, ym)
        probs = clf.predict_proba(Xm)[:, 1]
        auc = roc_auc_score(ym, probs)
        return {
            "coef": clf.coef_[0],
            "intercept": clf.intercept_[0],
            "y_pred_prob": probs,
            "auc": auc,
            "n": int(mask.sum()),
        }
    except ImportError:
        warnings.warn(
            "sklearn not available — skipping logistic regression.",
            UserWarning,
            stacklevel=2,
        )
        return None


def poisson_deviance_residuals(y_obs: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Poisson deviance residuals: sign(y - mu) * sqrt(2 * (y*log(y/mu) - (y-mu))).
    Zero-observation term handled with the convention 0*log(0) = 0.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_obs > 0, y_obs / np.maximum(y_pred, 1e-10), 1.0)
        inner = y_obs * np.log(ratio) - (y_obs - y_pred)
        inner = np.maximum(inner, 0.0)
    sign = np.sign(y_obs - y_pred)
    return sign * np.sqrt(2.0 * inner)


def fit_poisson_ols_proxy(overlap_um: pd.Series, n_syn: pd.Series):
    """
    Fit a Poisson-like model via log-linear OLS on connected pairs (n_syn >= 1).
    Model: log(E[n_synapses]) = a + b * log(overlap_um + eps)

    Returns
    -------
    dict with 'slope', 'intercept', 'r2', 'pval', fitted values series.
    """
    log_ov = log_overlap(overlap_um)
    log_ns = np.log(n_syn.clip(lower=1))
    mask = np.isfinite(log_ov) & np.isfinite(log_ns)
    if mask.sum() < 5:
        return None
    slope, intercept, r, p, _ = stats.linregress(log_ov[mask], log_ns[mask])
    fitted = np.exp(intercept + slope * log_ov)
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r ** 2,
        "pval": p,
        "fitted": fitted,
        "n": int(mask.sum()),
    }


def asymmetry_index(a_to_b: pd.Series, b_to_a: pd.Series) -> pd.Series:
    """
    Directional asymmetry index = (A→B − B→A) / (A→B + B→A).
    Result is in [-1, +1]. Returns NaN when sum is zero.
    """
    total = a_to_b + b_to_a
    with np.errstate(invalid="ignore"):
        ai = (a_to_b - b_to_a) / total.replace(0, np.nan)
    return ai


# ===========================================================================
# 7. Plotting helpers
# ===========================================================================

def _label_n(ax, n: int, loc: str = "upper left", fontsize: int = 9):
    """Add a small 'n=...' annotation to an axes."""
    anchors = {
        "upper left": (0.03, 0.97),
        "upper right": (0.97, 0.97),
        "lower left": (0.03, 0.05),
        "lower right": (0.97, 0.05),
    }
    xy = anchors.get(loc, (0.03, 0.97))
    ax.text(
        *xy, f"n={n:,}", transform=ax.transAxes,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=fontsize, color="gray",
    )


def scatter_overlap_vs_outcome(
    overlap: pd.Series,
    outcome: pd.Series,
    outcome_label: str,
    fit_line: bool = False,
    ax=None,
    title: str = "",
    log_x: bool = True,
    alpha: float = 0.4,
    color: str = "#2166ac",
):
    """
    Scatter plot of overlap (µm) vs. a continuous outcome (e.g. n_synapses).

    Parameters
    ----------
    log_x : if True, x-axis is log-transformed.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    x = np.log(overlap + OVERLAP_EPS) if log_x else overlap
    ax.scatter(x, outcome, alpha=alpha, s=18, color=color, linewidths=0)

    if fit_line:
        mask = np.isfinite(x) & np.isfinite(outcome)
        if mask.sum() > 5:
            slope, intercept, *_ = stats.linregress(x[mask], outcome[mask])
            xr = np.linspace(x[mask].min(), x[mask].max(), 200)
            ax.plot(xr, intercept + slope * xr, color="firebrick", lw=1.5)

    ax.set_xlabel(f"log(overlap + {OVERLAP_EPS}) µm" if log_x else "Overlap (µm)")
    ax.set_ylabel(outcome_label)
    if title:
        ax.set_title(title, fontsize=10)
    _label_n(ax, int(np.isfinite(x).sum()))
    ax.tick_params(labelsize=8)
    return ax


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, ax=None, label: str = ""):
    """Plot ROC curve and return AUC."""
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        lbl = f"{label} AUC={auc:.3f}" if label else f"AUC={auc:.3f}"
        ax.plot(fpr, tpr, lw=1.8, label=lbl)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC — connection existence")
        ax.legend(fontsize=8)
        return ax, auc
    except ImportError:
        warnings.warn("sklearn not available — ROC plot skipped.", UserWarning, stacklevel=2)
        return ax, np.nan


def boxplot_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    ax=None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    min_n: int = 3,
    palette: str = "tab10",
    log_y: bool = False,
    rotation: int = 45,
):
    """
    Box / strip plot of value_col grouped by group_col.
    Groups with fewer than min_n observations are dropped.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, df[group_col].nunique() * 0.8), 4))

    counts = df[group_col].value_counts()
    valid_groups = counts[counts >= min_n].index
    plot_df = df[df[group_col].isin(valid_groups)].copy()
    groups = sorted(valid_groups)

    cmap = plt.get_cmap(palette)
    for i, g in enumerate(groups):
        vals = plot_df.loc[plot_df[group_col] == g, value_col].dropna()
        ax.boxplot(
            vals, positions=[i], widths=0.5, patch_artist=True,
            boxprops=dict(facecolor=cmap(i % 10), alpha=0.6),
            medianprops=dict(color="black", lw=1.5),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
            whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
        )
        jitter = np.random.default_rng(i).uniform(-0.2, 0.2, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            alpha=0.35, s=10, color=cmap(i % 10), zorder=3,
        )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=rotation, ha="right", fontsize=8)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    return ax


def heatmap_matrix(
    mat: pd.DataFrame,
    ax=None,
    title: str = "",
    cmap: str = "Blues",
    fmt: str = ".0f",
    vmin=None,
    vmax=None,
    annot: bool = True,
):
    """Simple annotated heatmap for small matrices (e.g. type × type)."""
    if ax is None:
        h = max(4, mat.shape[0] * 0.5)
        w = max(5, mat.shape[1] * 0.6)
        _, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(mat.values.astype(float), aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i, format(v, fmt),
                        ha="center", va="center", fontsize=7, color="black",
                    )
    ax.set_title(title, fontsize=10)
    return ax
