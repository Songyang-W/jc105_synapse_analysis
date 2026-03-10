#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_overlap_synapse.py
----------------------------
Main overlap–synapse analysis script.
Run block-by-block in Spyder (#%% sections).

No CAVE / network access required. All data read from local files.

Block order
-----------
 1. Paths and imports
 2. Load and validate data
 3. Clean overlap matrix and deduplicate IDs
 4. Build full directed pair table (all N×N pairs)
 5. Aggregate synapses to directed pair level
 6. Merge overlap + synapse counts + lookup metadata
 7. Overlap vs. connection existence  (logistic regression + ROC)
 8. Overlap vs. synapse count         (scatter + log-linear fit)
 9. Synapse density by type and functional category pair
10. Directional asymmetry             (A→B vs B→A)
11. Soma-distance targeting
12. Branch-code feature analysis
13. Outlier pairs                     (residual table)

Known caveats (see README.md for full list)
-------------------------------------------
- 'overlap_length_nm' columns in source files contain µm values (naming bug).
  All analysis here uses the renamed 'overlap_length_um'.
- 29 root IDs are duplicated in the overlap matrix; deduplicated in Block 3.
- Root IDs are kept as str throughout to avoid float64 overflow.
- Lookup coverage is ~partial: functional category analyses are restricted
  to neurons with entries in LOOKUP_TABLE.xlsx.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

from analysis_overlap_synapse_utils import (
    OVERLAP_COL, OVERLAP_EPS,
    load_synapse_table, load_overlap_table, load_overlap_matrix,
    load_lookup_table, validate_id_column,
    matrix_to_pairs,
    aggregate_synapses_to_pairs,
    add_branch_features,
    join_lookup,
    log_overlap, spearman_with_n,
    fit_logistic, fit_poisson_ols_proxy,
    poisson_deviance_residuals, asymmetry_index,
    scatter_overlap_vs_outcome, plot_roc,
    boxplot_by_group, heatmap_matrix,
)

# Suppress noisy openpyxl / low_memory warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ===========================================================================
# %% Block 1 – Paths and imports
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SYNAPSE_PATH = os.path.join(BASE_DIR, "cleaned_synapse_table.csv")
OVERLAP_MATRIX_PATH = os.path.join(BASE_DIR, "axon_dend_overlap_matrix_5.csv")
OVERLAP_TABLE_PATH = os.path.join(BASE_DIR, "axon_dend_overlap_table.csv")
LOOKUP_PATH = os.path.join(BASE_DIR, "LOOKUP_TABLE.xlsx")

# Random seed for reproducible jitter
RNG = np.random.default_rng(42)

print("Paths configured. Run the next blocks sequentially.")

# ===========================================================================
# %% Block 2 – Load and validate data
# ===========================================================================

syn_df = load_synapse_table(SYNAPSE_PATH)
overlap_table = load_overlap_table(OVERLAP_TABLE_PATH)
overlap_mat_raw = load_overlap_matrix(OVERLAP_MATRIX_PATH)
lookup_df = load_lookup_table(LOOKUP_PATH)

# Quick sanity checks
validate_id_column(syn_df["pre_pt_root_id"], "syn pre_pt_root_id")
validate_id_column(syn_df["post_pt_root_id"], "syn post_pt_root_id")

print("\n--- Synapse table ---")
print(f"  Shape           : {syn_df.shape}")
print(f"  pre_pt_type     : {syn_df['pre_pt_type'].value_counts().to_dict()}")
print(f"  post_pt_type    : {syn_df['post_pt_type'].value_counts().to_dict()}")
print(f"  pre_pt_label    : {syn_df['pre_pt_label'].value_counts().to_dict()}")
print(f"  post_pt_label   : {syn_df['post_pt_label'].value_counts().to_dict()}")

print("\n--- Overlap table ---")
print(f"  Shape           : {overlap_table.shape}")
print(f"  overlap range   : {overlap_table[OVERLAP_COL].min():.3f} – "
      f"{overlap_table[OVERLAP_COL].max():.1f} µm")

print("\n--- Overlap matrix (raw) ---")
print(f"  Shape           : {overlap_mat_raw.shape}")

print("\n--- Lookup table ---")
fc_counts = lookup_df["Functional Category"].notna().sum() if "Functional Category" in lookup_df.columns else "N/A"
print(f"  Rows             : {len(lookup_df)}")
print(f"  With Func. Cat.  : {fc_counts}")

# ===========================================================================
# %% Block 3 – Clean overlap matrix and deduplicate IDs
# ===========================================================================

# overlap_mat_raw already deduplicated by load_overlap_matrix().
# Confirm shape and rename for clarity.
overlap_mat = overlap_mat_raw.copy()
overlap_mat.index.name = "pre_pt_root_id"
overlap_mat.columns.name = "post_pt_root_id"

# All neurons present in the matrix
matrix_ids = sorted(set(overlap_mat.index.tolist()) | set(overlap_mat.columns.tolist()))
print(f"Unique neuron IDs in matrix: {len(matrix_ids)}")

# Verify overlap matrix is NOT symmetric (directional)
# For a random sample of off-diagonal pairs, check A[i,j] vs A[j,i]
common = list(set(overlap_mat.index) & set(overlap_mat.columns))
sample_ids = common[:min(10, len(common))]
fwd = np.array([overlap_mat.loc[a, b] for a, b in zip(sample_ids[:-1], sample_ids[1:])])
rev = np.array([overlap_mat.loc[b, a] for a, b in zip(sample_ids[:-1], sample_ids[1:])])
n_asymmetric = np.sum(np.abs(fwd - rev) > 1e-6)
print(f"Asymmetry check: {n_asymmetric}/{len(fwd)} sampled pairs have fwd ≠ rev  "
      f"(expected >0 for directional matrix)")

# ===========================================================================
# %% Block 4 – Build full directed pair table
# ===========================================================================

# All non-self directed pairs, including zero-overlap pairs
all_pairs = matrix_to_pairs(overlap_mat, min_overlap=0.0)

# Flag: does this pair have any detected overlap?
all_pairs["has_overlap"] = all_pairs[OVERLAP_COL] > 0.0

print(f"\nFull pair table: {len(all_pairs):,} directed pairs")
print(f"  With overlap > 0 : {all_pairs['has_overlap'].sum():,}")
print(f"  Zero overlap     : {(~all_pairs['has_overlap']).sum():,}")

# ===========================================================================
# %% Block 5 – Aggregate synapses to directed pair level
# ===========================================================================

pair_syn = aggregate_synapses_to_pairs(syn_df)

# Also collect dominant pre/post labels per pair (most common across synapses)
label_agg = (
    syn_df
    .groupby(["pre_pt_root_id", "post_pt_root_id"])
    .agg(
        pre_pt_type=("pre_pt_type", lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan),
        post_pt_type=("post_pt_type", lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan),
        pre_pt_label=("pre_pt_label", lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan),
        post_pt_label=("post_pt_label", lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan),
    )
    .reset_index()
)

pair_syn = pair_syn.merge(label_agg, on=["pre_pt_root_id", "post_pt_root_id"], how="left")

print(f"\nPair-level synapse table: {len(pair_syn):,} connected pairs")
print(f"  n_synapses range: {pair_syn['n_synapses'].min()} – {pair_syn['n_synapses'].max()}")

# ===========================================================================
# %% Block 6 – Merge overlap + synapse counts + lookup metadata
# ===========================================================================

# Start from full pair table (all_pairs), left-join synapse counts
merged = all_pairs.merge(
    pair_syn[["pre_pt_root_id", "post_pt_root_id", "n_synapses",
              "mean_pre_soma_dist_nm", "mean_post_soma_dist_nm",
              "pre_pt_type", "post_pt_type",
              "pre_pt_label", "post_pt_label"]],
    on=["pre_pt_root_id", "post_pt_root_id"],
    how="left",
)

# Fill unconnected pairs with n_synapses = 0
merged["n_synapses"] = merged["n_synapses"].fillna(0).astype(int)
merged["connected"] = (merged["n_synapses"] > 0).astype(int)

# Join lookup metadata for pre and post
merged = join_lookup(merged, lookup_df)

# Convenience: synapse density (synapses per µm overlap) — only for overlapping pairs
merged["synapse_density_per_um"] = np.where(
    merged[OVERLAP_COL] > 0,
    merged["n_synapses"] / merged[OVERLAP_COL],
    np.nan,
)

# Functional category pair label
if "Functional Category_pre" in merged.columns and "Functional Category_post" in merged.columns:
    merged["fc_pair"] = (
        merged["Functional Category_pre"].fillna("?")
        + " → "
        + merged["Functional Category_post"].fillna("?")
    )
else:
    merged["fc_pair"] = np.nan

# Cell type pair label
if "Cell Type_pre" in merged.columns and "Cell Type_post" in merged.columns:
    merged["ct_pair"] = (
        merged["Cell Type_pre"].fillna("?")
        + " → "
        + merged["Cell Type_post"].fillna("?")
    )
else:
    merged["ct_pair"] = np.nan

print(f"\nMerged table: {len(merged):,} rows")
print(f"  Connected pairs  : {merged['connected'].sum():,}")
print(f"  Overlap coverage : {merged['has_overlap'].mean():.1%} of all pairs have overlap > 0")
print(f"  Overlap + synapse: {((merged['has_overlap']) & (merged['connected'] == 1)).sum():,}")
print(f"  Overlap only     : {((merged['has_overlap']) & (merged['connected'] == 0)).sum():,}")
print(f"  Synapse only     : {((~merged['has_overlap']) & (merged['connected'] == 1)).sum():,} "
      f"(synapses without detected overlap)")

# ===========================================================================
# %% Block 7 – Overlap vs. connection existence
# ===========================================================================
# Question: does overlap predict whether a synapse exists?
# Approach: logistic regression of connected ~ log(overlap + eps)
# Dataset: all directed pairs from the matrix.

# --- 7a. Group summary: connection rate by overlap bin ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Block 7 — Overlap vs. Connection Existence", fontsize=11, y=1.01)

# Jittered strip: log(overlap) vs connected (binary)
ax = axes[0]
log_ov = log_overlap(merged[OVERLAP_COL])
jitter = RNG.uniform(-0.1, 0.1, len(merged))
colors = np.where(merged["connected"] == 1, "#d73027", "#4575b4")
ax.scatter(log_ov, merged["connected"] + jitter * 0.15, s=5, alpha=0.25, c=colors)
ax.set_xlabel(f"log(overlap + {OVERLAP_EPS}) µm")
ax.set_ylabel("Connected (1) / Not connected (0)")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Not connected", "Connected"])
ax.set_title("All directed pairs")

# Binned connection probability
n_bins = 12
log_ov_binned = pd.cut(log_ov, bins=n_bins)
bin_stats = merged.groupby(log_ov_binned, observed=True)["connected"].agg(
    ["mean", "count"]
).reset_index()
bin_stats.columns = ["bin", "conn_prob", "n"]
bin_midpoints = bin_stats["bin"].apply(lambda b: b.mid)

ax = axes[1]
ax.bar(range(len(bin_stats)), bin_stats["conn_prob"],
       width=0.8, color="#4393c3", alpha=0.8)
ax.set_xticks(range(len(bin_stats)))
ax.set_xticklabels(
    [f"{v:.1f}" for v in bin_midpoints],
    rotation=45, ha="right", fontsize=7,
)
ax.set_xlabel(f"log(overlap + {OVERLAP_EPS}) µm (bin midpoint)")
ax.set_ylabel("Connection probability")
ax.set_title("Binned connection probability vs overlap")
for i, (prob, n) in enumerate(zip(bin_stats["conn_prob"], bin_stats["n"])):
    ax.text(i, prob + 0.01, f"n={n}", ha="center", va="bottom", fontsize=6, rotation=90)

plt.tight_layout()
plt.show()

# --- 7b. Logistic regression ---
X_log = log_ov.values.reshape(-1, 1)
y_bin = merged["connected"].values.astype(float)
logit_result = fit_logistic(X_log, y_bin)

if logit_result is not None:
    print(f"\nLogistic regression: connected ~ log(overlap + {OVERLAP_EPS})")
    print(f"  Coefficient (log_overlap) : {logit_result['coef'][0]:.4f}")
    print(f"  Intercept                 : {logit_result['intercept']:.4f}")
    print(f"  AUC                       : {logit_result['auc']:.4f}")
    print(f"  n                         : {logit_result['n']:,}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    plot_roc(y_bin[np.isfinite(X_log.ravel())],
             logit_result["y_pred_prob"], ax=ax, label="log_overlap")
    plt.tight_layout()
    plt.show()

# --- 7c. Spearman correlation (overlap vs connected) ---
r, p, n = spearman_with_n(merged[OVERLAP_COL], merged["connected"].astype(float))
print(f"\nSpearman r (overlap, connected): r={r:.4f}, p={p:.3g}, n={n:,}")

# ===========================================================================
# %% Block 8 – Overlap vs. synapse count (connected pairs only)
# ===========================================================================
# Question: among connected pairs, does overlap predict synapse count?

conn = merged[merged["connected"] == 1].copy()

# --- 8a. Scatter: overlap vs n_synapses ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Block 8 — Overlap vs. Synapse Count (connected pairs)", fontsize=11, y=1.01)

scatter_overlap_vs_outcome(
    conn[OVERLAP_COL], conn["n_synapses"],
    outcome_label="n_synapses",
    fit_line=True,
    ax=axes[0],
    title="log(overlap) vs n_synapses",
    log_x=True,
)

# log–log version
scatter_overlap_vs_outcome(
    conn[OVERLAP_COL], np.log(conn["n_synapses"].clip(lower=1)),
    outcome_label="log(n_synapses)",
    fit_line=True,
    ax=axes[1],
    title="log–log: overlap vs n_synapses",
    log_x=True,
)

plt.tight_layout()
plt.show()

# --- 8b. Log-linear fit ---
fit = fit_poisson_ols_proxy(conn[OVERLAP_COL], conn["n_synapses"])
if fit is not None:
    print(f"\nLog-linear fit: log(n_syn) ~ a + b*log(overlap + eps)")
    print(f"  slope (b)   : {fit['slope']:.4f}")
    print(f"  intercept   : {fit['intercept']:.4f}")
    print(f"  R²          : {fit['r2']:.4f}")
    print(f"  p-value     : {fit['pval']:.3g}")
    print(f"  n           : {fit['n']:,}")

# --- 8c. Spearman correlation ---
r8, p8, n8 = spearman_with_n(conn[OVERLAP_COL], conn["n_synapses"].astype(float))
print(f"\nSpearman r (overlap, n_synapses | connected): r={r8:.4f}, p={p8:.3g}, n={n8:,}")

# ===========================================================================
# %% Block 9 – Synapse density by type pair / functional category pair
# ===========================================================================
# Restrict to overlapping pairs (overlap > 0) for density calculations.

overlap_pairs = merged[merged[OVERLAP_COL] > 0].copy()

# --- 9a. Synapse density by cell type pair ---
if "ct_pair" in merged.columns and merged["ct_pair"].notna().any():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Block 9 — Synapse Density by Type Pair", fontsize=11, y=1.01)

    # Connection probability (prop. connected)
    ct_conn = (
        overlap_pairs.groupby("ct_pair", observed=True)["connected"]
        .agg(["mean", "count"])
        .reset_index()
    )
    ct_conn.columns = ["ct_pair", "conn_prob", "n"]
    ct_conn = ct_conn[ct_conn["n"] >= 5].sort_values("conn_prob", ascending=False)

    ax = axes[0]
    ax.barh(ct_conn["ct_pair"], ct_conn["conn_prob"], color="#4393c3")
    for i, (prob, n) in enumerate(zip(ct_conn["conn_prob"], ct_conn["n"])):
        ax.text(prob + 0.005, i, f"n={n}", va="center", fontsize=7)
    ax.set_xlabel("Connection probability")
    ax.set_title("Connection probability by cell-type pair\n(among overlapping pairs)")
    ax.tick_params(labelsize=8)

    # Median synapse density among connected overlapping pairs
    conn_ov = overlap_pairs[(overlap_pairs["connected"] == 1) & overlap_pairs["ct_pair"].notna()]
    ct_density = (
        conn_ov.groupby("ct_pair", observed=True)["synapse_density_per_um"]
        .agg(["median", "count"])
        .reset_index()
    )
    ct_density.columns = ["ct_pair", "median_density", "n"]
    ct_density = ct_density[ct_density["n"] >= 3].sort_values("median_density", ascending=False)

    ax = axes[1]
    ax.barh(ct_density["ct_pair"], ct_density["median_density"], color="#d73027")
    for i, (d, n) in enumerate(zip(ct_density["median_density"], ct_density["n"])):
        ax.text(d + 1e-4, i, f"n={n}", va="center", fontsize=7)
    ax.set_xlabel("Median synapse density (syn/µm)")
    ax.set_title("Median synapse density by cell-type pair\n(connected + overlapping)")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()
else:
    print("Cell Type columns not available — skipping type-pair plots.")

# --- 9b. Synapse density by functional category pair (abbreviated) ---
FUNC_ABBREV = {
    "A Match Enhancement": "AME", "A Match Suppression": "AMS",
    "A Modulation": "AM", "A Non-Match Enhancement": "ANME",
    "A NonMatchEnhance": "ANME", "A Non-Match Enhancement and Match Suppression": "ANME_MS",
    "A Only": "AO", "Categorical Match": "CM", "Categorical Non-Match": "CNM",
    "Not Tuned for Direction or Category": "NTDC", "Other Modulated": "OM",
    "P Match Enhancement": "PME", "P Match Suppression": "PMS",
    "P Modulation": "PM", "P Non-Match Enhancement": "PNME",
    "P Non-Match Enhancement and Match Suppression": "PNME_MS", "P Only": "PO",
}

if "Functional Category_pre" in merged.columns:
    merged["fc_pre_abbrev"] = merged["Functional Category_pre"].map(
        lambda x: FUNC_ABBREV.get(x, x) if pd.notna(x) else np.nan
    )
    merged["fc_post_abbrev"] = merged["Functional Category_post"].map(
        lambda x: FUNC_ABBREV.get(x, x) if pd.notna(x) else np.nan
    )
    overlap_pairs = merged[merged[OVERLAP_COL] > 0].copy()

    fc_subset = overlap_pairs[
        overlap_pairs["fc_pre_abbrev"].notna() & overlap_pairs["fc_post_abbrev"].notna()
    ].copy()

    if len(fc_subset) > 0:
        # Pivot: connection probability matrix (pre FC × post FC)
        fc_conn_mat = (
            fc_subset.pivot_table(
                index="fc_pre_abbrev", columns="fc_post_abbrev",
                values="connected", aggfunc="mean",
            )
        )
        fig, ax = plt.subplots(figsize=(max(8, fc_conn_mat.shape[1] * 0.9),
                                        max(6, fc_conn_mat.shape[0] * 0.6)))
        heatmap_matrix(fc_conn_mat, ax=ax,
                       title="Connection probability by functional category pair\n"
                             "(among overlapping pairs)",
                       cmap="Blues", fmt=".2f", vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

        # Pivot: median synapse density
        fc_conn_ov = fc_subset[fc_subset["connected"] == 1]
        if len(fc_conn_ov) > 0:
            fc_dens_mat = (
                fc_conn_ov.pivot_table(
                    index="fc_pre_abbrev", columns="fc_post_abbrev",
                    values="synapse_density_per_um", aggfunc="median",
                )
            )
            fig, ax = plt.subplots(figsize=(max(8, fc_dens_mat.shape[1] * 0.9),
                                            max(6, fc_dens_mat.shape[0] * 0.6)))
            heatmap_matrix(fc_dens_mat, ax=ax,
                           title="Median synapse density (syn/µm) by functional category pair\n"
                                 "(connected + overlapping)",
                           cmap="Reds", fmt=".3f")
            plt.tight_layout()
            plt.show()
    else:
        print("No pairs with both pre and post functional category labels — skipping FC heatmaps.")
else:
    print("Functional Category columns not present — skipping FC analysis.")

# ===========================================================================
# %% Block 10 – Directional asymmetry (A→B vs B→A)
# ===========================================================================
# For each unordered neuron pair {i, j} present in the matrix for BOTH
# directions, compute asymmetry index = (i→j − j→i) / (i→j + j→i).

# Find all reciprocal pairs
common_ids = list(set(overlap_mat.index) & set(overlap_mat.columns))
rows = []
seen = set()
for a in common_ids:
    for b in common_ids:
        if a == b:
            continue
        pair_key = tuple(sorted([a, b]))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        ov_ab = overlap_mat.loc[a, b] if (a in overlap_mat.index and b in overlap_mat.columns) else np.nan
        ov_ba = overlap_mat.loc[b, a] if (b in overlap_mat.index and a in overlap_mat.columns) else np.nan
        n_ab = pair_syn.loc[
            (pair_syn["pre_pt_root_id"] == a) & (pair_syn["post_pt_root_id"] == b), "n_synapses"
        ].values
        n_ba = pair_syn.loc[
            (pair_syn["pre_pt_root_id"] == b) & (pair_syn["post_pt_root_id"] == a), "n_synapses"
        ].values
        rows.append({
            "id_a": a, "id_b": b,
            "overlap_a_to_b": ov_ab,
            "overlap_b_to_a": ov_ba,
            "n_syn_a_to_b": int(n_ab[0]) if len(n_ab) else 0,
            "n_syn_b_to_a": int(n_ba[0]) if len(n_ba) else 0,
        })

recip_df = pd.DataFrame(rows)
recip_df["overlap_ai"] = asymmetry_index(
    recip_df["overlap_a_to_b"], recip_df["overlap_b_to_a"]
)

# Only keep pairs where at least one direction has overlap
both_ov = recip_df[(recip_df["overlap_a_to_b"] > 0) | (recip_df["overlap_b_to_a"] > 0)].copy()
print(f"\nReciprocal pairs with ≥1 direction having overlap: {len(both_ov):,}")

# Pairs where BOTH directions have overlap
mutual_ov = recip_df[(recip_df["overlap_a_to_b"] > 0) & (recip_df["overlap_b_to_a"] > 0)].copy()
print(f"Mutual overlap (both A→B and B→A > 0): {len(mutual_ov):,}")

# --- 10a. Scatter: overlap A→B vs B→A ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Block 10 — Directional Asymmetry", fontsize=11, y=1.01)

ax = axes[0]
ax.scatter(
    mutual_ov["overlap_a_to_b"], mutual_ov["overlap_b_to_a"],
    s=14, alpha=0.4, color="#4575b4", linewidths=0,
)
max_ov = max(mutual_ov["overlap_a_to_b"].max(), mutual_ov["overlap_b_to_a"].max())
ax.plot([0, max_ov], [0, max_ov], "k--", lw=0.8, label="diagonal (symmetric)")
ax.set_xlabel("Overlap A→B (µm)")
ax.set_ylabel("Overlap B→A (µm)")
ax.set_title(f"Mutual overlap (n={len(mutual_ov):,} pairs)")
ax.legend(fontsize=8)

r_recip, p_recip, _ = spearman_with_n(
    mutual_ov["overlap_a_to_b"], mutual_ov["overlap_b_to_a"]
)
ax.text(0.03, 0.97, f"r={r_recip:.3f}, p={p_recip:.3g}",
        transform=ax.transAxes, va="top", fontsize=8, color="gray")

# --- 10b. Asymmetry index distribution ---
ax = axes[1]
ai_vals = mutual_ov["overlap_ai"].dropna()
ax.hist(ai_vals, bins=40, color="#4575b4", alpha=0.7, edgecolor="none")
ax.axvline(0, color="black", lw=1, linestyle="--")
ax.set_xlabel("Asymmetry index (A→B − B→A) / (A→B + B→A)")
ax.set_ylabel("Count")
ax.set_title(f"Distribution of directional asymmetry\n(n={len(ai_vals):,} mutual pairs)")
med_ai = float(ai_vals.median())
ax.text(0.97, 0.97, f"median={med_ai:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")

plt.tight_layout()
plt.show()

# --- 10c. Does overlap asymmetry predict synapse direction? ---
# Among pairs where at least one direction has synapses:
syn_dir = mutual_ov[
    (mutual_ov["n_syn_a_to_b"] + mutual_ov["n_syn_b_to_a"]) > 0
].copy()

syn_dir["syn_ai"] = asymmetry_index(
    syn_dir["n_syn_a_to_b"].astype(float),
    syn_dir["n_syn_b_to_a"].astype(float),
)

if len(syn_dir) > 5:
    r10c, p10c, n10c = spearman_with_n(syn_dir["overlap_ai"], syn_dir["syn_ai"])
    print(
        f"\nAsymmetry correlation (overlap AI vs synapse AI): "
        f"r={r10c:.4f}, p={p10c:.3g}, n={n10c:,}"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(syn_dir["overlap_ai"], syn_dir["syn_ai"],
               s=18, alpha=0.45, color="#d73027", linewidths=0)
    ax.axhline(0, color="gray", lw=0.7, linestyle="--")
    ax.axvline(0, color="gray", lw=0.7, linestyle="--")
    ax.set_xlabel("Overlap asymmetry index")
    ax.set_ylabel("Synapse count asymmetry index")
    ax.set_title("Overlap asymmetry vs synapse direction asymmetry")
    ax.text(0.03, 0.97, f"r={r10c:.3f}, p={p10c:.3g}, n={n10c:,}",
            transform=ax.transAxes, va="top", fontsize=8, color="gray")
    plt.tight_layout()
    plt.show()

# ===========================================================================
# %% Block 11 – Soma-distance targeting
# ===========================================================================
# Question: does the pre neuron cell type predict where on the post neuron
# it targets (proximal vs. distal)?

# Use per-synapse data for this block (not pair-level)
syn_valid = syn_df[syn_df["post_pt_to_soma_distance_nm"].notna()].copy()

print(f"\nSynapses with valid post_pt_to_soma_distance_nm: {len(syn_valid):,}")

# Convert nm to µm for readability
syn_valid["post_soma_dist_um"] = syn_valid["post_pt_to_soma_distance_nm"] / 1000.0
syn_valid["pre_soma_dist_um"] = syn_valid["pre_pt_to_soma_distance_nm"] / 1000.0

# --- 11a. Post soma distance by pre_pt_type ---
if "pre_pt_type" in syn_valid.columns and syn_valid["pre_pt_type"].notna().any():
    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_by_group(
        syn_valid, group_col="pre_pt_type", value_col="post_soma_dist_um",
        ax=ax,
        title="Post-synaptic soma distance by pre-neuron type",
        xlabel="Pre-neuron type (pre_pt_type)",
        ylabel="Distance to post soma (µm)",
        log_y=False,
        rotation=30,
    )
    plt.tight_layout()
    plt.show()

# --- 11b. Post soma distance by post_pt_type ---
if "post_pt_type" in syn_valid.columns and syn_valid["post_pt_type"].notna().any():
    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_by_group(
        syn_valid, group_col="post_pt_type", value_col="post_soma_dist_um",
        ax=ax,
        title="Post-synaptic soma distance by post-neuron type",
        xlabel="Post-neuron type (post_pt_type)",
        ylabel="Distance to post soma (µm)",
        log_y=False,
        rotation=30,
    )
    plt.tight_layout()
    plt.show()

# --- 11c. Post soma distance split by post_pt_label (axon / dendrite / soma) ---
if "post_pt_label" in syn_valid.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_by_group(
        syn_valid, group_col="post_pt_label", value_col="post_soma_dist_um",
        ax=ax,
        title="Post-synaptic soma distance by post compartment label",
        xlabel="Post compartment (post_pt_label)",
        ylabel="Distance to post soma (µm)",
        log_y=False,
        rotation=0,
    )
    plt.tight_layout()
    plt.show()

# --- 11d. Pre soma distance by pre_pt_type ---
pre_valid = syn_df[syn_df["pre_pt_to_soma_distance_nm"].notna()].copy()
pre_valid["pre_soma_dist_um"] = pre_valid["pre_pt_to_soma_distance_nm"] / 1000.0
if "pre_pt_type" in pre_valid.columns and pre_valid["pre_pt_type"].notna().any():
    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_by_group(
        pre_valid, group_col="pre_pt_type", value_col="pre_soma_dist_um",
        ax=ax,
        title="Pre-synaptic soma distance (where synapses originate) by pre-neuron type",
        xlabel="Pre-neuron type (pre_pt_type)",
        ylabel="Distance from pre soma (µm)",
        log_y=False,
        rotation=30,
    )
    plt.tight_layout()
    plt.show()

# --- 11e. KDE: distribution of post soma distance by pre_pt_label ---
if "pre_pt_label" in syn_valid.columns:
    labels_present = syn_valid["pre_pt_label"].dropna().unique()
    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl in sorted(labels_present):
        subset = syn_valid.loc[syn_valid["pre_pt_label"] == lbl, "post_soma_dist_um"].dropna()
        if len(subset) < 5:
            continue
        try:
            kde = stats.gaussian_kde(subset, bw_method="scott")
            x_range = np.linspace(0, subset.quantile(0.99), 300)
            ax.plot(x_range, kde(x_range), label=f"{lbl} (n={len(subset):,})", lw=1.8)
        except Exception:
            pass
    ax.set_xlabel("Distance to post soma (µm)")
    ax.set_ylabel("Density")
    ax.set_title("Post soma distance distribution by pre compartment label")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# ===========================================================================
# %% Block 12 – Branch-code feature analysis
# ===========================================================================
# Parse pre_pt_location and post_pt_location from the synapse table.

syn_bc = syn_df.copy()
syn_bc = add_branch_features(syn_bc, loc_col="pre_pt_location", prefix="pre")
syn_bc = add_branch_features(syn_bc, loc_col="post_pt_location", prefix="post")

print(f"\nBranch-code parsing complete on {len(syn_bc):,} synapses")
print(f"  pre missing branch code : {syn_bc['pre_is_missing'].sum():,}")
print(f"  post missing branch code: {syn_bc['post_is_missing'].sum():,}")
print(f"  pre depth range: {syn_bc['pre_branch_depth'].min()} – "
      f"{syn_bc['pre_branch_depth'].max()}")
print(f"  post depth range: {syn_bc['post_branch_depth'].min()} – "
      f"{syn_bc['post_branch_depth'].max()}")

# --- 12a. Synapse count by post branch depth ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Block 12 — Branch-Code Feature Analysis", fontsize=11, y=1.01)

syn_depth_post = (
    syn_bc[~syn_bc["post_is_missing"] & ~syn_bc["post_is_soma"]]
    .groupby("post_branch_depth")
    .size()
    .reset_index(name="count")
)
ax = axes[0]
ax.bar(syn_depth_post["post_branch_depth"], syn_depth_post["count"],
       color="#4575b4", alpha=0.8)
ax.set_xlabel("Post branch depth (number of branch steps)")
ax.set_ylabel("Synapse count")
ax.set_title("Synapse count by post branch depth")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# --- 12b. Synapse count by pre branch depth ---
syn_depth_pre = (
    syn_bc[~syn_bc["pre_is_missing"] & ~syn_bc["pre_is_soma"]]
    .groupby("pre_branch_depth")
    .size()
    .reset_index(name="count")
)
ax = axes[1]
ax.bar(syn_depth_pre["pre_branch_depth"], syn_depth_pre["count"],
       color="#d73027", alpha=0.8)
ax.set_xlabel("Pre branch depth (number of branch steps)")
ax.set_ylabel("Synapse count")
ax.set_title("Synapse count by pre branch depth")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()

# --- 12c. Post tree-letter distribution by pre_pt_type ---
if "pre_pt_type" in syn_bc.columns:
    valid_bc = syn_bc[~syn_bc["post_is_missing"] & ~syn_bc["post_is_soma"]
                      & syn_bc["pre_pt_type"].notna()].copy()
    if len(valid_bc) > 0:
        letter_by_type = (
            valid_bc.groupby(["pre_pt_type", "post_tree_letter"])
            .size()
            .unstack(fill_value=0)
        )
        # Normalize to proportion within each pre_pt_type row
        letter_pct = letter_by_type.div(letter_by_type.sum(axis=1), axis=0) * 100
        # Keep only letters with ≥1% total share
        letter_pct = letter_pct.loc[:, letter_pct.max() >= 1.0]

        fig, ax = plt.subplots(figsize=(max(7, letter_pct.shape[1] * 0.5), 4))
        letter_pct.T.plot(kind="bar", ax=ax, alpha=0.75, width=0.75)
        ax.set_xlabel("Post tree letter (dendrite branch identity)")
        ax.set_ylabel("% of synapses")
        ax.set_title("Post branch-tree letter distribution by pre_pt_type")
        ax.legend(title="pre_pt_type", bbox_to_anchor=(1.01, 1), fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        plt.tight_layout()
        plt.show()

# --- 12d. Post soma distance vs. post branch depth ---
dist_depth = syn_bc[
    syn_bc["post_pt_to_soma_distance_nm"].notna()
    & ~syn_bc["post_is_missing"]
    & ~syn_bc["post_is_soma"]
].copy()
dist_depth["post_soma_dist_um"] = dist_depth["post_pt_to_soma_distance_nm"] / 1000.0

if len(dist_depth) > 10:
    r12d, p12d, n12d = spearman_with_n(
        dist_depth["post_branch_depth"].astype(float),
        dist_depth["post_soma_dist_um"],
    )
    print(
        f"\nPost branch depth vs. post soma distance: "
        f"r={r12d:.4f}, p={p12d:.3g}, n={n12d:,}"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    boxplot_by_group(
        dist_depth,
        group_col="post_branch_depth",
        value_col="post_soma_dist_um",
        ax=ax,
        title="Post soma distance by post branch depth",
        xlabel="Post branch depth",
        ylabel="Distance to post soma (µm)",
        rotation=0,
    )
    ax.text(0.97, 0.97, f"r={r12d:.3f}, p={p12d:.3g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")
    plt.tight_layout()
    plt.show()

# ===========================================================================
# %% Block 13 – Outlier pairs
# ===========================================================================
# Fit expected synapse count from overlap using the log-linear model (Block 8).
# Flag pairs whose Poisson deviance residual > 2 (more synapses than expected)
# or < -2 (fewer than expected).

conn_ov = merged[(merged["connected"] == 1) & (merged[OVERLAP_COL] > 0)].copy()

fit_out = fit_poisson_ols_proxy(conn_ov[OVERLAP_COL], conn_ov["n_synapses"])

if fit_out is not None:
    conn_ov["expected_n_syn"] = fit_out["fitted"].values
    conn_ov["deviance_residual"] = poisson_deviance_residuals(
        conn_ov["n_synapses"].values,
        conn_ov["expected_n_syn"].values,
    )

    RESID_THRESH = 2.0
    over_conn = conn_ov[conn_ov["deviance_residual"] > RESID_THRESH].sort_values(
        "deviance_residual", ascending=False
    )
    under_conn = conn_ov[conn_ov["deviance_residual"] < -RESID_THRESH].sort_values(
        "deviance_residual"
    )

    print(f"\n--- Outlier pairs (|deviance residual| > {RESID_THRESH}) ---")
    print(f"  Over-connected  : {len(over_conn):,} pairs")
    print(f"  Under-connected : {len(under_conn):,} pairs")

    display_cols = [
        "pre_pt_root_id", "post_pt_root_id",
        OVERLAP_COL, "n_synapses", "expected_n_syn", "deviance_residual",
    ]
    # Add type/FC columns if available
    for c in ["pre_pt_type", "post_pt_type",
              "Cell Type_pre", "Cell Type_post",
              "Functional Category_pre", "Functional Category_post"]:
        if c in conn_ov.columns:
            display_cols.append(c)

    display_cols = [c for c in display_cols if c in conn_ov.columns]

    print("\nTop over-connected pairs (more synapses than overlap predicts):")
    print(over_conn[display_cols].head(15).to_string(index=False))

    print("\nTop under-connected pairs (fewer synapses than overlap predicts):")
    print(under_conn[display_cols].head(15).to_string(index=False))

    # --- Residual scatter plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Block 13 — Outlier Pairs", fontsize=11, y=1.01)

    ax = axes[0]
    log_ov_conn = log_overlap(conn_ov[OVERLAP_COL])
    resid = conn_ov["deviance_residual"]
    colors_r = np.where(resid > RESID_THRESH, "#d73027",
                np.where(resid < -RESID_THRESH, "#4575b4", "#aaaaaa"))
    ax.scatter(log_ov_conn, resid, c=colors_r, s=14, alpha=0.55, linewidths=0)
    ax.axhline(RESID_THRESH, color="#d73027", lw=1, linestyle="--",
               label=f"+{RESID_THRESH} threshold")
    ax.axhline(-RESID_THRESH, color="#4575b4", lw=1, linestyle="--",
               label=f"-{RESID_THRESH} threshold")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel(f"log(overlap + {OVERLAP_EPS}) µm")
    ax.set_ylabel("Deviance residual")
    ax.set_title("Deviance residuals vs overlap\n(red=over, blue=under)")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.hist(resid.dropna(), bins=40, color="#888888", alpha=0.7, edgecolor="none")
    ax.axvline(RESID_THRESH, color="#d73027", lw=1.2, linestyle="--")
    ax.axvline(-RESID_THRESH, color="#4575b4", lw=1.2, linestyle="--")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Deviance residual")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of deviance residuals")

    plt.tight_layout()
    plt.show()

    # Save outlier table for Neuroglancer follow-up (optional, commented out by default)
    # over_conn[display_cols].to_csv(os.path.join(BASE_DIR, "outlier_over_connected.csv"), index=False)
    # under_conn[display_cols].to_csv(os.path.join(BASE_DIR, "outlier_under_connected.csv"), index=False)
    print("\n(Outlier CSVs not saved. Uncomment the lines above to save.)")

else:
    print("Log-linear fit failed — not enough connected+overlapping pairs for outlier analysis.")

print("\n=== Analysis complete ===")
