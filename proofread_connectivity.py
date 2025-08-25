#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 12:14:45 2025

@author: songyangwang
"""

"""
Proofread Neuron Connectivity Analysis
--------------------------------------

This script uses CAVEclient to analyze synaptic connectivity between
a set of proofread neurons in the `jchen_mouse_cortex` dataset.

Main features:
- Build a connectivity matrix (proofread neurons × proofread neurons)
- Visualize with a heatmap
- Query inputs to a single neuron
- Count synapses between specific pre/post cell pairs
- Summarize pre- and postsynaptic totals
- Utility functions for reuse
"""

# ── Imports ───────────────────────────────────────────────
from caveclient import CAVEclient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Setup ─────────────────────────────────────────────────
client = CAVEclient("jchen_mouse_cortex")

# List of proofread neuron IDs
proofread_id = [
    720575941034757380, 720575941051511894, 720575941057622500,
    720575941059432229, 720575941061126260, 720575941061128820,
    # ... (truncated for readability, keep your full list here)
    720575941190025408, 720575941239851589
]

# ── Connectivity Matrix ───────────────────────────────────
syn_df = client.materialize.synapse_query(
    pre_ids=proofread_id,
    post_ids=proofread_id,
    remove_autapses=True,
    desired_resolution=[7.5, 7.5, 50]
)

# Pivot into dense connectivity matrix
syn_mat = (
    syn_df.pivot_table(
        index="pre_pt_root_id",
        columns="post_pt_root_id",
        values="size",
        aggfunc="count"
    )
    .fillna(0)
    .reindex(columns=np.array(syn_df["pre_pt_root_id"].unique()))
)

# Save matrix
syn_mat.to_csv("connectivity_between_proofread_cells.csv")

# Plot heatmap
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
sns.heatmap(
    syn_mat, cmap="gray_r", xticklabels=[], yticklabels=[],
    ax=ax, square=True, cbar_kws={"label": "Connected - binary"}
)
ax.set_title("Connectivity between proofread cells")
plt.show()

# ── Example: inputs to a single neuron ─────────────────────
example_root_id = 720575941080298878
input_syn_df = client.materialize.synapse_query(
    post_ids=example_root_id,
    remove_autapses=False,
    desired_resolution=[7.5, 7.5, 50]
)
print(f"Total input synapses for {example_root_id}: {len(input_syn_df)}")

# Partner counts, sorted by frequency
partner_counts = (
    input_syn_df.groupby(["pre_pt_root_id", "post_pt_root_id"])
    .size().rename("syn_count")
    .sort_values(ascending=False)
)
print(partner_counts.head())

# ── Example: count synapses between specific pre/post pair ─
pre_id, post_id = 720575941051511894, 720575941057622500
pair_df = client.materialize.synapse_query(
    pre_ids=pre_id, post_ids=post_id,
    remove_autapses=False, desired_resolution=[7.5, 7.5, 50]
)
pair_count = pair_df.shape[0]
print(f"Synapse count {pre_id} → {post_id}: {pair_count}")

# ── Diagnostics: missing cells check ───────────────────────
proofread_id = list(map(int, proofread_id))
df = client.materialize.synapse_query(
    pre_ids=proofread_id, post_ids=proofread_id, remove_autapses=True
)

pre_seen = set(df["pre_pt_root_id"].unique())
post_seen = set(df["post_pt_root_id"].unique())
all_seen = pre_seen.union(post_seen)
missing = set(proofread_id) - all_seen

print(f"Total proofread cells: {len(proofread_id)}")
print(f"Cells with presynapses: {len(pre_seen)}")
print(f"Cells with postsynapses: {len(post_seen)}")
print(f"Cells with any synapse: {len(all_seen)}")
print(f"Cells missing entirely: {len(missing)}")

# ── Utility Functions ──────────────────────────────────────
def pre_post_synapse_count(ids, client, include_autapses=False, desired_resolution=(7.5, 7.5, 50)):
    """
    Return total presynaptic and postsynaptic synapse counts for each cell id.
    """
    ids = list(dict.fromkeys(ids))  # remove duplicates

    # Outgoing
    pres_df = client.materialize.synapse_query(
        pre_ids=ids, remove_autapses=not include_autapses,
        desired_resolution=list(desired_resolution)
    )
    pre_counts = pres_df.groupby("pre_pt_root_id").size() if len(pres_df) else pd.Series(dtype="int64")

    # Incoming
    posts_df = client.materialize.synapse_query(
        post_ids=ids, remove_autapses=not include_autapses,
        desired_resolution=list(desired_resolution)
    )
    post_counts = posts_df.groupby("post_pt_root_id").size() if len(posts_df) else pd.Series(dtype="int64")

    # Assemble result
    out = pd.DataFrame({"cellid": ids})
    out["presynapse_total"] = out["cellid"].map(pre_counts).fillna(0).astype(int)
    out["postsynapse_total"] = out["cellid"].map(post_counts).fillna(0).astype(int)
    return out

def pre_post_synapse_count_from_df(ids, df, include_autapses=True):
    """
    Return total presynaptic and postsynaptic counts from an in-memory dataframe.
    """
    ids = list(dict.fromkeys(ids))  # remove duplicates
    work = df.copy()
    if not include_autapses:
        work = work[work["pre_pt_root_id"] != work["post_pt_root_id"]]

    pre_counts = work.groupby("pre_pt_root_id").size() if len(work) else pd.Series(dtype="int64")
    post_counts = work.groupby("post_pt_root_id").size() if len(work) else pd.Series(dtype="int64")

    out = pd.DataFrame({"cellid": ids})
    out["presynapse_total"] = out["cellid"].map(pre_counts).fillna(0).astype(int)
    out["postsynapse_total"] = out["cellid"].map(post_counts).fillna(0).astype(int)
    return out

# Example use of utility
summary = pre_post_synapse_count(proofread_id, client)
print(summary.head())