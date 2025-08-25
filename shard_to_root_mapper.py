#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 12:04:12 2025

@author: songyangwang
"""
from caveclient import CAVEclient
import pandas as pd

# ── Load Excel ───────────────────────────────────────────
df = pd.read_excel(
    "lookup table.xlsx",
    sheet_name="MASTER_LIST",
    dtype={"SHARD ID": str, "FINAL NEURON ID": str}
)

# Extract columns (already as strings because of dtype above)
supervoxel_ids = df["SHARD ID"]
root_ids_from_lookup_table = df["FINAL NEURON ID"]

# Convert shard IDs to a plain Python list (optional)
supervoxel_ids_list = supervoxel_ids.tolist()

# ── Connect to CAVEclient ────────────────────────────────
client = CAVEclient("jchen_mouse_cortex")

# ── Query root IDs for each shard ────────────────────────
root_ids = []
for shard_id in df["SHARD ID"]:
    try:
        root_id = client.chunkedgraph.get_root_id(int(shard_id))
    except Exception:
        root_id = None   # store None if lookup fails
    root_ids.append(root_id)

# Add new column with root IDs (as strings for consistency)
df["root id"] = [str(x) if x is not None else "" for x in root_ids]

# ── Filter for a specific root ID ────────────────────────
target_root_id = "720575941088408585"
match = df[df["root id"] == target_root_id]

# ── Output ───────────────────────────────────────────────
print(match)
# To save back into Excel, uncomment the next line:
# df.to_excel("lookup_table_with_root_ids.xlsx", index=False)