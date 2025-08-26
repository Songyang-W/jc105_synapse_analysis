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

#%% ── Imports ───────────────────────────────────────────────
from caveclient import CAVEclient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% ── Setup ─────────────────────────────────────────────────
def remove_duplicates(df):
    return df.drop_duplicates(subset=['pre_pt_supervoxel_id', 'post_pt_supervoxel_id'])
def remove_false_autapses(df):
    '''caveclient's remove_autapses function remove cell with the same pre_pt_root_id
    and the same post_pt_root_id, this function use supervoxel id as a filter'''
    return df[df['pre_pt_supervoxel_id'] != df['post_pt_supervoxel_id']]
client = CAVEclient("jchen_mouse_cortex")

# List of proofread neuron IDs
proofread_id = [720575941034757380, 720575941051511894, 720575941057622500, 720575941059432229, 720575941061126260, 720575941061128820, 720575941061146740, 720575941064570382, 720575941067985577, 720575941070678791, 720575941070687028, 720575941071245374, 720575941071680793, 720575941073277164, 720575941073777040, 720575941073819897, 720575941074810450, 720575941075022386, 720575941076332278, 720575941076653272, 720575941076758746, 720575941077619944, 720575941077776594, 720575941077787090, 720575941077967095, 720575941077982455, 720575941078093229, 720575941079569046, 720575941080298878, 720575941081168930, 720575941086022449, 720575941086431003, 720575941086890090, 720575941086891882, 720575941086918398, 720575941088008859, 720575941088408585, 720575941089298708, 720575941089521081, 720575941089937542, 720575941090078452, 720575941090093556, 720575941091424604, 720575941091445340, 720575941091459932, 720575941091629489, 720575941092450933, 720575941092498657, 720575941092554470, 720575941092565911, 720575941092573335, 720575941092921542, 720575941093443783, 720575941094478620, 720575941094704098, 720575941095549619, 720575941095921486, 720575941096535230, 720575941096669548, 720575941096896673, 720575941096935073, 720575941097347217, 720575941098226189, 720575941098234637, 720575941098551211, 720575941098673705, 720575941098730638, 720575941099131663, 720575941099212764, 720575941099267859, 720575941099645213, 720575941100429392, 720575941101423751, 720575941101509409, 720575941102657600, 720575941102668096, 720575941102712896, 720575941102870780, 720575941103533573, 720575941103924589, 720575941104131039, 720575941104297454, 720575941104309998, 720575941104699619, 720575941104700643, 720575941104705251, 720575941105379336, 720575941106152505, 720575941106477402, 720575941106600928, 720575941106639968, 720575941107285320, 720575941107298120, 720575941107437825, 720575941109570314, 720575941109605752, 720575941113287333, 720575941114006852, 720575941114011460, 720575941114493115, 720575941114493371, 720575941114494139, 720575941114687139, 720575941114711203, 720575941114933738, 720575941115332394, 720575941115960573, 720575941116548740, 720575941116572664, 720575941116663435, 720575941116976341, 720575941116997589, 720575941117000149, 720575941118374002, 720575941118374514, 720575941120382861, 720575941120702062, 720575941120904478, 720575941121184239, 720575941123725826, 720575941123736578, 720575941124436677, 720575941124446149, 720575941125743543, 720575941125953181, 720575941126229860, 720575941127918527, 720575941128195568, 720575941128216130, 720575941128500170, 720575941128608670, 720575941128703583, 720575941130257298, 720575941131576319, 720575941131588607, 720575941132405960, 720575941134134113, 720575941139252669, 720575941139510444, 720575941143312378, 720575941144379311, 720575941145828823, 720575941148267857, 720575941149954781, 720575941149993181, 720575941150910648, 720575941152155190, 720575941152796324, 720575941153471868, 720575941153561801, 720575941153566409, 720575941154431375, 720575941154453647, 720575941156114821, 720575941157944107, 720575941158042818, 720575941162298010, 720575941174542567, 720575941183694464, 720575941190025408, 720575941239851589
]

#%% ── Connectivity Matrix ───────────────────────────────────
syn_df = client.materialize.synapse_query(
    pre_ids=proofread_id,
    post_ids=proofread_id,
    remove_autapses=False,
    desired_resolution=[7.5, 7.5, 50]
)




#%% Pivot into dense connectivity matrix
syn_mat = (
    syn_df.pivot_table(
        index="pre_pt_root_id",
        columns="post_pt_root_id",
        values="size",
        aggfunc="count"
    )
    .fillna(0)
    
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
# to check the connectivity result, go count synapses between specific pre/post pair
# to check the number missing because of materialization go diagnostics missing cells check
#%% ── Example: inputs to a single neuron ─────────────────────
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
pre_id, post_id = 720575941057622500, 720575941057622500
pair_df = client.materialize.synapse_query(
    pre_ids=pre_id, post_ids=post_id,
    remove_autapses=False, desired_resolution=[7.5, 7.5, 50]
)
pair_count = pair_df.shape[0]
print(f"Synapse count {pre_id} → {post_id}: {pair_count}")

#%% ── Diagnostics: missing cells check ───────────────────────
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
if len(missing)>0:
    print("use live query function to generate dataframe, and concatenate with current dataframe")
#%% ── Utility Functions ──────────────────────────────────────
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


# Example use of utility
summary = pre_post_synapse_count(proofread_id, client)
print(summary.head())

#%% RUN THIS IF NOT ALL NEURONS ARE IN MATERIALIZATION
# after getting missings from Diagnostics: missing cells check
# we need to pull data for those not on materlization neuron

# creating dataframe for missing neurons using live query
def remove_duplicates(df):
    return df.drop_duplicates(subset=['pre_pt_supervoxel_id', 'post_pt_supervoxel_id'])
def remove_false_autapses(df):
    return df[df['pre_pt_supervoxel_id'] != df['post_pt_supervoxel_id']]

missing_list = list(missing)

import datetime
synapse_table = client.info.get_datastack_info()['synapse_table']

syn_proof_post_missing_df = pd.DataFrame()
syn_proof_pre_missing_df = pd.DataFrame()

for neuron in missing_list:
    syn_missing_post_df_individual=client.materialize.live_query(synapse_table,
                                                                 datetime.datetime.now(datetime.timezone.utc),
                                                                 filter_equal_dict = {'post_pt_root_id': neuron},desired_resolution=[7.5, 7.5, 50])
    syn_missing_pre_df_individual=client.materialize.live_query(synapse_table,
                                                                 datetime.datetime.now(datetime.timezone.utc),
                                                                 filter_equal_dict = {'pre_pt_root_id': neuron},desired_resolution=[7.5, 7.5, 50])
    syn_proof_post_missing_df = pd.concat(
        [syn_proof_post_missing_df, syn_missing_post_df_individual],
        ignore_index=True)
    syn_proof_pre_missing_df = pd.concat(
        [syn_proof_pre_missing_df, syn_missing_pre_df_individual],
        ignore_index=True)
    
# creating dataframe for other neurons using materialize
syn_proof_pre_df = client.materialize.synapse_query(
    pre_ids=proofread_id,
    remove_autapses=False,
    desired_resolution=[7.5, 7.5, 50]
)
syn_proof_post_df = client.materialize.synapse_query(
    post_ids=proofread_id,
    remove_autapses=False,
    desired_resolution=[7.5, 7.5, 50]
)

syn_proof_conc_df = pd.concat([syn_proof_post_missing_df,syn_proof_post_df, syn_proof_pre_missing_df,syn_proof_pre_df], ignore_index=True)

syn_proof_conc_unique_df = remove_duplicates(syn_proof_conc_df)
syn_proof_conc_unique_no_autapses_df = remove_false_autapses(syn_proof_conc_unique_df)

#%% same analysis but use df
def pre_post_synapse_count_from_df(ids, df):
    """
    Return total presynaptic and postsynaptic counts from an in-memory dataframe.
    """
    ids = list(dict.fromkeys(ids))  # remove duplicates
    work = df.copy()

    pre_counts = work.groupby("pre_pt_root_id").size() if len(work) else pd.Series(dtype="int64")
    post_counts = work.groupby("post_pt_root_id").size() if len(work) else pd.Series(dtype="int64")

    out = pd.DataFrame({"cellid": ids})
    out["presynapse_total"] = out["cellid"].map(pre_counts).fillna(0).astype(int)
    out["postsynapse_total"] = out["cellid"].map(post_counts).fillna(0).astype(int)
    return out

mask = syn_proof_conc_unique_no_autapses_df["pre_pt_root_id"].isin(proofread_id) & syn_proof_conc_unique_no_autapses_df["post_pt_root_id"].isin(proofread_id)
sub = syn_proof_conc_unique_no_autapses_df.loc[mask]
syn_mat = (
        sub.pivot_table(
            index="pre_pt_root_id",
            columns="post_pt_root_id",
            values="size",
            aggfunc="count"
        )
        .fillna(0)
        
    )


pre_post_synapse_count_all = pre_post_synapse_count_from_df(proofread_id,syn_proof_conc_unique_no_autapses_df)

pre_post_synapse_count_all.to_csv('pre_post_synapse_count.csv')
syn_mat.to_csv("connectivity_between_proofread_cells.csv")
sub.to_csv("synapse_coordinates_between_pairs_neurons.csv")
syn_proof_conc_unique_no_autapses_df.to_csv("synapse_coordinates_all_neurons.csv")


