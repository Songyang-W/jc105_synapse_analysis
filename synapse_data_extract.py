#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 17:17:40 2025

@author: songyangwang
"""

from caveclient import CAVEclient
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

client = CAVEclient("jchen_mouse_cortex")


proofread_id = [720575941034757380, 720575941051511894, 720575941057622500, 720575941059432229, 720575941061126260, 720575941061128820, 720575941061146740, 720575941064570382, 720575941067985577, 720575941070678791, 720575941070687028, 720575941071245374, 720575941071680793, 720575941073277164, 720575941073777040, 720575941073819897, 720575941074810450, 720575941075022386, 720575941076332278, 720575941076653272, 720575941076758746, 720575941077619944, 720575941077776594, 720575941077787090, 720575941077967095, 720575941077982455, 720575941078093229, 720575941079569046, 720575941080298878, 720575941081168930, 720575941086022449, 720575941086431003, 720575941086890090, 720575941086891882, 720575941086918398, 720575941088008859, 720575941088408585, 720575941089298708, 720575941089521081, 720575941089937542, 720575941090078452, 720575941090093556, 720575941091424604, 720575941091445340, 720575941091459932, 720575941091629489, 720575941092450933, 720575941092498657, 720575941092554470, 720575941092565911, 720575941092573335, 720575941092921542, 720575941093443783, 720575941094478620, 720575941094704098, 720575941095549619, 720575941095921486, 720575941096535230, 720575941096669548, 720575941096896673, 720575941096935073, 720575941097347217, 720575941098226189, 720575941098234637, 720575941098551211, 720575941098673705, 720575941098730638, 720575941099131663, 720575941099212764, 720575941099267859, 720575941099645213, 720575941100429392, 720575941101423751, 720575941101452679, 720575941101509409, 720575941102657600, 720575941102668096, 720575941102712896, 720575941102870780, 720575941103533573, 720575941103924589, 720575941104131039, 720575941104297454, 720575941104309998, 720575941104699619, 720575941104700643, 720575941104705251, 720575941105379336, 720575941106152505, 720575941106477402, 720575941106600928, 720575941106639968, 720575941107285320, 720575941107298120, 720575941107437825, 720575941109570314, 720575941109605752, 720575941113287333, 720575941114006852, 720575941114011460, 720575941114493115, 720575941114493371, 720575941114494139, 720575941114687139, 720575941114711203, 720575941114933738, 720575941115332394, 720575941115960573, 720575941116548740, 720575941116572664, 720575941116663435, 720575941116976341, 720575941116997589, 720575941117000149, 720575941118374002, 720575941118374514, 720575941120382861, 720575941120702062, 720575941120904478, 720575941121184239, 720575941123725826, 720575941123736578, 720575941124436677, 720575941125743543, 720575941125953181, 720575941126229860, 720575941127918527, 720575941128195568, 720575941128216130, 720575941128500170, 720575941128608670, 720575941128703583, 720575941131576319, 720575941131588607, 720575941132405960, 720575941134134113, 720575941136000282, 720575941139252669, 720575941139510444, 720575941143312378, 720575941144379311, 720575941145828823, 720575941148267857, 720575941149954781, 720575941149993181, 720575941150910648, 720575941152155190, 720575941152796324, 720575941153471868, 720575941153561801, 720575941153566409, 720575941154431375, 720575941154453647, 720575941156114821, 720575941157944107, 720575941158042818, 720575941162298010, 720575941174542567, 720575941183694464, 720575941190025408, 720575941239851589]
syn_proof_only_df = client.materialize.synapse_query(pre_ids=proofread_id,
                                                  post_ids=proofread_id,
                                                  remove_autapses=False,
                                                  desired_resolution=[7.5,7.5,50]
                                                 )


syn_mat = syn_proof_only_df.pivot_table(index="pre_pt_root_id", 
                                        columns="post_pt_root_id", 
                                        values="size", 
                                        aggfunc="count"
                                       ).fillna(0)
syn_mat = syn_mat.reindex(columns=np.array(syn_mat.columns))

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
sns.heatmap(syn_mat, cmap="gray_r", xticklabels=[], yticklabels=[], 
            ax=ax, square=True,
            cbar_kws={"label": "Connected - binary"})
ax.set_title('Connectivity between proofread cells')

syn_mat.to_csv("connectivity_between_proofread_cells.csv")

#%%

# Pick example cell
example_root_id = 720575941080298878

# Query synapse table with synapse_query()
input_syn_df = client.materialize.synapse_query(post_ids=example_root_id,
                                                remove_autapses=False,
                                                desired_resolution=[7.5,7.5,50])

print(f"Total number of input synapses for {example_root_id}: {len(input_syn_df)}")
input_syn_df

# get count of synapses between presynaptic and postsynaptic partners
input_syn_df.groupby(
  ['pre_pt_root_id', 'post_pt_root_id']
).count()[['id']].rename(
  columns={'id': 'syn_count'}
).sort_values(
  by='syn_count',
  ascending=False,
)
# Note that the 'id' part here is just a way to quickly extract one column. This could be any of the remaining column names, 
# but `id` is often convenient because it is common to all tables.


pre_id  = 720575941073777040
post_id = 720575941073777040
syn_df = client.materialize.synapse_query(post_ids=post_id,pre_ids=pre_id,
                                          remove_autapses=False,
                                          desired_resolution=[7.5,7.5,50])

count = (
    syn_df
    .query('pre_pt_root_id == @pre_id and post_pt_root_id == @post_id')
    .shape[0]
)

print(count)  # 0 if not present




#%%

import pandas as pd

proofread_id = list(map(int, proofread_id))  # or map(str, ...) to match your df dtypes

# Pull synapses among your set (include autapses if you want diagonal counts)
df = client.materialize.synapse_query(
    pre_ids=proofread_id, post_ids=proofread_id, remove_autapses=True
)

# Ensure id dtypes match
for col in ['pre_pt_root_id','post_pt_root_id']:
    if df[col].dtype == 'O':
        df[col] = df[col].astype(str)
        proof_ids_cast = list(map(str, proofread_id))
    else:
        df[col] = df[col].astype('int64')
        proof_ids_cast = list(map(int, proofread_id))

# Keep only pairs within the set (defensive)
df = df.query('pre_pt_root_id in @proof_ids_cast and post_pt_root_id in @proof_ids_cast')

# Build dense matrix with all cells present as rows/cols
conn = (
    df.groupby(['pre_pt_root_id','post_pt_root_id'])
      .size().unstack(fill_value=0)
      .reindex(index=proof_ids_cast, columns=proof_ids_cast, fill_value=0)
      .astype(int)
)

# --- Diagnostics: which cells disappeared before reindex? ---
pre_seen  = set(df['pre_pt_root_id'].unique())
post_seen = set(df['post_pt_root_id'].unique())
all_seen  = pre_seen.union(post_seen)
missing   = set(proof_ids_cast) - all_seen

print(f"Input cells: {len(proofread_id)}")
print(f"Cells present in presynapses: {len(pre_seen)}")
print(f"Cells present in postsynapses: {len(post_seen)}")
print(f"Cells present in synapses (pre or post): {len(all_seen)}")
print(f"Cells missing entirely (no synapses after filters): {len(missing)}")
# If you want to see them:
# print(sorted(missing))

#%%
shard_id = 79033446164400438

root_id = client.chunkedgraph.get_root_id(shard_id)
print(root_id)

#%%
import pandas as pd

def pre_post_synapse_count(ids, client, include_autapses=False, desired_resolution=(7.5, 7.5, 50)):
    """
    Return total presynaptic and postsynaptic synapse counts for each cell id.

    Parameters
    ----------
    ids : iterable of int
        Cell (root) ids to summarize.
    client : object
        CAVE / Materialization client with .materialize.synapse_query(...)
    include_autapses : bool, default True
        If False, autapses are excluded (remove_autapses=True).
    desired_resolution : tuple, default (7.5, 7.5, 50)
        Resolution passed through to synapse_query for consistency.

    Returns
    -------
    pandas.DataFrame with columns:
        - cellid
        - presynapse_total
        - postsynapse_total
    """
    ids = list(dict.fromkeys(ids))  # de-duplicate, preserve order

    # 1) All outgoing synapses from these cells (to anyone)
    pres_df = client.materialize.synapse_query(
        pre_ids=ids,
        desired_resolution=list(desired_resolution),
        remove_autapses=(not include_autapses)
    )
    pre_counts = (
        pres_df.groupby("pre_pt_root_id")
        .size()
        .rename("presynapse_total")
        .astype("int64")
        if len(pres_df)
        else pd.Series(dtype="int64", name="presynapse_total")
    )

    # 2) All incoming synapses to these cells (from anyone)
    posts_df = client.materialize.synapse_query(
        post_ids=ids,
        desired_resolution=list(desired_resolution),
        remove_autapses=(not include_autapses)
    )
    post_counts = (
        posts_df.groupby("post_pt_root_id")
        .size()
        .rename("postsynapse_total")
        .astype("int64")
        if len(posts_df)
        else pd.Series(dtype="int64", name="postsynapse_total")
    )

    # Assemble output (zeros for ids with no matches)
    out = pd.DataFrame({"cellid": ids})
    out["presynapse_total"] = out["cellid"].map(pre_counts).fillna(0).astype("int64")
    out["postsynapse_total"] = out["cellid"].map(post_counts).fillna(0).astype("int64")
    return out

output_df = pre_post_synapse_count(proofread_id, client)
print(output_df.head())


#%%

import pandas as pd

def pre_post_synapse_count_from_df(ids, df, include_autapses=True):
    """
    Return total presynaptic and postsynaptic synapse counts for each cell id
    using an in-memory synapse dataframe.

    Parameters
    ----------
    ids : iterable of int
        Cell (root) ids to summarize.
    df : pandas.DataFrame
        Synapse dataframe with at least 'pre_pt_root_id' and 'post_pt_root_id'.
        (Optionally already filtered to the set of synapses you care about.)
    include_autapses : bool, default True
        If False, rows with pre_pt_root_id == post_pt_root_id are excluded.

    Returns
    -------
    pandas.DataFrame with columns:
        - cellid
        - presynapse_total
        - postsynapse_total
    """
    # De-duplicate ids while preserving order
    ids = list(dict.fromkeys(ids))

    # Ensure the columns we need exist
    required_cols = {"pre_pt_root_id", "post_pt_root_id"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    # Optionally remove autapses
    work = df
    if not include_autapses:
        work = work[work["pre_pt_root_id"] != work["post_pt_root_id"]]

    # Pre counts: count rows by presynaptic cell id
    if len(work):
        pre_counts = (
            work.groupby("pre_pt_root_id").size()
            .rename("presynapse_total")
            .astype("int64")
        )
        post_counts = (
            work.groupby("post_pt_root_id").size()
            .rename("postsynapse_total")
            .astype("int64")
        )
    else:
        pre_counts = pd.Series(dtype="int64", name="presynapse_total")
        post_counts = pd.Series(dtype="int64", name="postsynapse_total")

    # Build output aligned to the requested ids (fill zeros if no matches)
    out = pd.DataFrame({"cellid": ids})
    out["presynapse_total"] = out["cellid"].map(pre_counts).fillna(0).astype("int64")
    out["postsynapse_total"] = out["cellid"].map(post_counts).fillna(0).astype("int64")
    return out

result = pre_post_synapse_count_from_df(proofread_id, syn_proof_unique_df, include_autapses=False)
# result -> columns: cellid, presynapse_total, postsynapse_total
