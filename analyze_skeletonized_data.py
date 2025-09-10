#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:07:45 2025
run after finish skeletonize_data.py
right now doesn't have load function yet, so just run the skeletonize_data
@author: songyangwang
"""

#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root_id = 720575941086918398
soma_location = [145964, 169824, 3455]

# skel = pcg_skel.pcg_skeleton(root_id=root_id, client=client)

root_resolution = [7.5,7.5,50]

client = CAVEclient("jchen_mouse_cortex")     # dataset name
root_id = 720575941086918398                  # neuron root ID
# nrn = ...                      # meshwork with restore_properties=True
# skel = ...         # or however you build it

voxel_res = [7.5,7.5,50]

#%%
deg = np.bincount(skel.edges.flatten(), minlength=len(skel.vertices))
skel.vertex_properties["degree"] = deg
skel.vertex_properties["is_tip"] = (deg == 1)
skel.vertex_properties["is_branch"] = (deg >= 3)

#%%

l2_ids = client.chunkedgraph.get_leaves(root_id)
print("Number of L2 leaves:", len(l2_ids))

# all L2 leaves under the root (ground truth)
all_l2 = set(client.chunkedgraph.get_leaves(root_id))

# L2 IDs present in your meshwork
try:
    df_l2 = nrn.anno.lvl2_ids.df          # AnchoredAnnotation
except AttributeError:
    df_l2 = pd.DataFrame(nrn.anno.lvl2_ids)
mw_l2 = set(df_l2["lvl2_id"].astype(np.int64))  # ensure ints

only_in_mw   = mw_l2
missing_in_mw = all_l2 - mw_l2
coverage = len(mw_l2) / len(all_l2) if all_l2 else np.nan
print(f"meshwork has {len(mw_l2):,} unique L2s; root has {len(all_l2):,}. Coverage = {coverage:.3%}")


counts = [len(mw_l2), len(all_l2)]
labels = ["In meshwork", "All under root"]

plt.figure(figsize=(4,3))
plt.bar(labels, counts)
plt.ylabel("# of L2 IDs")
plt.title(f"L2 coverage (root {root_id})\nCoverage = {coverage:.2%}")
plt.show()


#%% view branch points

is_branch = (deg >= 3)
branch_idx = np.where(is_branch)[0]
branch_xyz_nm = skel.vertices[branch_idx]     

from nglui.statebuilder import ViewerState, ImageLayer, SegmentationLayer

IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"

# Convert to list[tuple] for the API
voxel_res = np.array([7.5, 7.5, 50.0])  # nm per voxel
branch_points = [tuple((p / voxel_res).astype(float)) for p in branch_xyz_nm]

img_layer = ImageLayer(source=IMAGE_SOURCE_URL)
seg_layer = SegmentationLayer().add_source(SEG_SOURCE_URL).add_segments([root_id])

viewer = (
    ViewerState()
    .add_layer(img_layer)
    .add_layer(seg_layer)
    .add_points(
        name="branch_points",
        point_column=branch_points
        # (Optional) you can also pass a per-point radius or description columns if you have them.
    )
)

link = viewer.to_link_shortener(client=client)
print(link)

#%% --- Map L2 IDs to supervoxels and assign axon/dendrite labels ---

def getting_supervoxel_id_from_nrn(nrn):
    axon_l2_ids = nrn.anno.lvl2_ids['lvl2_id']
    root_column = nrn.anno.segment_properties['is_root']
    axon_mesh_idxs = nrn.anno.is_axon["mesh_index"].to_numpy(int)
    axon_l2_id = set([axon_l2_ids[axon_ind] for axon_ind in axon_mesh_idxs])

    records = []
    for i, l2 in enumerate(axon_l2_ids):
        children = client.chunkedgraph.get_children(l2)
        is_axon = int(l2 in axon_l2_id)
        is_root = bool(root_column.iloc[i])
        for sv in children:
            records.append({"l2_id": l2, "supervoxel_id": sv,
                            "is_axon": is_axon, "is_root": is_root})

    axon_dend_df = pd.DataFrame(records)
    return axon_dend_df
    
# run this when you have two neurons



pcg_skel.features.add_volumetric_properties(nrn1, client)
pcg_skel.features.add_segment_properties(nrn1,strahler_by_compartment=True)
axon_dend_df1 = getting_supervoxel_id_from_nrn(nrn1)

pcg_skel.features.add_volumetric_properties(nrn2, client)
pcg_skel.features.add_segment_properties(nrn2,strahler_by_compartment=True)
axon_dend_df2 = getting_supervoxel_id_from_nrn(nrn2)

#%% filter syn_df and correct the synapse assignment
ROOT_ID1 = 720575941071680793
ROOT_ID2 = 720575941057622500
syn_df = client.materialize.synapse_query(
    pre_ids=[ROOT_ID1,ROOT_ID2],
    post_ids=[ROOT_ID1,ROOT_ID2],
    remove_autapses=False,
    desired_resolution=[7.5, 7.5, 50]
)
syn_df=remove_false_autapses(remove_duplicates(syn_df))

def _split_axon_dend_root(axon_dend_df):
    axon_df = axon_dend_df[axon_dend_df['is_axon']==1]
    dend_df = axon_dend_df[(axon_dend_df['is_axon'] == 0) & ~(axon_dend_df['is_root'])]
    root_df = axon_dend_df[(axon_dend_df['is_axon'] == 0) & (axon_dend_df['is_root'])]
    return axon_df,dend_df,root_df

def _find_invalid_synapses(syn_df, axon_df, dend_df):
    axon_set = set(axon_df['supervoxel_id'].astype(np.int64))
    dend_set = set(dend_df['supervoxel_id'].astype(np.int64))

    # Boolean mask
    valid_mask = (
        syn_df['pre_pt_supervoxel_id'].astype(np.int64).isin(axon_set) &
        syn_df['post_pt_supervoxel_id'].astype(np.int64).isin(dend_set)
    )

    # Rows that fail the condition
    invalid_syn = syn_df[~valid_mask]
    return invalid_syn
    
def filter_syndf(syn_df, axon_dend_df1,axon_dend_df2):
    axon_df,dend_df,root_df = _split_axon_dend_root(pd.concat([axon_dend_df1, axon_dend_df2], ignore_index=True))
    
    invalid_syn_df = _find_invalid_synapses(syn_df, axon_df, dend_df)
    
    return invalid_syn_df

invalid_syn_df=filter_syndf(syn_df, axon_dend_df1,axon_dend_df2)

