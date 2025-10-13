#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:17:15 2025

@author: songyangwang
"""

from caveclient import CAVEclient
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from nglui.statebuilder import *
client = CAVEclient("jchen_mouse_cortex")



def view_connections_between_two_cells(pre_id,post_id):
    "TODO change color, change transparent level"
    
    syn_df = client.materialize.synapse_query(post_ids=post_id,pre_ids=pre_id,
                                              remove_autapses=True,
                                              desired_resolution=[7.5,7.5,50])
    presynapse_coordinates = [tuple(coord) for coord in syn_df["pre_pt_position"].values]
    postsynapse_coordinates = [tuple(coord) for coord in syn_df["post_pt_position"].values]
    
    syn_df2 = client.materialize.synapse_query(post_ids=pre_id,pre_ids=post_id,
                                              remove_autapses=False,
                                              desired_resolution=[7.5,7.5,50])
    presynapse_coordinates2 = [tuple(coord) for coord in syn_df2["pre_pt_position"].values]
    postsynapse_coordinates2 = [tuple(coord) for coord in syn_df2["post_pt_position"].values]

    
    IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
    SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"
    
    img_layer=ImageLayer(source=IMAGE_SOURCE_URL)

    seg_layer = (
        SegmentationLayer()
        .add_source(SEG_SOURCE_URL)
        .add_segments([pre_id,post_id]))

    link = (
        ViewerState()
        .add_layer(img_layer)
        .add_layer(seg_layer)
        .add_lines(
            name=f"{pre_id} --> {post_id}",
            point_a_column=presynapse_coordinates,
            point_b_column=postsynapse_coordinates,
            color='tomato'
        )
        .add_lines(
            name=f"{post_id} --> {pre_id}",
            point_a_column=presynapse_coordinates2,
            point_b_column=postsynapse_coordinates2,
            color='green'
        )
    ).to_link_shortener(client=client)
    return link


pre_id  = 720575941071680793
post_id = 720575941057622500
link = view_connections_between_two_cells(pre_id,post_id)
link

#%%
def view_connections_between_two_cells_from_dfs(syn_df,syn_df2):
    "TODO change color, change transparent level"
    
    presynapse_coordinates = [tuple(coord) for coord in syn_df["pre_pt_position"].values]
    postsynapse_coordinates = [tuple(coord) for coord in syn_df["post_pt_position"].values]
    
    
    presynapse_coordinates2 = [tuple(coord) for coord in syn_df2["pre_pt_position"].values]
    postsynapse_coordinates2 = [tuple(coord) for coord in syn_df2["post_pt_position"].values]

    pre_id,post_id=np.unique(pd.concat(
        [invalid_syn_df['pre_pt_root_id'], invalid_syn_df['post_pt_root_id']], 
        axis=1
    ))
    
    IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
    SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"
    
    img_layer=ImageLayer(source=IMAGE_SOURCE_URL)

    seg_layer = (
        SegmentationLayer()
        .add_source(SEG_SOURCE_URL)
        .add_segments([pre_id,post_id]))

    link = (
        ViewerState()
        .add_layer(img_layer)
        .add_layer(seg_layer)
        .add_lines(
            name=f"reversed assignment",
            point_a_column=presynapse_coordinates,
            point_b_column=postsynapse_coordinates,
            color='tomato'
        )
        .add_lines(
            name=f"original detection",
            point_a_column=presynapse_coordinates2,
            point_b_column=postsynapse_coordinates2,
            color='green'
        )
    ).to_link_shortener(client=client)
    return link

link = view_connections_between_two_cells_from_dfs(invalid_syn_df,syn_df)
print(link)

#%%
IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"
 

def view_axon_dendrite_points(skel, root_id,
                              image_source=IMAGE_SOURCE_URL,
                              seg_source=SEG_SOURCE_URL,
                              axon_color="tomato",
                              dend_color="dodgerblue"):
    """
    Build a Neuroglancer link with two point-annotation layers:
      - one for axon (compartment==2)
      - one for dendrite (compartment==3)
    No line annotations are added.
    """
    # Pull vertices and compartment labels
    verts = np.asarray(skel.vertices)
    comps = np.asarray(skel.vertex_properties["compartment"])
    verts = verts / np.array([7.5, 7.5, 50.0])

    # Split into axon/dendrite point lists (x, y, z)
    axon_pts   = [tuple(p) for p in verts[comps == 2]]
    dend_pts   = [tuple(p) for p in verts[comps == 3]]

    # Base layers
    img_layer = ImageLayer(source=image_source)
    seg_layer = (
        SegmentationLayer()
        .add_source(seg_source)
        .add_segments([int(root_id)])
    )

    # Build viewer with two point layers
    vs = (
        ViewerState()
        .add_layer(img_layer)
        .add_layer(seg_layer)
        .add_points(
            name=f"{root_id} — axon (n={len(axon_pts)})",
            point_column=axon_pts,
            color=axon_color
        )
        .add_points(
            name=f"{root_id} — dendrite (n={len(dend_pts)})",
            point_column=dend_pts,
            color=dend_color
        )
    ).to_link_shortener(client=client)

    # Title/position is optional; uncomment to set the viewer title
    # vs = vs.set_state({"layout": "4panel", "title": str(root_id)})
    return vs

# Example:
link = view_axon_dendrite_points(skel, root_id)
print(link)

#%%

# --- Config you already have ---
IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"
VOXEL_RES = np.array([7.5, 7.5, 50.0], dtype=float)

# --- Neuroglancer helpers (your wrapper) ---
# from your environment:
# from nglui.statebuilder import ViewerState, ImageLayer, SegmentationLayer

def _unwrap_seg_props(nrn):
    """Return segment_properties as a pandas DataFrame."""
    sp = nrn.anno.segment_properties
    # common attributes across versions
    if hasattr(sp, "df"):
        return sp.df
    if hasattr(sp, "dataframe"):
        return sp.dataframe
    if hasattr(sp, "to_dataframe"):
        return sp.to_dataframe()
    raise AttributeError("Could not unwrap nrn.anno.segment_properties to a DataFrame.")

def _segprops_mesh_to_strahler(seg_props_df):
    """
    Build dict: mesh_index(int) -> strahler(int).
    Prefers 'mesh_ind_filt' if present, else falls back to 'mesh_ind'.
    """
    if "strahler" not in seg_props_df.columns:
        raise KeyError("segment_properties missing 'strahler' column.")

    mesh_col = "mesh_ind_filt" if "mesh_ind_filt" in seg_props_df.columns else "mesh_ind"
    if mesh_col not in seg_props_df.columns:
        raise KeyError("segment_properties missing 'mesh_ind' / 'mesh_ind_filt' column.")

    mesh_to_strahler = {}
    for _, row in seg_props_df.iterrows():
        s = int(row["strahler"])
        inds = row[mesh_col]
        if inds is None:
            continue
        # inds can be list/array-like of mesh vertex indices
        for mi in np.atleast_1d(inds):
            mesh_to_strahler[int(mi)] = s
    return mesh_to_strahler

def _branch_vertex_mask(skel):
    """Return boolean mask of skeleton vertices that are branch points (degree >= 3)."""
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(np.asarray(skel.edges, dtype=int))
    deg = np.array([G.degree(i) for i in range(len(skel.vertices))])
    return deg >= 2

def view_strahler_branch_points(skel, nrn, root_id,
                                image_source=IMAGE_SOURCE_URL,
                                seg_source=SEG_SOURCE_URL):
    """
    Build a Neuroglancer link with five point-annotation layers, one per Strahler order (1..5),
    showing ONLY branch vertices (degree >= 3). Points are in voxel coords.
    """
    # unwrap seg props and build mesh->strahler lookup
    seg_props = _unwrap_seg_props(nrn)
    mesh_to_strahler = _segprops_mesh_to_strahler(seg_props)

    # skeleton vertices (nm) -> to voxel
    verts_nm = np.asarray(skel.vertices, dtype=float)
    verts_vx = (verts_nm / VOXEL_RES).astype(float)

    # branch mask
    br_mask = _branch_vertex_mask(skel)
    br_idx = np.where(br_mask)[0]

    # map branch vertices -> strahler via mesh_index
    skel_to_mesh = np.asarray(skel.mesh_index, dtype=int)
    br_mesh_idx = skel_to_mesh[br_idx]
    br_strahler = [mesh_to_strahler.get(int(mi), None) for mi in br_mesh_idx]

    # bucketize points by strahler 1..5
    order_to_points = {k: [] for k in [1, 2, 3, 4, 5]}
    for idx, s in zip(br_idx, br_strahler):
        if s in order_to_points:
            order_to_points[s].append(tuple(verts_vx[idx]))

    # base layers
    img_layer = ImageLayer(source=image_source)
    seg_layer = SegmentationLayer().add_source(seg_source).add_segments([int(root_id)])

    vs = ViewerState().add_layer(img_layer).add_layer(seg_layer)

    # add one points layer per order
    # (colors are optional; you can omit to use defaults)
    order_colors = {
        1: "lime",
        2: "gold",
        3: "deepskyblue",
        4: "violet",
        5: "red"
    }
    for order in [1, 2, 3, 4, 5]:
        pts = order_to_points[order]
        if len(pts) == 0:
            continue
        vs = vs.add_points(
            name=f"{root_id} — Strahler {order} (branch pts: n={len(pts)})",
            data=pts,
            color=order_colors.get(order, "white")
        )

    # optional: set layout/title
    # vs = vs.set_state({"layout": "4panel", "title": f"{root_id} Strahler branches"})

    return vs.to_link_shortener(client=client)

# Example usage:
link = view_strahler_branch_points(skel, nrn, root_id)
print(link)


#%%
# --- Synapse visualization ---
def view_synapses_with_location(syn_df, root_id,
                                image_source=IMAGE_SOURCE_URL,
                                seg_source=SEG_SOURCE_URL):
    """
    Neuroglancer link with a single point-annotation layer for all synapses.
    Each point shows its branch-code location (post_pt_location) as description.
    """
    # base layers
    img_layer = ImageLayer(source=image_source)
    seg_layer = SegmentationLayer().add_source(seg_source).add_segments([int(root_id)])
    vs = ViewerState().add_layer(img_layer).add_layer(seg_layer)

    # collect points & descriptions
    pts   = [tuple(np.asarray(p, float)) for p in syn_df["post_pt_position"]]
    descs = syn_df["post_pt_location"].astype(str).tolist()

    vs = vs.add_points(
        name=f"{root_id} — synapses (n={len(pts)})",
        data=pd.DataFrame({"pt": pts, "desc": descs}),
        point_column="pt",
        description_column="desc",
        color="yellow"
    )

    return vs.to_link_shortener(client=client)

# Example:
link = view_synapses_with_location(syn_in_df, root_id)
print(link)


#%%

def view_pre_post_synapse_on_one_cell(cell_id):
    "TODO change color, change transparent level"
    
    dend_syn_df = client.materialize.synapse_query(post_ids=cell_id,
                                              remove_autapses=True,
                                              desired_resolution=[7.5,7.5,50])
    axon_syn_df = client.materialize.synapse_query(pre_ids=cell_id,
                                              remove_autapses=True,
                                              desired_resolution=[7.5,7.5,50])
    
    dend_coord_in = [tuple(coord) for coord in dend_syn_df["pre_pt_position"].values]
    dend_coord_out = [tuple(coord) for coord in dend_syn_df["post_pt_position"].values]
    
    
    axon_coord_in = [tuple(coord) for coord in axon_syn_df["pre_pt_position"].values]
    axon_coord_out = [tuple(coord) for coord in axon_syn_df["post_pt_position"].values]

    
    IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
    SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"
    
    img_layer=ImageLayer(source=IMAGE_SOURCE_URL)

    seg_layer = (
        SegmentationLayer()
        .add_source(SEG_SOURCE_URL)
        .add_segments([cell_id]))

    link = (
        ViewerState()
        .add_layer(img_layer)
        .add_layer(seg_layer)
        .add_lines(
            name=f"input",
            point_a_column=dend_coord_in,
            point_b_column=dend_coord_out,
            color='tomato'
        )
        .add_lines(
            name=f"output",
            point_a_column=axon_coord_in,
            point_b_column=axon_coord_out,
            color='green'
        )
    ).to_link_shortener(client=client)
    return link


cell_id  = 720575941174542567
link = view_pre_post_synapse_on_one_cell(cell_id)
link
