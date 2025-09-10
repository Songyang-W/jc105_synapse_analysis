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
                                              remove_autapses=False,
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