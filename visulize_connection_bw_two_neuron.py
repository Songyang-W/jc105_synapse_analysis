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


pre_id  = 720575941088408585
post_id = 720575941153476300
link = view_connections_between_two_cells(pre_id,post_id)
link
