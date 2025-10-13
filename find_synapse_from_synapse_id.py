#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 21:02:51 2025

@author: songyangwang
"""
import numpy as np
import pandas as pd
import networkx as nx
import pcg_skel
from skeleton_plot.plot_tools import plot_skel
from caveclient import CAVEclient
from meshparty import skeleton as mp_skel
client = CAVEclient("jchen_mouse_cortex")

synapse_ids = [83398293, 55833048, 69676970, 41991265, 83581117, 63853090, 45588477, 83340195, 66906827, 92989087, 85349891, 76615672, 50652853, 78595511, 66906899, 52210119, 58751547, 86996476]
#%%
synapse_info = client.annotation.get_annotation(table_name='synapses', annotation_ids = synapse_ids,)
#%%
from nglui.statebuilder import *
presynapse_coordinates = [tuple(s["pre_pt_position"]) for s in synapse_info_7p5]
postsynapse_coordinates = [tuple(s["post_pt_position"]) for s in synapse_info_7p5]

synapse_ids = [s["id"] for s in synapse_info]

IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"

# Define layers
img_layer = ImageLayer(source=IMAGE_SOURCE_URL)

seg_layer = (
    SegmentationLayer()
    .add_source(SEG_SOURCE_URL)
)

# Add lines + annotations
viewer = (
    ViewerState()
    .add_layer(img_layer)
    .add_layer(seg_layer)
    .add_lines(
        name='synapses',
        point_a_column=presynapse_coordinates,
        point_b_column=postsynapse_coordinates,
    )
)

# Generate sharable Neuroglancer link
link = viewer.to_link_shortener(client=client)
print("Neuroglancer link:", link)