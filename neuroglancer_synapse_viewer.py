#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 21:40:31 2025
This script uses the CAVEclient and Neuroglancer statebuilder tools to query, 
visualize, and generate a shareable link for synaptic connectivity data from 
the jchen_mouse_cortex dataset, showing both input and output connections.

@author: songyangwang
"""

from caveclient import CAVEclient
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from nglui.statebuilder import *

# Initialize client
client = CAVEclient("jchen_mouse_cortex")

# Example neuron root ID
example_root_id = 720575941080167279

# Query for input and output synapses
input_syn_df = client.materialize.synapse_query(
    post_ids=example_root_id,
    remove_autapses=True,
    desired_resolution=[7.5, 7.5, 50]
)

output_syn_df = client.materialize.synapse_query(
    pre_ids=example_root_id,
    remove_autapses=True,
    desired_resolution=[7.5, 7.5, 50]
)

# Extract coordinates
input_pre_coords = [tuple(coord) for coord in input_syn_df["pre_pt_position"].values]
input_post_coords = [tuple(coord) for coord in input_syn_df["post_pt_position"].values]

output_pre_coords = [tuple(coord) for coord in output_syn_df["pre_pt_position"].values]
output_post_coords = [tuple(coord) for coord in output_syn_df["post_pt_position"].values]

# Define Neuroglancer layers
IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"

img_layer = ImageLayer(source=IMAGE_SOURCE_URL)

seg_layer = (
    SegmentationLayer()
    .add_source(SEG_SOURCE_URL)
    .add_segments([example_root_id])
)

# Build the viewer with both input and output synapse line layers
viewer_state = (
    ViewerState()
    .add_layer(img_layer)
    .add_layer(seg_layer)
    .add_lines(
        name='input_synapses',
        point_a_column=input_pre_coords,
        point_b_column=input_post_coords,
        color='blue'  # input connections in blue
    )
    .add_lines(
        name='output_synapses',
        point_a_column=output_pre_coords,
        point_b_column=output_post_coords,
        color='red'   # output connections in red
    )
)

# Generate and print shareable link
viewer_link = viewer_state.to_link_shortener(client=client)
print("Neuroglancer link showing both input & output synapses:")
print(viewer_link)