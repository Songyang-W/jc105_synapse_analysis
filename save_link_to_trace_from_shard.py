#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:01:55 2025

@author: songyangwang
"""

#%%

import argparse 
import numpy as np
import pandas as pd
import networkx as nx
import pcg_skel
from skeleton_plot.plot_tools import plot_skel
from caveclient import CAVEclient
from meshparty import skeleton as mp_skel
from nglui.statebuilder import *

client = CAVEclient("jchen_mouse_cortex")

#table_directory = '/net/claustrum/mnt/data/Dropbox/Chen Lab Dropbox/Chen Lab Team Folder/Projects/Connectomics/Animals/jc105/LOOKUP_TABLE.xlsx'
#saving_directory = '/net/claustrum/mnt/data/Dropbox/Chen Lab Dropbox/Chen Lab Team Folder/Projects/Connectomics/Animals/jc105/skeleton_synapse/'
table_directory = '/Users/songyangwang/Downloads/LOOKUP_TABLE.xlsx'
lookuptable_df = pd.read_excel(table_directory, sheet_name="MASTER_LIST",
                   dtype={
        "SHARD ID": "string",
        "FINAL NEURON ID": "string",
        "EM NAME (Nuclear)": "string",
        "SHARD ID": "string"
    })

#%%

lookuptable_df['ROOT ID'] = lookuptable_df['SHARD ID'].apply(
    lambda sid: str(client.chunkedgraph.get_root_id(sid)) if pd.notna(sid) else None
)

IMAGE_SOURCE_URL = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img"
SEG_SOURCE_URL   = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"
 

#%%
def view_img_segmentation(root_id, nuc_id,
                          image_source=IMAGE_SOURCE_URL,
                          seg_source=SEG_SOURCE_URL):
    """
    Create a Neuroglancer link showing the image, one segmentation layer,
    and an empty annotation layer. The segmentation layer is named with nuc_id.
    """
    # Base image
    img_layer = ImageLayer(source=image_source)
    
    # Segmentation layer named by nuc_id
    seg_layer = (
        SegmentationLayer(name=str(nuc_id))
        .add_source(seg_source)
        .add_segments([int(root_id)])
    )
    
    # Empty annotation layer
    anno_layer = AnnotationLayer(name="annotations")
    
    # Build viewer state
    vs = ViewerState().add_layer([img_layer, seg_layer, anno_layer])
    
    # Return shortened link
    return vs.to_link_shortener(client=client)


# Example:
link = view_img_segmentation(root_id,nuc_id)

#%%

def add_links_to_df(df):
    def generate_link(row):
        # If WEBLINK already exists, keep it
        if pd.notna(row['WEBLINK TO CONSENSUS ID']) and row['WEBLINK TO CONSENSUS ID'] != "":
            return row['WEBLINK TO CONSENSUS ID']
        
        # Skip if missing root_id or nuc_id
        if pd.isna(row['ROOT ID']) or pd.isna(row['EM NAME (Nuclear)']):
            return None
        
        # Build link
        try:
            return view_img_segmentation(
                root_id=row['ROOT ID'],
                nuc_id=row['EM NAME (Nuclear)']
            )
        except Exception as e:
            print(f"Error generating link for row {row.name}: {e}")
            return None
    
    # Apply row-wise
    df['WEBLINK TO CONSENSUS ID'] = df.apply(generate_link, axis=1)
    return df

# Usage:
lookuptable_df = add_links_to_df(lookuptable_df)

#%%
lookuptable_df.to_excel("lookuptable_updated.xlsx", index=False)
