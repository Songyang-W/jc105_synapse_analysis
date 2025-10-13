#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:56:54 2025

@author: songyangwang
"""

import pandas as pd
import numpy as np
from caveclient import CAVEclient

def voxel_to_nm(coord, voxel_res=[7.5, 7.5, 50]):
    """
    Convert coordinates from voxel (pixel) space to nanometers.

    """
    return np.multiply(coord, voxel_res)

def nm_to_voxel(coord_nm, voxel_res=[7.5, 7.5, 50]):
    """
    Convert coordinates from nanometers to voxel (pixel) space.
    """
    return np.divide(coord_nm, voxel_res)

def convert_um_to_pixels(df, res_nm=(7.5, 7.5, 50)):
    """
    Convert skeleton coordinates from micrometers (µm) to voxel indices (pixels).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['x', 'y', 'z'] in µm.
    res_nm : tuple
        Resolution in nm per pixel, default (7.5, 7.5, 50).

    Returns
    -------
    pd.DataFrame
        Copy of df with new columns ['x_pix', 'y_pix', 'z_pix'] in voxel units.
    """
    df = df.copy()
    # µm → nm
    df[['x_nm','y_nm','z_nm']] = df[['x','y','z']] * 1000
    # nm → pixels
    df['x'] = df['x_nm'] / res_nm[0]
    df['y'] = df['y_nm'] / res_nm[1]
    df['z'] = df['z_nm'] / res_nm[2]
    return df

def reassign_soma_coord(df, new_xyz):
    """
    Reassigns the coordinates of the root node (parent == -1) 
    to a new soma location.

    Parameters
    ----------
    df : pandas.DataFrame
        Skeleton DataFrame with columns ['x','y','z','parent'].
    new_xyz : tuple or list of length 3
        New coordinates (x,y,z).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame (copy).
    """
    df = df.copy()
    root_mask = df['parent'] == -1
    df.loc[root_mask, ['x','y','z']] = new_xyz
    return df

table_directory = '/net/claustrum/mnt/data/Dropbox/Chen Lab Dropbox/Chen Lab Team Folder/Projects/Connectomics/Animals/jc105/LOOKUP_TABLE.xlsx'
lookuptable_df = pd.read_excel(table_directory, sheet_name="MASTER_LIST",
                   dtype={
        "SHARD ID": "string",
        "FINAL NEURON ID": "string",
        "EM NAME (Nuclear)": "string",
    })

client = CAVEclient("jchen_mouse_cortex")
root_id = 720575941071245374
sk_df = client.skeleton.get_skeleton(root_id, output_format="swc")
sk_df_pix = convert_um_to_pixels(sk_df)
row = lookuptable_df.loc[lookuptable_df['FINAL NEURON ID'] == str(root_id)].iloc[0]
soma_xyz = [int(row['Xprod']), int(row['Yprod']), int(row['Zprod'])]
updated_skel_df = reassign_soma_coord(sk_df_pix,soma_xyz)
updated_skel_df = updated_skel_df.drop(columns=['x_nm', 'y_nm', 'z_nm'])
updated_skel_df.to_csv(str(root_id)+'.csv')

