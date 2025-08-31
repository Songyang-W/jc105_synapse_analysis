#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:59:19 2025
based on
https://www.caveconnecto.me/pcg_skel/reference/features/#pcg_skel.features.add_is_axon_annotation
@author: songyangwang
"""

import caveclient
import pcg_skel
import skeleton_plot as skelplot
import matplotlib.pyplot as plt


datastack = 'jchen_mouse_cortex'
client = caveclient.CAVEclient(datastack)

root_id = 720575941086918398
soma_location = [145964, 169824, 3455]

# skel = pcg_skel.pcg_skeleton(root_id=root_id, client=client)

root_resolution = [7.5,7.5,50]

skel, mesh, (l2_to_skel, skel_to_l2)  = pcg_skel.pcg_skeleton(
    root_id,
    client,
    return_mesh=True,
    root_point=soma_location,
    root_point_resolution=root_resolution,
    collapse_soma=True,
    collapse_radius=7500,
    return_l2dict=True,
)

f, ax = plt.subplots(figsize=(7, 10))
skelplot.plot_tools.plot_skel(
    skel,
    line_width=1,
    plot_soma=True,
    invert_y=True,
)

nrn = pcg_skel.pcg_meshwork(
    root_id = root_id,
    client = client,
    root_point = soma_location,
    root_point_resolution = root_resolution,
    collapse_soma = True,
    collapse_radius = 7500,
    synapses=True,
)

pcg_skel.features.add_synapse_count(
    nrn,
)
#%%
pcg_skel.features.add_is_axon_annotation(nrn, pre_anno='pre_syn', post_anno='post_syn',
                                         annotation_name='is_axon',return_quality=True,
                                         threshold_quality=0.3)



syn_count_df = nrn.anno.synapse_count.df


skel_df = pcg_skel.features.aggregate_property_to_skeleton(
    nrn,
    'synapse_count',
    agg_dict={'num_syn_in': 'sum', 'num_syn_out': 'sum', 'net_size_in': 'sum', 'net_size_out': 'sum'},
)

#%%
pcg_skel.features.add_volumetric_properties(nrn, client)
pcg_skel.features.add_segment_properties(nrn,strahler_by_compartment=True)



#%%
import numpy as np
import pandas as pd

def build_skel_compartments_from_mesh_annos(
    nrn,
    skel,
    axon_anno="is_axon",
    apical_anno=None,
    basal_anno=None,
    default_non_axon=3,
):
    """
    Project mesh-level annotations to a full per-skeleton-vertex 'compartment' label array.

    Returns: np.ndarray[str] of shape (n_vertices,) with labels in {'axon','apical','basal',default_non_axon}.
    Also writes skel.vertex_properties['compartment'].
    """
    # 1) map mesh_index -> label(s)
    def _mesh_bool_from_anno(name):
        if name is None or not hasattr(nrn.anno, name):
            return None
        df = getattr(nrn.anno, name).df  # expects a column 'mesh_index'
        if "mesh_index" not in df.columns:
            # sometimes it's 'mesh_ind' in other tables; try to be flexible
            col = "mesh_ind" if "mesh_ind" in df.columns else None
            if col is None:
                raise ValueError(f"{name} has no mesh_index/mesh_ind column.")
            idxs = df[col].to_numpy()
        else:
            idxs = df["mesh_index"].to_numpy()
        return set(np.asarray(idxs, dtype=np.int64))

    axon_mesh = _mesh_bool_from_anno(axon_anno)
    apical_mesh = _mesh_bool_from_anno(apical_anno) if apical_anno else None
    basal_mesh = _mesh_bool_from_anno(basal_anno) if basal_anno else None

    # 2) skeleton vertex -> mesh index
    if not hasattr(skel, "mesh_index") or skel.mesh_index is None:
        raise ValueError("skel.mesh_index is required to project mesh annotations to skeleton.")
    sk_mi = np.asarray(skel.mesh_index, dtype=np.int64)

    # 3) start with default everywhere
    comp = np.full(len(sk_mi), default_non_axon, dtype=int)

    # 4) assign labels by membership of the mapped mesh index
    if axon_mesh:
        comp[np.isin(sk_mi, list(axon_mesh))] = 2
    if apical_mesh:
        comp[np.isin(sk_mi, list(apical_mesh))] = 4
    if basal_mesh:
        comp[np.isin(sk_mi, list(basal_mesh))] = 3

    # 5) (optional) ensure no leftover Nones
    comp = np.where(comp == None, default_non_axon, comp)  # noqa: E711

    # write back for plotting
    skel.vertex_properties["compartment"] = comp
    return comp

#%%
comp = build_skel_compartments_from_mesh_annos(nrn, skel, axon_anno="is_axon")
# Now you can plot:
from skeleton_plot.plot_tools import plot_skel

plot_skel(
    skel,
    pull_compartment_colors=True,
    plot_soma=True,
)


#%%

import numpy as np
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

import matplotlib.pyplot as plt

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

