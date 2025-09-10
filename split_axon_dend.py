#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:57:14 2025

@author: songyangwang
"""

from caveclient import CAVEclient
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import caveclient
import pcg_skel
import skeleton_plot as skelplot
import matplotlib.pyplot as plt

client = CAVEclient("jchen_mouse_cortex")
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

def _return_xyz_coordinates(cell_id):
    x=lookuptable_df['Xprod'][lookuptable_df['FINAL NEURON ID'] == str(cell_id)].astype(int).values[0].item()
    y=lookuptable_df['Yprod'][lookuptable_df['FINAL NEURON ID'] == str(cell_id)].astype(int).values[0].item()
    z=lookuptable_df['Zprod'][lookuptable_df['FINAL NEURON ID'] == str(cell_id)].astype(int).values[0].item()
    return [x,y,z]

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

def _download_skel_identify_axon(root_id,root_resolution, client,soma_location):
    skel, mesh, (l2_to_skel, skel_to_l2)  = pcg_skel.pcg_skeleton(
        root_id,
        client,
        return_mesh=True,
        root_point=soma_location,
        root_point_resolution=root_resolution,
        collapse_soma=True,
        collapse_radius=4500,
        return_l2dict=True,
    )
    nrn = pcg_skel.pcg_meshwork(
        root_id = root_id,
        client = client,
        root_point = soma_location,
        root_point_resolution = root_resolution,
        collapse_soma = True,
        collapse_radius = 4500,
        synapses=True,
    )
    pcg_skel.features.add_synapse_count(
        nrn,
    )
    pcg_skel.features.add_is_axon_annotation(nrn, pre_anno='pre_syn', post_anno='post_syn',
                                             annotation_name='is_axon',return_quality=True,
                                             threshold_quality=0.1)
    
    
    pcg_skel.features.add_volumetric_properties(nrn, client)
    pcg_skel.features.add_segment_properties(nrn,strahler_by_compartment=True)
    comp = build_skel_compartments_from_mesh_annos(nrn, skel, axon_anno="is_axon")
    return skel,nrn
    
#%%
table_directory = '/Users/songyangwang/Downloads/LOOKUP_TABLE.xlsx'
lookuptable_df = pd.read_excel(table_directory, sheet_name="MASTER_LIST",
                   dtype={
        "SHARD ID": "string",
        "FINAL NEURON ID": "string",
        "EM NAME (Nuclear)": "string",
    })


proofread_id = [720575941034757380, 720575941051511894, 720575941057622500, 720575941059432229, 720575941061126260, 720575941061128820, 720575941061146740, 720575941064570382, 720575941067985577, 720575941070678791, 720575941070687028, 720575941071245374, 720575941071680793, 720575941073277164, 720575941073777040, 720575941073819897, 720575941074810450, 720575941075022386, 720575941076332278, 720575941076653272, 720575941076758746, 720575941077619944, 720575941077776594, 720575941077787090, 720575941077967095, 720575941077982455, 720575941078093229, 720575941079569046, 720575941080298878, 720575941081168930, 720575941086022449, 720575941086431003, 720575941086890090, 720575941086891882, 720575941086918398, 720575941088008859, 720575941088408585, 720575941089298708, 720575941089521081, 720575941089937542, 720575941090078452, 720575941090093556, 720575941091424604, 720575941091445340, 720575941091459932, 720575941091629489, 720575941092450933, 720575941092498657, 720575941092554470, 720575941092565911, 720575941092573335, 720575941092921542, 720575941093443783, 720575941094478620, 720575941094704098, 720575941095549619, 720575941095921486, 720575941096535230, 720575941096669548, 720575941096896673, 720575941096935073, 720575941097347217, 720575941098226189, 720575941098234637, 720575941098551211, 720575941098673705, 720575941098730638, 720575941099131663, 720575941099212764, 720575941099267859, 720575941099645213, 720575941100429392, 720575941101423751, 720575941101452679, 720575941101509409, 720575941102657600, 720575941102668096, 720575941102712896, 720575941102870780, 720575941103533573, 720575941103924589, 720575941104131039, 720575941104297454, 720575941104309998, 720575941104699619, 720575941104700643, 720575941104705251, 720575941105379336, 720575941106152505, 720575941106477402, 720575941106600928, 720575941106639968, 720575941107285320, 720575941107298120, 720575941107437825, 720575941109570314, 720575941109605752, 720575941113287333, 720575941114006852, 720575941114011460, 720575941114493115, 720575941114493371, 720575941114494139, 720575941114687139, 720575941114711203, 720575941114933738, 720575941115332394, 720575941115960573, 720575941116548740, 720575941116572664, 720575941116663435, 720575941116976341, 720575941116997589, 720575941117000149, 720575941118374002, 720575941118374514, 720575941120382861, 720575941120702062, 720575941120904478, 720575941121184239, 720575941123725826, 720575941123736578, 720575941124436677, 720575941125743543, 720575941125953181, 720575941126229860, 720575941127918527, 720575941128195568, 720575941128216130, 720575941128500170, 720575941128608670, 720575941128703583, 720575941131576319, 720575941131588607, 720575941132405960, 720575941134134113, 720575941136000282, 720575941139252669, 720575941139510444, 720575941143312378, 720575941144379311, 720575941145828823, 720575941148267857, 720575941149954781, 720575941149993181, 720575941150910648, 720575941152155190, 720575941152796324, 720575941153471868, 720575941153561801, 720575941153566409, 720575941154431375, 720575941154453647, 720575941156114821, 720575941157944107, 720575941158042818, 720575941162298010, 720575941174542567, 720575941183694464, 720575941190025408, 720575941239851589]
syn_proof_only_df = client.materialize.synapse_query(pre_ids=proofread_id,
                                                  post_ids=proofread_id,
                                                  remove_autapses=True,
                                                  desired_resolution=[7.5,7.5,50]
                                                 )

#%% failed but has axon

failed_ids=[720575941059432229, 720575941061146740, 720575941070678791, 720575941070687028, 720575941071245374, 720575941073277164, 720575941073819897, 720575941075022386, 720575941077619944, 720575941077776594, 720575941077787090, 720575941077967095, 720575941077982455, 720575941078093229, 720575941080298878, 720575941081168930, 720575941086431003, 720575941086891882, 720575941086918398, 720575941088008859, 720575941089298708, 720575941089937542, 720575941090078452, 720575941090093556, 720575941091424604, 720575941091445340, 720575941091459932, 720575941091629489, 720575941092565911, 720575941092573335, 720575941092921542, 720575941095549619, 720575941097347217, 720575941098673705, 720575941099212764, 720575941099267859, 720575941099645213, 720575941101423751, 720575941101452679, 720575941102657600, 720575941102668096, 720575941102870780, 720575941105379336, 720575941106477402, 720575941107298120, 720575941109605752, 720575941113287333, 720575941114006852, 720575941114011460, 720575941114493371, 720575941114711203, 720575941114933738, 720575941115332394, 720575941116572664, 720575941116997589, 720575941117000149, 720575941118374002, 720575941123725826, 720575941125743543, 720575941128608670, 720575941139510444, 720575941144379311, 720575941149954781, 720575941149993181, 720575941152155190, 720575941153471868, 720575941153561801, 720575941154453647, 720575941183694464]

#%%
for root_id in failed_ids:
    soma_location = _return_xyz_coordinates(root_id)
    
    # skel = pcg_skel.pcg_skeleton(root_id=root_id, client=client)
    
    root_resolution = [7.5,7.5,50]
    
    try:
        skel,nrn = _download_skel_identify_axon(root_id,root_resolution, client,soma_location)
        # f, ax = plt.subplots(figsize=(7, 10))
        # skelplot.plot_tools.plot_skel(
        #     skel,
        #     line_width=1,
        #     plot_soma=True,
        #     invert_y=True,
        # )
        syn_count_df = nrn.anno.synapse_count.df
        
        skel_df = pcg_skel.features.aggregate_property_to_skeleton(
            nrn,
            'synapse_count',
            agg_dict={'num_syn_in': 'sum', 'num_syn_out': 'sum', 'net_size_in': 'sum', 'net_size_out': 'sum'},
        )
        
        
        # Now you can plot:
        from skeleton_plot.plot_tools import plot_skel
        
        plot_skel(
            skel,
            pull_compartment_colors=True,
            plot_soma=True,title=root_id
        )
        axon_dend_df = getting_supervoxel_id_from_nrn(nrn)
        
        axon_dend_df.to_pickle(str(root_id)+".pkl")
    except Exception as e:
        print(f"❌ Failed for root_id {root_id}: {e}")
