import numpy as np
import pandas as pd
from collections import deque
from scipy.spatial import cKDTree
from types import SimpleNamespace
import networkx as nx
import pcg_skel
import numpy as np
from skeleton_plot.plot_tools import plot_skel
from caveclient import CAVEclient
from meshparty import skeleton as mp_skel

#%%
table_directory = '/Users/songyangwang/Downloads/LOOKUP_TABLE.xlsx'
lookuptable_df = pd.read_excel(table_directory, sheet_name="MASTER_LIST",
                   dtype={
        "SHARD ID": "string",
        "FINAL NEURON ID": "string",
        "EM NAME (Nuclear)": "string",
    })


#%%
def nearest_vertex_idx(vertices, coord):
    """
    Return (index, distance) of the skeleton vertex nearest to coord.
    - vertices: array-like of shape (N, 3)
    - coord: iterable [x, y, z] in the SAME UNITS as `vertices`
    """
    V = np.asarray(vertices, dtype=float)
    q = np.asarray(voxel_to_nm(coord), dtype=float)
    d2 = np.sum((V - q)**2, axis=1)         # squared distances
    i = int(np.argmin(d2))                  # index of nearest vertex
    return i, float(np.sqrt(d2[i]))         # (index, Euclidean distance)

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

def set_compartment_from_indices(skel, axon_vertex_indices, axon_label=2, dend_label=3):
    """
    Write per-vertex labels into skel.vertex_properties['compartment'].
    - axon_vertex_indices: iterable of vertex indices to mark as axon (2)
      everything else becomes dendrite (3).
    """
    # get number of vertices
    V = np.asarray(getattr(skel, "vertices"))
    n = V.shape[0]

    # start all as dendrite
    comp = np.full(n, dend_label, dtype=int)

    # clean and clip indices
    ax_idx = np.asarray(axon_vertex_indices, dtype=int)
    ax_idx = ax_idx[(ax_idx >= 0) & (ax_idx < n)]
    if ax_idx.size:
        comp[np.unique(ax_idx)] = axon_label

    # write to skeleton
    if not hasattr(skel, "vertex_properties") or skel.vertex_properties is None:
        skel.vertex_properties = {}
    skel.vertex_properties["compartment"] = comp
    return comp

def label_axon_by_simple_split(skel, soma_idx, axon_idx, axon_label=2, dend_label=3, prop_name="compartment"):
    G = nx.Graph()
    edges = getattr(skel, "edges", None)
    G.add_edges_from(np.asarray(edges, dtype=int))
    path = nx.shortest_path(G, source=soma_idx, target=axon_idx)
    if len(path) < 2:
        raise ValueError("Find a further Axon Coordinate")
    split_u, split_v = path[-2], path[-1]   # <— changed line
    H = G.copy()
    if H.has_edge(split_u, split_v):
        H.remove_edge(split_u, split_v)
    else:
        # if graph stored as directed or missing that edge, ensure removal either way
        H.remove_edges_from([(split_u, split_v), (split_v, split_u)])
    comps = list(nx.connected_components(H))
    # map vertex -> component index
    comp_idx = {}
    for i, c in enumerate(comps):
        for v in c:
            comp_idx[v] = i
    axon_comp = comp_idx[axon_idx]
    n = np.asarray(skel.vertices).shape[0]
    labels = np.full(n, dend_label, dtype=int)
    for v, i in comp_idx.items():
        if i == axon_comp:
            labels[v] = axon_label
    if not hasattr(skel, "vertex_properties") or skel.vertex_properties is None:
        skel.vertex_properties = {}
    skel.vertex_properties[prop_name] = labels
    return labels, {"split_edge": (split_u, split_v), "path_len": len(path)}


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
    return skel, nrn,(l2_to_skel, skel_to_l2)

def make_is_axon_annotation(nrn, skel,
                            compartment_key="compartment",
                            axon_label=2,
                            anno_name="is_axon"):
    comp = np.asarray(skel.vertex_properties[compartment_key])
    mesh_idx = np.asarray(skel.mesh_index, dtype=np.int64)

    if comp.shape[0] != mesh_idx.shape[0]:
        raise ValueError("Length mismatch: compartments vs mesh_index")

    axon_mesh = mesh_idx[comp == axon_label]
    df = pd.DataFrame({"mesh_index": np.unique(axon_mesh)})

    # attach the dataframe itself, no wrapper
    setattr(nrn.anno, anno_name, df)

    return df

def getting_supervoxel_id_from_nrn_manual_version(nrn):
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
#%%
# TODO get soma_location from lookup table

root_id=720575941086918398
root_resolution = [7.5,7.5,50]
soma_xyz=[145981, 169840, 3432]
axon_xyz=[145516, 168285, 3419]
client = CAVEclient("jchen_mouse_cortex")
skel,nrn,(l2_to_skel, skel_to_l2) = _download_skel_identify_axon(root_id,root_resolution, client,soma_xyz)

# Now your existing projector will work:
axon_index,distance1 = nearest_vertex_idx(skel.vertices,axon_xyz)
soma_index,distance2 = nearest_vertex_idx(skel.vertices,soma_xyz)
labels, info = label_axon_by_simple_split(skel, soma_index, axon_index)

df_is_axon = make_is_axon_annotation(nrn, skel, compartment_key="compartment", axon_label=2)
pcg_skel.features.add_volumetric_properties(nrn, client)
pcg_skel.features.add_segment_properties(nrn)
plot_skel(
    skel,
    pull_compartment_colors=True,
    plot_soma=True,title=root_id
)
axon_dend_df = getting_supervoxel_id_from_nrn(nrn)

axon_dend_df.to_pickle(str(root_id)+".pkl")

#%% measure the distance

sk = skeleton.Skeleton(
    vertices=skel.vertices,
    edges=skel.edges,
    root=soma_index
)
dists = sk.distance_to_root  # numpy array: distance (nm) from soma to each vertex

l2_to_min_dist_nm = {
    l2: np.min(dists[vids]) for l2, vids in l2_to_skel.items() 
}

axon_dend_df["dist_nm"] = axon_dend_df["l2_id"].map(l2_to_min_dist_nm)


#%%
import numpy as np

def branchpoint_indices_and_coords(skel, prop_name="is_branch"):
    """
    Identify branchpoints (deg >= 3).
    Returns:
      bp_idx : array of branchpoint vertex indices
      bp_coords : (N,3) array of xyz coords (same units as skel.vertices)
      deg : full degree array
    """
    import numpy as np, networkx as nx

    G = nx.Graph()
    G.add_edges_from(np.asarray(skel.edges, dtype=int))

    deg = np.zeros(len(skel.vertices), dtype=int)
    for v, d in G.degree():
        deg[v] = d

    bp_mask = deg >= 3
    bp_idx = np.flatnonzero(bp_mask)
    bp_coords = np.asarray(skel.vertices)[bp_idx]

    # store for later use
    if not hasattr(skel, "vertex_properties") or skel.vertex_properties is None:
        skel.vertex_properties = {}
    skel.vertex_properties[prop_name] = bp_mask

    return bp_idx, bp_coords, deg

bp_idx, bp_coords, deg = branchpoint_indices_and_coords(skel)
print("branchpoint indices:", bp_idx[:5])
print("branchpoint coords:", bp_coords[:5])  # in nm


