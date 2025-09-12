import numpy as np
import pandas as pd
import networkx as nx
import pcg_skel
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
        collapse_radius=4000,
        return_l2dict=True,
    )
    nrn = pcg_skel.pcg_meshwork(
        root_id = root_id,
        client = client,
        root_point = soma_location,
        root_point_resolution = root_resolution,
        collapse_soma = True,
        collapse_radius = 4000,
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

root_id=720575941086918398
root_resolution = [7.5,7.5,50]
soma_xyz=[int(x.strip()) for x in lookuptable_df['soma coord'][lookuptable_df['FINAL NEURON ID']==str(root_id)].dropna().iloc[0].split(',')]
axon_xyz=[int(x.strip()) for x in lookuptable_df['axon coord'][lookuptable_df['FINAL NEURON ID']==str(root_id)].dropna().iloc[0].split(',')]
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
axon_dend_df = getting_supervoxel_id_from_nrn_manual_version(nrn)

#axon_dend_df.to_pickle(str(root_id)+".pkl")


#%% functions to calculate distance
from collections import defaultdict

def build_l2_neighbors(skel, nrn):
    lvl2_df = nrn.anno.lvl2_ids
    # skeleton vertices -> mesh indices -> L2 IDs
    skel_to_mesh = np.asarray(skel.mesh_index, dtype=int)
    mesh_to_l2   = np.asarray(lvl2_df["lvl2_id"], dtype=np.int64)
    skel_to_l2 = mesh_to_l2[skel_to_mesh]
    l2_neighbors = defaultdict(set)
    for u, v in skel.edges:
        l2_u = int(skel_to_l2[u])
        l2_v = int(skel_to_l2[v])
        if l2_u != l2_v:  # only connect different L2s
            l2_neighbors[l2_u].add(l2_v)
            l2_neighbors[l2_v].add(l2_u)
    return {k: list(vs) for k, vs in l2_neighbors.items()}


def nearest_l2_neighbor_by_sv(sv_coord, nbrs_coords: pd.Series, voxel_res=(7.5, 7.5, 50.0)):
    """
    sv_coord_nm : [x_nm, y_nm, z_nm]
    nbrs_coords : pandas.Series, index = l2_id, values = [x_vx, y_vx, z_vx] (voxel coords)
    returns     : (closest_l2_id, distance_voxels, distances_series)
    """
    if len(nbrs_coords) == 0:
        raise ValueError("nbrs_coords is empty")

    sv_vx = voxel_to_nm(np.array(sv_coord, dtype=float))   # ensure it's a NumPy array

    coords = voxel_to_nm(np.vstack([np.asarray(c, float) for c in nbrs_coords.values]))
    dists = np.linalg.norm(coords - sv_vx[None, :], axis=1)
    # Pack distances back to a Series aligned with l2_ids
    dist_ser = pd.Series(dists, index=nbrs_coords.index, name="euclid_dist_vox")
    closest_id = dist_ser.idxmin()
    return closest_id, float(dist_ser.loc[closest_id]), dist_ser.sort_values()

def closer_to_soma(l2_a, l2_b, l2_to_soma_dist_nm):
    d_a = l2_to_soma_dist_nm.get(l2_a, np.inf)
    d_b = l2_to_soma_dist_nm.get(l2_b, np.inf)
    if d_a < d_b:
        return l2_a, d_a
    else:
        return l2_b, d_b
    
def _project_point_to_segment(P, A, B, eps=1e-12):
    """
    Project point P onto the segment AB (all 3D, in nm).
    Returns (P_proj, t_clamped) where P_proj = A + t*(B-A), t in [0,1].
    """
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom < eps:
        # A and B are the same point
        return A.copy(), 0.0
    t = float(np.dot(P - A, AB) / denom)
    t_clamped = max(0.0, min(1.0, t))
    return A + t_clamped * AB, t_clamped

def _find_closest_l2(
    sv_coord,
    l2_id_current,
    l2_centroids,        # dict {l2_id: np.array([x,y,z])} in nm
    l2_neighbors,           # dict {l2_id: [neighbor_l2_ids]}
    l2_to_soma_dist_nm      # dict {l2_id: float} (min geodesic dist to soma, nm)
):

    # --- inputs & guards
    nbrs = list(l2_neighbors.get(l2_id_current, []))
    nbrs_coords = l2_centroids[nbrs]
    closest_id, d_closest, dist_table = nearest_l2_neighbor_by_sv(sv_coord, nbrs_coords)
    return closest_id

def synapse_geodesic_to_soma(
    sv_coord_nm,
    sv_id,
    l2_centroids,
    l2_centroids_nm,     # dict {l2_id: np.array([x,y,z])} in nm
    l2_to_soma_dist_nm,                # np.array, sk.distance_to_root in nm
    l2_to_skel,axon_dend_df=axon_dend_df,soma_xyz=soma_xyz
):
    """
    Compute:
      1) perpendicular distance from synapse (nm) to segment between L2_a and L2_b centroids
      2) geodesic distance (nm) from soma to the nearest skeleton vertex to that intersection,
         restricted to vertices belonging to L2_a or L2_b.

    Returns dict with:
      - 'intersection_point_nm' (3,)
      - 'perpendicular_distance_nm' (float)
      - 'nearest_vertex_index' (int)
      - 'geodesic_to_soma_nm' (float)
      - 't_on_segment' (float in [0,1])
    """
    if axon_dend_df['is_root'][axon_dend_df['supervoxel_id']==sv_id].item():
        return float(np.linalg.norm(sv_coord_nm-voxel_to_nm(soma_xyz) ))
    l2_id_current = axon_dend_df['l2_id'][axon_dend_df['supervoxel_id']==sv_id]
    l2_a = l2_id_current.item()
    l2_b = _find_closest_l2(nm_to_voxel(sv_coord_nm),l2_a,l2_centroids,l2_neighbors,l2_to_soma_dist_nm)
    # coordinates (nm)
    P = np.asarray(sv_coord_nm, dtype=float)
    cand_a = l2_to_skel[l2_a]
    cand_b = l2_to_skel[l2_b]

    A = skel.vertices[cand_a]
    B = skel.vertices[cand_b]

    # 1) project onto segment AB
    P_proj, t_seg = _project_point_to_segment(P, A, B)
    perp_dist = float(np.linalg.norm(P - P_proj))
    
    closer_id, dist = closer_to_soma(l2_a, l2_b, l2_to_soma_dist_nm)
    dist_proj_l2vertice = float(np.linalg.norm(l2_centroids_nm[closer_id]-P_proj))
    geom_to_soma = l2_to_soma_dist_nm[closer_id]
    total_length = perp_dist+dist_proj_l2vertice+geom_to_soma

    return total_length
    

#%% measure the coarse purely from l2 distance

sk = mp_skel.Skeleton(
    vertices=skel.vertices,
    edges=skel.edges,
    root=soma_index
)
dists = sk.distance_to_root  # numpy array: distance (nm) from soma to each vertex

l2_to_soma_dist_nm = {
    l2: np.min(dists[vids]) for l2, vids in l2_to_skel.items() 
}



#%% finer measurement

verts_vx = np.array([nm_to_voxel(v) for v in nrn.mesh.vertices])
l2_ids = np.array(nrn.anno.lvl2_ids["lvl2_id"], dtype=np.int64)
# make dataframe
df = pd.DataFrame({"l2_id": l2_ids, "coord": list(verts_vx)})
df_nm = pd.DataFrame({"l2_id": l2_ids, "coord": list(voxel_to_nm(verts_vx))})

# group by l2_id and take centroid
l2_coord_table = df.groupby("l2_id")["coord"].apply(
    lambda coords: np.mean(np.stack(coords), axis=0)
)
l2_coord_table_nm = df_nm.groupby("l2_id")["coord"].apply(
    lambda coords: np.mean(np.stack(coords), axis=0)
)
l2_neighbors = build_l2_neighbors(skel, nrn)
l2_centroids_nm=l2_coord_table_nm

#%% filter the axon_dend_df and reassign l2

def reassign_l2(l2_id, l2_neighbors, l2_centroids_nm):
    """
    Return a non-empty neighbor list for l2_id.
    - If graph neighbors exist -> return them.
    - Else -> return [closest_l2_by_centroid] among L2s that appear in the graph.
    """
    nbrs = l2_neighbors.get(int(l2_id), [])
    if nbrs:
        return l2_id

    # fallback: nearest by centroid (nm) among L2s present in the graph
    if int(l2_id) not in l2_centroids_nm.index:
        raise KeyError(f"L2 {l2_id} not found in centroids.")

    P = np.asarray(l2_centroids_nm.loc[int(l2_id)], float)

    # candidates = L2s that have at least one graph neighbor entry
    candidates = [k for k in l2_neighbors.keys() if k != int(l2_id)]
    if not candidates:
        # last resort: use all centroids except itself
        candidates = [int(k) for k in l2_centroids_nm.index if int(k) != int(l2_id)]

    C = l2_centroids_nm.loc[candidates]
    d = C.apply(lambda q: np.linalg.norm(np.asarray(q, float) - P))
    return [int(d.idxmin())]

# Ensure types are consistent
l2_centroids_nm.index = l2_centroids_nm.index.astype(np.int64)
l2_neighbors = {int(k): list(map(int, v)) for k, v in l2_neighbors.items()}

# Wrapper to make your function always return an int
def _reassign_l2_int(l2_id):
    res = reassign_l2(int(l2_id), l2_neighbors, l2_centroids_nm)
    return int(res if isinstance(res, (int, np.integer)) else res[0])

# Apply to all rows so every supervoxel binds to an L2 that has at least one neighbor
axon_dend_df["l2_id"] = axon_dend_df["l2_id"].apply(_reassign_l2_int)




#%% working on synapse labelling
import string
def build_l2_graph_and_centroids_nm(skel, nrn):
    """Returns G (L2 graph, undirected), l2_centroids_nm (pd.Series index=l2_id -> [x,y,z])."""
    lvl2_df = nrn.anno.lvl2_ids
    skel_to_mesh = np.asarray(skel.mesh_index, dtype=int)
    mesh_to_l2   = np.asarray(lvl2_df["lvl2_id"], dtype=np.int64)
    skel_to_l2   = mesh_to_l2[skel_to_mesh]

    # Graph: connect L2s that meet along skeleton edges
    G = nx.Graph()
    for u, v in skel.edges:
        lu, lv = int(skel_to_l2[u]), int(skel_to_l2[v])
        if lu != lv:
            G.add_edge(lu, lv)

    # Centroids (nm): mean of mesh vertices per L2
    verts_nm = np.asarray(nrn.mesh.vertices, float)
    l2_ids   = np.asarray(lvl2_df["lvl2_id"], np.int64)
    df_nm    = pd.DataFrame({"l2_id": l2_ids, "coord": list(verts_nm)})
    l2_centroids_nm = df_nm.groupby("l2_id")["coord"].apply(lambda cs: np.mean(np.stack(cs), axis=0))

    return G, l2_centroids_nm
def find_root_l2(soma_index, skel, nrn):
    lvl2_df = nrn.anno.lvl2_ids
    skel_to_mesh = np.asarray(skel.mesh_index, dtype=int)
    mesh_to_l2   = np.asarray(lvl2_df["lvl2_id"], dtype=np.int64)
    skel_to_l2   = mesh_to_l2[skel_to_mesh]
    return int(skel_to_l2[int(soma_index)])

def _child_order_at_branch(G, node, l2_centroids_nm, parent=None):
    """
    Determine a deterministic order of children around 'node' using 3D angle sorting.
    If 'parent' is provided, exclude it and treat remaining as 'children' (BFS orientation).
    """
    ctr = np.asarray(l2_centroids_nm.loc[node], float)
    nbrs = list(G.neighbors(node))
    if parent is not None:
        nbrs = [n for n in nbrs if n != parent]

    if not nbrs:
        return []

    # Vectors from node to neighbors; sort by azimuth in XY plane as a simple, stable rule
    vecs = {n: np.asarray(l2_centroids_nm.loc[n], float) - ctr for n in nbrs}
    angs = {n: np.arctan2(v[1], v[0]) for n, v in vecs.items()}
    ordered = sorted(nbrs, key=lambda n: angs[n])
    return ordered

def label_branch_children(G, root_l2, l2_centroids_nm):
    """
    Walk a BFS tree from root_l2. At each branch node (deg>=3), label outgoing child edges A,B,C,...
    Returns a dict mapping directed edges (parent -> child) to a letter, e.g. edge_label[(u,v)] = 'A'.
    """
    # BFS tree to define parent/child orientation
    T = nx.bfs_tree(G, root_l2)
    edge_label = {}

    for node in T.nodes():
        children = list(T.successors(node))  # neighbors one step farther from root
        if len(children) >= 2:
            # Restart alphabet here; sort children deterministically
            ordered_children = _child_order_at_branch(G, node, l2_centroids_nm, parent=None)
            # Keep only those that are actual children in BFS tree
            ordered_children = [c for c in ordered_children if c in children]
            for i, child in enumerate(ordered_children):
                edge_label[(node, child)] = LETTERS[i % len(LETTERS)]
        elif len(children) == 1:
            # Pass-through (deg==2 in tree): no letters assigned here
            pass

    return edge_label, T

def synapse_branch_code(G, edge_label, T, root_l2, sv_id, axon_dend_df):
    """
    Returns:
      - code (str): e.g. 'A1B2D'
      - steps (list of tuples): [(branch_node, child, letter, branch_idx), ...]
      - n_branches (int)
    """
    synapse_l2 = axon_dend_df.loc[axon_dend_df['supervoxel_id'] == sv_id, 'l2_id'].item()
    if synapse_l2 not in T:
        path = nx.shortest_path(G, root_l2, synapse_l2)
    else:
        path = nx.shortest_path(T, root_l2, synapse_l2)

    code = []
    steps = []
    branch_idx = 1
    for u, v in zip(path[:-1], path[1:]):
        letter = edge_label.get((u, v), None)
        if letter is not None:
            code.append(f"{letter}{branch_idx}")
            steps.append((u, v, letter, branch_idx))
            branch_idx += 1

    return "".join(code), steps, len(code)


LETTERS = list(string.ascii_uppercase)  # A, B, C, ...

G, l2_centroids_nm = build_l2_graph_and_centroids_nm(skel, nrn)
root_l2 = find_root_l2(soma_index, skel, nrn)
edge_label, T = label_branch_children(G, root_l2, l2_centroids_nm)


#%% load to synapse matrix
syn_in_df = client.materialize.synapse_query(
                                                  post_ids=root_id,
                                                  remove_autapses=True,
                                                  desired_resolution=[7.5,7.5,50]
                                                 )

syn_out_df = client.materialize.synapse_query(
                                                  pre_ids=root_id,
                                                  remove_autapses=True,
                                                  desired_resolution=[7.5,7.5,50]
                                                 )

#%%

# --- Enrich syn_in_df with distance, location code, and label (with try/except) ---

sv_to_l2   = axon_dend_df.set_index("supervoxel_id")["l2_id"].to_dict()
sv_is_axon = axon_dend_df.set_index("supervoxel_id")["is_axon"].to_dict()
sv_is_root = axon_dend_df.set_index("supervoxel_id")["is_root"].to_dict()

dist_vals = []
loc_codes = []
labels    = []

for r in syn_in_df.itertuples(index=False):
    try:
        sv_id = int(r.post_pt_supervoxel_id)
        sv_coord_nm = voxel_to_nm(r.post_pt_position)
        l2_id_current = int(sv_to_l2[sv_id])

        out = synapse_geodesic_to_soma(
            sv_coord_nm,
            sv_id,
            l2_coord_table,
            l2_coord_table_nm,     # dict {l2_id: np.array([x,y,z])} in nm
            l2_to_soma_dist_nm,                # np.array, sk.distance_to_root in nm
            l2_to_skel,axon_dend_df
        )
        dist_vals.append(out)

        code, _, _ = synapse_branch_code(G, edge_label, T, root_l2,sv_id,axon_dend_df)
        loc_codes.append(code)

        if sv_is_root.get(sv_id, False):
            labels.append("soma")
        else:
            labels.append("axon" if sv_is_axon.get(sv_id, 0) == 1 else "dendrite")

    except Exception as e:
        dist_vals.append(np.nan)
        loc_codes.append(None)
        labels.append("unknown")
        print(f"Warning: failed on SV {getattr(r,'post_pt_supervoxel_id',None)}: {e}")

syn_in_df["post_pt_to_soma_distance_nm"] = dist_vals
syn_in_df["post_pt_location"] = loc_codes
syn_in_df["post_pt_label"] = labels

dist_vals = []
loc_codes = []
labels    = []

for r in syn_out_df.itertuples(index=False):
    try:
        sv_id = int(r.pre_pt_supervoxel_id)
        sv_coord_nm = voxel_to_nm(r.pre_pt_position)
        l2_id_current = int(sv_to_l2[sv_id])

        out = synapse_geodesic_to_soma(
            sv_coord_nm,
            sv_id,
            l2_coord_table,
            l2_coord_table_nm,
            l2_to_soma_dist_nm,
            l2_to_skel,
            axon_dend_df
        )
        dist_vals.append(out)

        code, _, _ = synapse_branch_code(G, edge_label, T, root_l2, sv_id, axon_dend_df)
        loc_codes.append(code)

        if sv_is_root.get(sv_id, False):
            labels.append("soma")
        else:
            labels.append("axon" if sv_is_axon.get(sv_id, 0) == 1 else "dendrite")

    except Exception as e:
        dist_vals.append(np.nan)
        loc_codes.append(None)
        labels.append("unknown")
        print(f"Warning: failed on SV {getattr(r,'pre_pt_supervoxel_id',None)}: {e}")

syn_out_df["pre_pt_to_soma_distance_nm"] = dist_vals
syn_out_df["pre_pt_location"]            = loc_codes
syn_out_df["pre_pt_label"]               = labels

common_cols = list(set(syn_in_df.columns) & set(syn_out_df.columns))

syn_all = pd.concat([syn_in_df, syn_out_df], join="outer", ignore_index=True)
syn_all.to_csv(str(root_id)+".csv", index=False)
