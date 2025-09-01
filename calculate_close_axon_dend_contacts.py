#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 12:30:35 2025

@author: songyangwang
"""

#%% Imports & config
import numpy as np  # used in build_skel_compartments_from_mesh_annos
import caveclient
import pcg_skel
import skeleton_plot as skelplot
import matplotlib.pyplot as plt

# ---- edit these for your run ----
DATASTACK = "jchen_mouse_cortex"
ROOT_ID1 = 720575941071680793
SOMA1 = [168103, 158705, 2257]
ROOT_ID2 = 720575941057622500
SOMA2 = [170885, 158942, 3651]
ROOT_RESOLUTION = (7.5, 7.5, 50)
COLLAPSE_RADIUS = 7500
AXON_QUALITY_THRESH = 0.2
DIST_THRESHOLD_NM = 5000.0
MAX_PAIRS_PER_DIR = 5000
OUT_PATH = "close_contacts_artifacts.npz"   # saved for visualization script

client = caveclient.CAVEclient(DATASTACK)

#%% Utility: coordinate converters (public)
def nm_to_px(x_nm, res=ROOT_RESOLUTION):
    x_nm = np.asarray(x_nm, dtype=float)
    return x_nm / np.asarray(res, dtype=float)

def px_to_nm(x_px, res=ROOT_RESOLUTION):
    x_px = np.asarray(x_px, dtype=float)
    return x_px * np.asarray(res, dtype=float)

#%% Mesh→skeleton compartment projection (unchanged)
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

    Returns: np.ndarray[int] with labels in {2='axon', 3='basal', 4='apical', default_non_axon}.
    Also writes skel.vertex_properties['compartment'].
    """
    def _mesh_bool_from_anno(name):
        if name is None or not hasattr(nrn.anno, name):
            return None
        df = getattr(nrn.anno, name).df  # expects a column 'mesh_index'
        if "mesh_index" not in df.columns:
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

    if not hasattr(skel, "mesh_index") or skel.mesh_index is None:
        raise ValueError("skel.mesh_index is required to project mesh annotations to skeleton.")
    sk_mi = np.asarray(skel.mesh_index, dtype=np.int64)

    comp = np.full(len(sk_mi), default_non_axon, dtype=int)
    if axon_mesh:
        comp[np.isin(sk_mi, list(axon_mesh))] = 2
    if apical_mesh:
        comp[np.isin(sk_mi, list(apical_mesh))] = 4
    if basal_mesh:
        comp[np.isin(sk_mi, list(basal_mesh))] = 3

    comp = np.where(comp == None, default_non_axon, comp)  # noqa: E711
    skel.vertex_properties["compartment"] = comp
    return comp

#%% Build two neurons & annotate axon (unchanged)
def build_two_neurons_with_is_axon(
    root_id1,
    soma_location1,
    root_id2,
    soma_location2,
    client,
    root_resolution=ROOT_RESOLUTION,
    collapse_radius=COLLAPSE_RADIUS,
    threshold_quality=AXON_QUALITY_THRESH,
):
    """
    Build (skel, nrn) for each neuron and annotate axon via is_axon heuristic.
    Returns: skel1, nrn1, comp1, skel2, nrn2, comp2
    """
    skel1, mesh1, (l2_to_skel1, skel_to_l21) = pcg_skel.pcg_skeleton(
        root_id1,
        client,
        return_mesh=True,
        root_point=soma_location1,
        root_point_resolution=root_resolution,
        collapse_soma=True,
        collapse_radius=collapse_radius,
        return_l2dict=True,
    )

    nrn1 = pcg_skel.pcg_meshwork(
        root_id=root_id1,
        client=client,
        root_point=soma_location1,
        root_point_resolution=root_resolution,
        collapse_soma=True,
        collapse_radius=collapse_radius,
        synapses=True,
    )
    pcg_skel.features.add_synapse_count(nrn1)
    pcg_skel.features.add_is_axon_annotation(
        nrn1,
        pre_anno="pre_syn",
        post_anno="post_syn",
        annotation_name="is_axon",
        return_quality=True,
        threshold_quality=threshold_quality,
    )

    skel2, mesh2, (l2_to_skel2, skel_to_l22) = pcg_skel.pcg_skeleton(
        root_id2,
        client,
        return_mesh=True,
        root_point=soma_location2,
        root_point_resolution=root_resolution,
        collapse_soma=True,
        collapse_radius=collapse_radius,
        return_l2dict=True,
    )

    nrn2 = pcg_skel.pcg_meshwork(
        root_id=root_id2,
        client=client,
        root_point=soma_location2,
        root_point_resolution=root_resolution,
        collapse_soma=True,
        collapse_radius=collapse_radius,
        synapses=True,
    )
    pcg_skel.features.add_synapse_count(nrn2)
    pcg_skel.features.add_is_axon_annotation(
        nrn2,
        pre_anno="pre_syn",
        post_anno="post_syn",
        annotation_name="is_axon",
        return_quality=True,
        threshold_quality=threshold_quality,
    )

    comp1 = build_skel_compartments_from_mesh_annos(nrn1, skel1, axon_anno="is_axon")
    comp2 = build_skel_compartments_from_mesh_annos(nrn2, skel2, axon_anno="is_axon")

    # Temporary color preview (unchanged behavior)
    skel1.vertex_properties = dict(getattr(skel1, "vertex_properties", {}) or {})
    skel1.vertex_properties["compartment"] = comp1
    skelplot.plot_tools.plot_skel(skel1, line_width=1, plot_soma=True, invert_y=True)
    del skel1.vertex_properties["compartment"]

    skel2.vertex_properties = dict(getattr(skel2, "vertex_properties", {}) or {})
    skel2.vertex_properties["compartment"] = comp2
    skelplot.plot_tools.plot_skel(skel2, line_width=1, plot_soma=True, invert_y=True)
    del skel2.vertex_properties["compartment"]

    return skel1, nrn1, comp1, skel2, nrn2, comp2

#%% Pair finding (unchanged core, includes KD-tree + fallback)
from scipy.spatial import cKDTree

def _find_pairs_within_threshold(A_nm, B_nm, thr_nm, use_kdtree=True):
    """
    Return (pairs, dists) where pairs[:,0] are indices in A, pairs[:,1] in B,
    for all pairs within Euclidean distance <= thr_nm. Works in nm space.
    """
    if A_nm.size == 0 or B_nm.size == 0:
        return np.empty((0,2), dtype=int), np.empty((0,), dtype=float)

    if use_kdtree:
        try:
            tree = cKDTree(B_nm)
            neigh = tree.query_ball_point(A_nm, r=thr_nm)
            pairs = []
            dists = []
            for i, js in enumerate(neigh):
                if not js:
                    continue
                a = A_nm[i]
                bsel = B_nm[np.asarray(js, dtype=int)]
                ds = np.linalg.norm(bsel - a, axis=1)
                dists.extend(ds.tolist())
                gi = np.full(len(js), i, dtype=int)
                gj = np.asarray(js, dtype=int)
                pairs.append(np.stack([gi, gj], axis=1))
            if pairs:
                pairs = np.concatenate(pairs, axis=0)
                dists = np.asarray(dists, dtype=float)
            else:
                pairs = np.empty((0,2), dtype=int)
                dists = np.empty((0,), dtype=float)
            return pairs, dists
        except Exception:
            pass

    # NumPy fallback
    thr2 = float(thr_nm) ** 2
    A2 = np.sum(A_nm*A_nm, axis=1, keepdims=True)
    B2 = np.sum(B_nm*B_nm, axis=1, keepdims=True).T
    AB = A_nm @ B_nm.T
    dist2 = A2 + B2 - 2.0 * AB
    np.maximum(dist2, 0.0, out=dist2)
    ai, bj = np.where(dist2 <= thr2)
    if ai.size == 0:
        return np.empty((0,2), dtype=int), np.empty((0,), dtype=float)
    pairs = np.stack([ai, bj], axis=1)
    dists = np.sqrt(dist2[ai, bj], dtype=float)
    return pairs, dists

#%% Compute close axon↔dend pairs + build meta (unchanged logic)
def compute_close_pairs_and_meta(
    root_id1, verts1_nm, comp1,
    root_id2, verts2_nm, comp2,
    threshold_nm=DIST_THRESHOLD_NM,
    root_resolution=ROOT_RESOLUTION,
    max_pairs_per_dir=MAX_PAIRS_PER_DIR,
    dendrite_labels=(3, 4),
):
    axon_label = 2
    verts1_nm = np.asarray(verts1_nm, dtype=float)
    verts2_nm = np.asarray(verts2_nm, dtype=float)
    comp1 = np.asarray(comp1)
    comp2 = np.asarray(comp2)

    mask1_ax = (comp1 == axon_label)
    mask1_dn = np.isin(comp1, dendrite_labels)
    mask2_ax = (comp2 == axon_label)
    mask2_dn = np.isin(comp2, dendrite_labels)

    v1_ax_nm = verts1_nm[mask1_ax]
    v1_dn_nm = verts1_nm[mask1_dn]
    v2_ax_nm = verts2_nm[mask2_ax]
    v2_dn_nm = verts2_nm[mask2_dn]

    pairs_1ax_2dn, dists_12 = _find_pairs_within_threshold(v1_ax_nm, v2_dn_nm, threshold_nm)
    pairs_2ax_1dn, dists_21 = _find_pairs_within_threshold(v2_ax_nm, v1_dn_nm, threshold_nm)

    def _cap_pairs(pairs, dists):
        if pairs.shape[0] <= max_pairs_per_dir:
            return pairs, dists
        idx = np.linspace(0, pairs.shape[0]-1, max_pairs_per_dir, dtype=int)
        return pairs[idx], (dists[idx] if dists is not None and dists.size else dists)

    pairs_1ax_2dn, dists_12 = _cap_pairs(pairs_1ax_2dn, dists_12)
    pairs_2ax_1dn, dists_21 = _cap_pairs(pairs_2ax_1dn, dists_21)

    idx1_all = np.arange(verts1_nm.shape[0])
    idx2_all = np.arange(verts2_nm.shape[0])
    idx1_ax = idx1_all[mask1_ax]
    idx1_dn = idx1_all[mask1_dn]
    idx2_ax = idx2_all[mask2_ax]
    idx2_dn = idx2_all[mask2_dn]

    global_pairs_12 = np.column_stack([idx1_ax[pairs_1ax_2dn[:,0]],
                                       idx2_dn[pairs_1ax_2dn[:,1]]]).astype(int)
    global_pairs_21 = np.column_stack([idx2_ax[pairs_2ax_1dn[:,0]],
                                       idx1_dn[pairs_2ax_1dn[:,1]]]).astype(int)

    meta = {
        "n_pairs_1axon_2dend": int(global_pairs_12.shape[0]),
        "n_pairs_2axon_1dend": int(global_pairs_21.shape[0]),
        "threshold_nm": float(threshold_nm),
        "resolution_nm_per_pixel": tuple(float(x) for x in root_resolution),
        # arrays below saved explicitly in .npz
    }
    return meta, global_pairs_12, global_pairs_21, dists_12, dists_21

#%% Main: build, compute, save artifacts for visualization
if __name__ == "__main__":
    skel1, nrn1, comp1, skel2, nrn2, comp2 = build_two_neurons_with_is_axon(
        ROOT_ID1, SOMA1, ROOT_ID2, SOMA2, client,
        root_resolution=ROOT_RESOLUTION,
        collapse_radius=COLLAPSE_RADIUS,
        threshold_quality=AXON_QUALITY_THRESH,
    )

    meta, gp12, gp21, d12, d21 = compute_close_pairs_and_meta(
        ROOT_ID1, skel1.vertices, comp1,
        ROOT_ID2, skel2.vertices, comp2,
        threshold_nm=DIST_THRESHOLD_NM,
        root_resolution=ROOT_RESOLUTION,
        max_pairs_per_dir=MAX_PAIRS_PER_DIR
    )

    # Save minimal bridge so the viz script doesn't recompute heavy steps
    # (vertices + edges + global pairs + dists + scalar meta)
    np.savez(
        OUT_PATH,
        root_id1=np.array([ROOT_ID1], dtype=np.int64),
        root_id2=np.array([ROOT_ID2], dtype=np.int64),
        root_resolution=np.array(ROOT_RESOLUTION, dtype=float),
        skel1_vertices=skel1.vertices,
        skel2_vertices=skel2.vertices,
        skel1_edges=skel1.edges,
        skel2_edges=skel2.edges,
        global_pairs_1axon_2dend=gp12,
        global_pairs_2axon_1dend=gp21,
        dists_1axon_2dend_nm=d12,
        dists_2axon_1dend_nm=d21,
        n_pairs_1axon_2dend=np.array([meta["n_pairs_1axon_2dend"]], dtype=int),
        n_pairs_2axon_1dend=np.array([meta["n_pairs_2axon_1dend"]], dtype=int),
        threshold_nm=np.array([meta["threshold_nm"]], dtype=float),
    )
    print(f"[saved] {OUT_PATH}")