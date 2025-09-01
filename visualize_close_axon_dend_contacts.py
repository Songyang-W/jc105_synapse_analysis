#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 12:31:15 2025

@author: songyangwang
"""

#%% Imports & input
import numpy as np
from heapq import heappush, heappop
from collections import deque, defaultdict
from nglui.statebuilder import ViewerState, ImageLayer, SegmentationLayer
import caveclient


IN_PATH = "close_contacts_artifacts.npz"  # produced by calculation script
DATA = np.load(IN_PATH, allow_pickle=True)

root_id1 = int(DATA["root_id1"][0])
root_id2 = int(DATA["root_id2"][0])
root_resolution = tuple(DATA["root_resolution"].tolist())

skel1_vertices = DATA["skel1_vertices"]
skel2_vertices = DATA["skel2_vertices"]
skel1_edges = DATA["skel1_edges"]
skel2_edges = DATA["skel2_edges"]

pairs_12 = DATA["global_pairs_1axon_2dend"]
pairs_21 = DATA["global_pairs_2axon_1dend"]

#%% Utility: coordinate converters (public)
def nm_to_px(x_nm, res=root_resolution):
    x_nm = np.asarray(x_nm, dtype=float)
    return x_nm / np.asarray(res, dtype=float)

def px_to_nm(x_px, res=root_resolution):
    x_px = np.asarray(x_px, dtype=float)
    return x_px * np.asarray(res, dtype=float)

#%% Graph helpers (unchanged)
def build_adj_from_edges(edges):
    adj = defaultdict(list)
    for u, v in np.asarray(edges, dtype=int):
        adj[u].append(v)
        adj[v].append(u)
    return adj

def k_hop_neighborhoods(adj, seeds, k=1):
    #given a skeleton vertex, what other vertices are connected within 1 or 2 hops along the skeleton edges?
    seeds = list(map(int, set(seeds)))
    out = {s: set([s]) for s in seeds}
    for s in seeds:
        if k <= 0:
            continue
        seen = {s}
        q = deque([(s, 0)])
        while q:
            u, d = q.popleft()
            if d == k:
                continue
            for w in adj.get(u, []):
                if w not in seen:
                    seen.add(w)
                    out[s].add(w)
                    q.append((w, d + 1))
    return out
def induced_geodesic_diameter(adj, coords_nm, nodes_subset):

    nodes = list(set(map(int, nodes_subset)))
    if len(nodes) == 0:
        return 0.0, (None, None)
    allowed = set(nodes)

    # --- helper: edge length ---
    def elen(u, v):
        return float(np.linalg.norm(coords_nm[u] - coords_nm[v]))

    # --- induced degrees & internal-edge sum (count each once) ---
    deg_in = {u: 0 for u in nodes}
    internal_len = 0.0
    seen_pairs = set()
    for u in nodes:
        for v in adj.get(u, []):
            if v in allowed:
                if u < v and (u, v) not in seen_pairs:
                    internal_len += elen(u, v)
                    seen_pairs.add((u, v))
                deg_in[u] += 1

    # --- endpoints in the induced subgraph ---
    endpoints = [u for u in nodes if deg_in[u] == 1]

    # --- boundary half-extensions ---
    half_ext_total = 0.0
    if len(nodes) == 1:
        # singleton: take two shortest incident edges to neighbors outside (if they exist)
        u = nodes[0]
        outs = [elen(u, v) for v in adj.get(u, []) if v not in allowed]
        outs.sort()
        if len(outs) >= 2:
            half_ext_total += 0.5 * (outs[0] + outs[1])
        elif len(outs) == 1:
            half_ext_total += 0.5 * outs[0]
        # else: no outside neighbors → no extension
    else:
        # for each endpoint, add 0.5 * (shortest edge to any neighbor outside)
        for u in endpoints:
            outs = [elen(u, v) for v in adj.get(u, []) if v not in allowed]
            if outs:
                half_ext_total += 0.5 * min(outs)

    adjusted_len = float(internal_len + half_ext_total)

    # --- choose endpoints to return (for viz): prefer the two induced endpoints if exactly two ---
    if len(endpoints) == 2:
        return adjusted_len, (int(endpoints[0]), int(endpoints[1]))

    # otherwise fall back to farthest pair inside the induced subgraph (Dijkstra)
    def dijkstra_subset_weighted(adj, coords_nm, start, allowed):
        allowed = set(allowed)
        if start not in allowed:
            return {}
        dist = {start: 0.0}
        pq = [(0.0, start)]
        while pq:
            du, u = heappop(pq)
            if du > dist[u]:
                continue
            for v in adj.get(u, []):
                if v not in allowed:
                    continue
                w = float(np.linalg.norm(coords_nm[u] - coords_nm[v]))
                alt = du + w
                if v not in dist or alt < dist[v]:
                    dist[v] = alt
                    heappush(pq, (alt, v))
        return dist

    best_len = -1.0
    best_pair = (nodes[0], nodes[0])
    for s in nodes:
        dist = dijkstra_subset_weighted(adj, coords_nm, s, allowed)
        if not dist:
            continue
        v = max(dist, key=dist.get)
        if dist[v] > best_len:
            best_len = dist[v]
            best_pair = (s, v)

    return adjusted_len, (int(best_pair[0]), int(best_pair[1]))

#%% Cluster pairs into regions using connectivity on BOTH skeletons (unchanged)
def cluster_pairs_by_skeleton_connectivity(
    pairs, edgesA, edgesB, coordsA_nm, coordsB_nm, hop_tol_A=1, hop_tol_B=1
):
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        return []

    adjA = build_adj_from_edges(edgesA)
    adjB = build_adj_from_edges(edgesB)

    A_verts = pairs[:, 0]
    B_verts = pairs[:, 1]

    A_khop = k_hop_neighborhoods(adjA, A_verts, k=hop_tol_A)
    B_khop = k_hop_neighborhoods(adjB, B_verts, k=hop_tol_B)

    M = pairs.shape[0]
    pair_adj = [[] for _ in range(M)]
    for i in range(M):
        a_i, b_i = int(pairs[i, 0]), int(pairs[i, 1])
        A_near = A_khop.get(a_i, {a_i})
        B_near = B_khop.get(b_i, {b_i})
        for j in range(i + 1, M):
            a_j, b_j = int(pairs[j, 0]), int(pairs[j, 1])
            if (a_j in A_near) and (b_j in B_near):
                pair_adj[i].append(j)
                pair_adj[j].append(i)

    seen = np.zeros(M, dtype=bool)
    comps = []
    for i in range(M):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in pair_adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(np.array(comp, dtype=int))

    regions = []
    for rows in comps:
        A_nodes = np.unique(pairs[rows, 0])
        B_nodes = np.unique(pairs[rows, 1])

        A_diam_nm, (Au, Av) = induced_geodesic_diameter(adjA, coordsA_nm, A_nodes)
        B_diam_nm, (Bu, Bv) = induced_geodesic_diameter(adjB, coordsB_nm, B_nodes)

        regions.append({
            "pair_rows": rows,
            "A_vertices": A_nodes,
            "B_vertices": B_nodes,
            "A_diameter_nm": float(A_diam_nm),
            "B_diameter_nm": float(B_diam_nm),
            "overlap_distance_nm": float(min(A_diam_nm, B_diam_nm)),
            "A_endpoints": (int(Au) if Au is not None else None,
                            int(Av) if Av is not None else None),
            "B_endpoints": (int(Bu) if Bu is not None else None,
                            int(Bv) if Bv is not None else None),
            "n_pairs": int(len(rows)),
            "n_A_vertices": int(len(A_nodes)),
            "n_B_vertices": int(len(B_nodes)),
        })
    regions.sort(key=lambda r: r["overlap_distance_nm"], reverse=True)
    return regions

#%% Build regions for both directions
regions_12 = cluster_pairs_by_skeleton_connectivity(
    pairs_12,
    edgesA=skel1_edges, edgesB=skel2_edges,
    coordsA_nm=skel1_vertices, coordsB_nm=skel2_vertices,
    hop_tol_A=1, hop_tol_B=1
)
regions_21 = cluster_pairs_by_skeleton_connectivity(
    pairs_21,
    edgesA=skel2_edges, edgesB=skel1_edges,
    coordsA_nm=skel2_vertices, coordsB_nm=skel1_vertices,
    hop_tol_A=1, hop_tol_B=1
)

print(f"[regions] 1→2: {len(regions_12)} | 2→1: {len(regions_21)}")

#%% Region midpoint lines (centroid or endpoint-midpoint)
def regions_midpoint_lines(regions, coordsA_nm, coordsB_nm, root_resolution, method="centroid"):
    A_px, B_px = [], []
    for r in regions:
        if method == "centroid":
            A_mid_nm = coordsA_nm[np.asarray(r["A_vertices"], dtype=int)].mean(axis=0)
            B_mid_nm = coordsB_nm[np.asarray(r["B_vertices"], dtype=int)].mean(axis=0)
        elif method == "endpoints":
            Au, Av = r.get("A_endpoints", (None, None))
            Bu, Bv = r.get("B_endpoints", (None, None))
            if Au is None or Av is None or Bu is None or Bv is None:
                A_mid_nm = coordsA_nm[np.asarray(r["A_vertices"], dtype=int)].mean(axis=0)
                B_mid_nm = coordsB_nm[np.asarray(r["B_vertices"], dtype=int)].mean(axis=0)
            else:
                A_mid_nm = 0.5 * (coordsA_nm[int(Au)] + coordsA_nm[int(Av)])
                B_mid_nm = 0.5 * (coordsB_nm[int(Bu)] + coordsB_nm[int(Bv)])
        else:
            raise ValueError("method must be 'centroid' or 'endpoints'")
        A_px.append(tuple(nm_to_px(A_mid_nm, root_resolution).tolist()))
        B_px.append(tuple(nm_to_px(B_mid_nm, root_resolution).tolist()))
    return A_px, B_px

#%% One viewer with both directions + labeled layers
def build_combined_midpoint_link(
    root_id1, root_id2,
    regions_12, regions_21,
    coords1_nm, coords2_nm,
    root_resolution,
    method="centroid",
    IMAGE_SOURCE_URL="precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img",
    SEG_SOURCE_URL="graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/",
    client=None,
    color_12="gold",
    color_21="deepskyblue"
):
    A12_mid, B12_mid = regions_midpoint_lines(regions_12, coords1_nm, coords2_nm, root_resolution, method)
    A21_mid, B21_mid = regions_midpoint_lines(regions_21, coords2_nm, coords1_nm, root_resolution, method)

    v = (
        ViewerState()
        .add_layer(ImageLayer(source=IMAGE_SOURCE_URL))
        .add_layer(
            SegmentationLayer()
            .add_source(SEG_SOURCE_URL)
            .add_segments([int(root_id1), int(root_id2)])
        )
    )
    v = v.add_lines(
        name=f"{root_id1} → {root_id2} region midpoints ({method})",
        point_a_column=A12_mid, point_b_column=B12_mid, color=color_12
    )
    v = v.add_lines(
        name=f"{root_id2} → {root_id1} region midpoints ({method})",
        point_a_column=A21_mid, point_b_column=B21_mid, color=color_21
    )
    return v.to_link_shortener(client=client) if client is not None else v.to_url()

#%% Build & print link (add your CAVEclient here if you want short URLs)
client = caveclient.CAVEclient("jchen_mouse_cortex")
link_both = build_combined_midpoint_link(
    root_id1, root_id2,
    regions_12, regions_21,
    coords1_nm=skel1_vertices, coords2_nm=skel2_vertices,
    root_resolution=root_resolution,
    method="centroid",
    client=client
)
print(link_both)