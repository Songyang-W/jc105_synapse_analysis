# jc105 Synapse Analysis

Pairwise axon–dendrite overlap and synapse connectivity analysis for the `jchen_mouse_cortex` dataset (MICrONS-style jc105 experiment).

---

## Key Data Files

| File | Description |
|---|---|
| `cleaned_synapse_table.csv` | Main synapse table. One row per synapse. Contains `pre/post_pt_root_id`, voxel positions (7.5×7.5×50 nm), geodesic soma distances (nm), branch-tree location codes (`pre/post_pt_location`), compartment labels (`pre/post_pt_label`), and cell-type labels (`pre/post_pt_type`). |
| `axon_dend_overlap_matrix_5.csv` | Square matrix of pairwise axon–dendrite co-travel length. **Row = pre (axon donor), Column = post (dendrite target)**. Values in **µm** (5 µm contact threshold). **Warning: 29 root IDs are duplicated in both rows and columns — deduplicate before use.** |
| `axon_dend_overlap_table.csv` | Long-format version of the overlap matrix. Columns: `pre_pt_root_id`, `post_pt_root_id`, `overlap_length_nm`. **The column name is a known bug: values are in µm, not nm.** |
| `neuron_pairs_with_freq_labeled.csv` | Derived pair-level summary (only pairs with lookup coverage). Columns: `pre/post_pt_root_id`, `overlap_length_nm` (µm, same naming bug), `n_synapses`, `synapse_freq` (synapses/µm), `pre/post_functional_category`, `pre/post_func_type`. |
| `LOOKUP_TABLE.xlsx` | Sheet `MASTER_LIST`: one row per neuron. Key columns: `FINAL NEURON ID` (root ID, string), `Cell Type`, `finer label`, `Functional Category`, `2P NAME`, soma/nuclear/axon coordinates. |

---

## Main Scripts

### Data generation (CAVE / network required)

| Script | Purpose |
|---|---|
| `manual_split_axon_dend_clean.py` | Downloads skeletons from CAVE, splits axon vs. dendrite by graph cut, assigns branch-tree codes and geodesic distances to each synapse. Produces per-neuron CSV files that are merged into `cleaned_synapse_table.csv`. |
| `manual_split_axon_dendrite_individual.py` | Per-neuron version of the above; run per root ID. The actual data-generation workhorse. |
| `overlap_matrix_calculate.py` | Computes the pairwise axon–dendrite overlap matrix using kD-tree (5 µm threshold) + NetworkX region grouping. Outputs `axon_dend_overlap_matrix.csv` / `axon_dend_overlap_matrix_5.csv`. |
| `shard_to_root_mapper.py` | Maps LOOKUP_TABLE shard IDs to current root IDs via CAVE chunkedgraph. |
| `proofread_connectivity.py` | Queries CAVE for pairwise synapse counts among proofread neurons; builds and visualizes synapse connectivity matrix. |

### Local analysis (no CAVE required)

| Script | Purpose |
|---|---|
| `merge_overlap_synapse_lookup.py` | Merges overlap table + synapse table + LOOKUP_TABLE. Produces synapse-count confusion matrices by cell type and functional category. **Good starting point for understanding the merged data.** |
| `function_structure_matrix_corr.py` | Correlates synapse connectivity with 2P functional tuning (requires `.mat` file + CAVE). Spectral clustering of connectivity. |
| `analysis_overlap_synapse.py` | **NEW.** Main analysis script (12 Spyder `#%%` blocks). Covers: overlap vs. connection existence, overlap vs. synapse count, synapse density by type/FC pair, directional asymmetry, soma-distance targeting, branch-code analysis, outlier pairs. No CAVE required. |
| `analysis_overlap_synapse_utils.py` | **NEW.** Reusable helper functions for the above: data loading, overlap matrix cleaning, pair-table construction, branch-code parsing, lookup joins, statistics, and plotting. |

### Visualization

| Script | Purpose |
|---|---|
| `visualize_close_axon_dend_contacts.py` | Generates Neuroglancer annotation layers for overlap regions between a pair of neurons. |
| `visulize_connection_bw_two_neuron.py` | Neuroglancer link showing synapses + overlap regions between two specific neurons. |
| `bound_box_view.py` | Bounding-box viewer utility. |

---

## Which File to Run for Which Purpose

| Goal | Run |
|---|---|
| Re-generate `cleaned_synapse_table.csv` from scratch | `manual_split_axon_dend_clean.py` → `manual_split_axon_dendrite_individual.py` (per neuron, needs CAVE) |
| Re-generate overlap matrices | `overlap_matrix_calculate.py` (needs skeleton `.npz` files + CAVE) |
| Understand merged data + basic confusion matrices | `merge_overlap_synapse_lookup.py` |
| Full overlap–structure–synapse analysis (recommended entry point) | `analysis_overlap_synapse.py` (local only, Spyder `#%%` blocks) |
| View overlaps in Neuroglancer for a specific pair | `visualize_close_axon_dend_contacts.py` |

---

## Known Caveats and Assumptions

1. **Unit naming bug.** `overlap_length_nm` in `axon_dend_overlap_table.csv` and `neuron_pairs_with_freq_labeled.csv` contains values in **µm**, not nm. The column has been renamed to `overlap_length_um` in `analysis_overlap_synapse.py`.

2. **Duplicate root IDs in overlap matrix.** `axon_dend_overlap_matrix_5.csv` has 29 root IDs that appear twice as both row labels and column labels (likely from double entries in a prior LOOKUP_TABLE version). Deduplicate on load by keeping the first occurrence. `analysis_overlap_synapse_utils.py` handles this automatically.

3. **Root IDs must be loaded as strings.** Root IDs are 18-digit integers. They overflow float64 and convert to scientific notation if read as numeric. Always use `dtype=str` when loading CSVs. Never cast to `int` for merging.

4. **Overlap is directional.** `matrix[i, j]` = µm of neuron i's axon within 5 µm of neuron j's dendrite. `matrix[i, j] ≠ matrix[j, i]` in general. Do not symmetrize.

5. **Lookup coverage is partial.** `neuron_pairs_with_freq_labeled.csv` (3,339 pairs) covers only pairs where both neurons have a `FINAL NEURON ID` entry in LOOKUP_TABLE. The full overlap table has ~38,000 pairs. Analyses requiring `Functional Category` are restricted to the covered subset.

6. **Synapse positions are in voxel space.** Resolution is `[7.5, 7.5, 50]` nm per voxel. Use `coord_nm = coord_voxel * [7.5, 7.5, 50]` to convert.

7. **PV cells (ChC, BC) special case.** In `overlap_matrix_calculate.py`, PV-type neurons use the full skeleton (not just dendrite) for the overlap target, because PV axons wrap around the axon initial segment. Keep this in mind when interpreting overlap for PV post-synaptic neurons.
