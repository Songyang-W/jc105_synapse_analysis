#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:40:33 2025

@author: songyangwang
merge lookup table and cave for connectivity table (since we haven't uploaded CAVE table')
include more properties: such as inhibitory vs excitatory
"""

from caveclient import CAVEclient
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd



client = CAVEclient("jchen_mouse_cortex")

table_directory = '/Users/songyangwang/Downloads/LOOKUP_TABLE.xlsx'
lookuptable_df = pd.read_excel(table_directory, sheet_name="MASTER_LIST",
                   dtype={
        "SHARD ID": "string",
        "FINAL NEURON ID": "string",
        "EM NAME (Nuclear)": "string",
    })

pre_post_synapse_count_directory = '/Users/songyangwang/repos/jc105_synapse_analysis/synapse_data/pre_post_synapse_count.csv'
pre_post_df = pd.read_csv(pre_post_synapse_count_directory)
proofread_id = [720575941034757380, 720575941051511894, 720575941057622500, 720575941059432229, 720575941061126260, 720575941061128820, 720575941061146740, 720575941064570382, 720575941067985577, 720575941070687028, 720575941071245374, 720575941071680793, 720575941073277164, 720575941073777040, 720575941073819897, 720575941074810450, 720575941075022386, 720575941076332278, 720575941076653272, 720575941076758746, 720575941077619944, 720575941077776594, 720575941077787090, 720575941077967095, 720575941077982455, 720575941078093229, 720575941079569046, 720575941080298878, 720575941081168930, 720575941086022449, 720575941086431003, 720575941086890090, 720575941086891882, 720575941086918398, 720575941088008859, 720575941088408585, 720575941089298708, 720575941089521081, 720575941089937542, 720575941090078452, 720575941090093556, 720575941091424604, 720575941091445340, 720575941091459932, 720575941091629489, 720575941092498657, 720575941092554470, 720575941092565911, 720575941092573335, 720575941092598423, 720575941092921542, 720575941093443783, 720575941094478620, 720575941094704098, 720575941095549619, 720575941095921486, 720575941096535230, 720575941096669548, 720575941096896673, 720575941096935073, 720575941097347217, 720575941098226189, 720575941098234637, 720575941098551211, 720575941098673705, 720575941098730638, 720575941099131663, 720575941099212764, 720575941099267859, 720575941099645213, 720575941100429392, 720575941101423751, 720575941101452679, 720575941101509409, 720575941102657600, 720575941102668096, 720575941102712896, 720575941102870780, 720575941103533573, 720575941103924589, 720575941104131039, 720575941104297454, 720575941104309998, 720575941104699619, 720575941104700643, 720575941104705251, 720575941105379336, 720575941106152505, 720575941106477402, 720575941106600928, 720575941106639968, 720575941107285320, 720575941107298120, 720575941107437825, 720575941107829776, 720575941109570314, 720575941109605752, 720575941113287333, 720575941114006852, 720575941114011460, 720575941114493115, 720575941114493371, 720575941114494139, 720575941114687139, 720575941114711203, 720575941114933738, 720575941115332394, 720575941115960573, 720575941116548740, 720575941116572664, 720575941116663435, 720575941116976341, 720575941116997589, 720575941117000149, 720575941118374002, 720575941118374514, 720575941120382861, 720575941120702062, 720575941120904478, 720575941121184239, 720575941123725826, 720575941123736578, 720575941124436677, 720575941125743543, 720575941125953181, 720575941126229860, 720575941127918527, 720575941128195568, 720575941128216130, 720575941128500170, 720575941128608670, 720575941128703583, 720575941131576319, 720575941131588607, 720575941132405960, 720575941134134113, 720575941136000282, 720575941139252669, 720575941139510444, 720575941143312378, 720575941144379311, 720575941145828823, 720575941148267857, 720575941149954781, 720575941149993181, 720575941150910648, 720575941152155190, 720575941152796324, 720575941153471868, 720575941153561801, 720575941153566409, 720575941154431375, 720575941154453647, 720575941156114821, 720575941157944107, 720575941158042818, 720575941162298010, 720575941174542567, 720575941183694464, 720575941190025408, 720575941239851589]
selected_ids=[720575941034757380, 720575941051511894, 720575941057622500, 720575941059432229, 720575941061126260, 720575941061128820, 720575941061146740, 720575941064570382, 720575941067985577, 720575941070687028, 720575941071245374, 720575941071680793, 720575941073277164, 720575941073777040, 720575941073819897, 720575941074810450, 720575941075022386, 720575941076332278, 720575941076653272, 720575941076758746, 720575941077619944, 720575941077776594, 720575941077787090, 720575941077967095, 720575941077982455, 720575941078093229, 720575941079569046, 720575941080298878, 720575941081168930, 720575941086022449, 720575941086431003, 720575941086890090, 720575941086891882, 720575941086918398, 720575941088008859, 720575941088408585, 720575941089298708, 720575941089521081, 720575941089937542, 720575941090078452, 720575941090093556, 720575941091424604, 720575941091445340, 720575941091459932, 720575941091629489, 720575941092498657, 720575941092554470, 720575941092565911, 720575941092573335, 720575941092598423, 720575941092921542, 720575941093443783, 720575941094478620, 720575941094704098, 720575941095549619, 720575941095921486, 720575941096535230, 720575941096669548, 720575941096896673, 720575941096935073, 720575941097347217, 720575941098226189, 720575941098234637, 720575941098551211, 720575941098673705, 720575941098730638, 720575941099131663, 720575941099212764, 720575941099267859, 720575941099645213, 720575941100429392, 720575941101423751, 720575941101452679, 720575941101509409, 720575941102657600, 720575941102668096, 720575941102712896, 720575941102870780, 720575941103533573, 720575941103924589, 720575941104131039, 720575941104297454, 720575941104309998, 720575941104699619, 720575941104700643, 720575941104705251, 720575941105379336, 720575941106152505, 720575941106477402, 720575941106600928, 720575941106639968, 720575941107285320, 720575941107298120, 720575941107437825, 720575941107829776, 720575941109570314, 720575941109605752, 720575941113287333, 720575941114006852, 720575941114011460, 720575941114493115, 720575941114493371, 720575941114494139, 720575941114687139, 720575941114711203, 720575941114933738, 720575941115332394, 720575941115960573, 720575941116548740, 720575941116572664, 720575941116663435, 720575941116976341, 720575941116997589, 720575941117000149, 720575941118374002, 720575941118374514, 720575941120382861, 720575941120702062, 720575941120904478, 720575941121184239, 720575941123725826, 720575941123736578, 720575941124436677, 720575941125743543, 720575941125953181, 720575941126229860, 720575941127918527, 720575941128195568, 720575941128216130, 720575941128500170, 720575941128608670, 720575941128703583, 720575941131576319, 720575941131588607, 720575941132405960, 720575941134134113, 720575941136000282, 720575941139252669, 720575941139510444, 720575941143312378, 720575941144379311, 720575941145828823, 720575941148267857, 720575941149954781, 720575941149993181, 720575941150910648, 720575941152155190, 720575941152796324, 720575941153471868, 720575941153561801, 720575941153566409, 720575941154431375, 720575941154453647, 720575941156114821, 720575941157944107, 720575941158042818, 720575941162298010, 720575941174542567, 720575941183694464, 720575941190025408, 720575941239851589]

syn_proof_only_df = client.materialize.synapse_query(pre_ids=proofread_id,
                                                  post_ids=proofread_id,
                                                  remove_autapses=True,
                                                  desired_resolution=[7.5,7.5,50]
                                                 )


syn_mat = syn_proof_only_df.pivot_table(index="pre_pt_root_id", 
                                        columns="post_pt_root_id", 
                                        values="size", 
                                        aggfunc="count"
                                       ).fillna(0)
syn_mat = syn_mat.reindex(index=np.array(syn_mat.columns)).fillna(0)
index_order = syn_mat.index.astype(int)
sorted_df = pre_post_df.set_index("cellid").loc[index_order].reset_index()
syn_mat = syn_mat.astype(float)  # make sure it's float, so division works
denom = sorted_df.set_index("pre_pt_root_id")["presynapse_total"]
normalized_mat = syn_mat.div(denom, axis=0)


# Restrict to selected_ids only
syn_mat_selected = syn_mat.loc[selected_ids, selected_ids]
normalized_mat_selected = normalized_mat.loc[selected_ids, selected_ids]


fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
sns.heatmap(syn_mat, cmap="gray_r", xticklabels=[], yticklabels=[], 
            ax=ax, square=False,
            cbar_kws={"label": "Connected"})
ax.set_title('Connectivity between proofread cells')

#%%
def _cast_to_target_dtype(s: pd.Series, to_int: bool):
    if to_int:
        return pd.to_numeric(s, errors='coerce').dropna().astype('int64')
    else:
        return s.astype(str).str.strip()
ids_are_int = pd.api.types.is_integer_dtype(syn_proof_only_df['pre_pt_root_id'])    
lookup_ids = _cast_to_target_dtype(lookuptable_df['FINAL NEURON ID'].dropna(), ids_are_int)
lookup_func = lookuptable_df.loc[lookup_ids.index, 'Functional Category'].astype(str)
id_to_func = pd.Series(lookup_func.values, index=lookup_ids.values)
id_to_func = id_to_func[~id_to_func.index.duplicated(keep='first')]
# Cast syn_proof_only_df id columns to the same dtype as the mapping keys
if ids_are_int:
    pre_ids  = pd.to_numeric(syn_proof_only_df['pre_pt_root_id'],  errors='coerce').astype('Int64')
    post_ids = pd.to_numeric(syn_proof_only_df['post_pt_root_id'], errors='coerce').astype('Int64')
else:
    pre_ids  = syn_proof_only_df['pre_pt_root_id'].astype(str).str.strip()
    post_ids = syn_proof_only_df['post_pt_root_id'].astype(str).str.strip()

# Map to new columns
syn_proof_only_df['pre_pt_function']  = pre_ids.map(id_to_func)
syn_proof_only_df['post_pt_function'] = post_ids.map(id_to_func)

# (Optional) Make missing labels explicit
syn_proof_only_df[['pre_pt_function','post_pt_function']] = \
    syn_proof_only_df[['pre_pt_function','post_pt_function']].fillna('Unknown')
    
    
syn_mat_try = syn_proof_only_df.pivot_table(index="pre_pt_function", 
                                        columns="post_pt_function", 
                                        values="size", 
                                        aggfunc="count"
                                       ).fillna(0)
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
sns.heatmap(syn_mat_try>0, cmap="gray_r", 
            #xticklabels=[], yticklabels=[], 
            ax=ax, square=False,
            cbar_kws={"label": "Connected"})
ax.set_title('Connectivity between proofread cells')
#%% filter the lookuptable_df so it only contains the proofread_id
proofread_id_str = [str(i) for i in proofread_id]
filtered_lookuptable_df = lookuptable_df[lookuptable_df['FINAL NEURON ID'].isin(proofread_id_str)]
filtered_lookuptable_df = filtered_lookuptable_df[~filtered_lookuptable_df['Cell Type'].isna()]

selected_id_str = [str(i) for i in selected_ids]
filtered_lookuptable_selected_df = lookuptable_df[lookuptable_df['FINAL NEURON ID'].isin(selected_id_str)]
filtered_lookuptable_selected_df = filtered_lookuptable_selected_df[~filtered_lookuptable_selected_df['Cell Type'].isna()]

#%% some tiny functions
def _return_xyz_coordinates(cell_id):
    x=lookuptable_df['Xprod'][lookuptable_df['FINAL NEURON ID'] == str(cell_id)].astype(int).values[0].item()
    y=lookuptable_df['Yprod'][lookuptable_df['FINAL NEURON ID'] == str(cell_id)].astype(int).values[0].item()
    z=lookuptable_df['Zprod'][lookuptable_df['FINAL NEURON ID'] == str(cell_id)].astype(int).values[0].item()
    return [x,y,z]

#%% simple interaction between exc vs inh
def sort_connectivity_matrix_exc_inh(
    syn_mat: pd.DataFrame,
    lookup_df: pd.DataFrame,
    id_col: str = 'FINAL NEURON ID',
    type_col: str = 'Cell Type',
    exc_labels=('exc', 'excitatory'),
    inh_labels=('inh', 'inhibitory')
):
    """
    Sort a square connectivity matrix by cell type (exc first, then inh).
    
    Returns:
        syn_sorted : pd.DataFrame
            Reordered connectivity matrix.
        order : list
            The neuron ID order used.
        types_ordered : pd.Series
            The cell-type series aligned to the order.
    """
    # detect dtype of syn_mat IDs
    index_is_int = pd.api.types.is_integer_dtype(syn_mat.index)

    # helper: cast to matching dtype
    def _to_dtype(x):
        if index_is_int:
            return pd.to_numeric(x, errors='coerce').dropna().astype(int)
        else:
            return x.astype(str).str.strip()

    ids_cast = _to_dtype(lookup_df[id_col].dropna())
    types_norm = lookup_df.loc[ids_cast.index, type_col].astype(str).str.strip().str.lower()

    # build mapping (keep first if duplicates)
    map_series = pd.Series(types_norm.values, index=ids_cast.values)
    map_series = map_series[~map_series.index.duplicated(keep='first')]

    all_ids = syn_mat.index
    types_aligned = map_series.reindex(all_ids).fillna('other')

    # masks
    exc_mask = types_aligned.isin([s.lower() for s in exc_labels])
    inh_mask = types_aligned.isin([s.lower() for s in inh_labels])
    other_mask = ~(exc_mask | inh_mask)

    order = list(all_ids[exc_mask]) + list(all_ids[inh_mask]) + list(all_ids[other_mask])
    syn_sorted = syn_mat.loc[order, order]
    types_ordered = types_aligned.loc[order]

    return syn_sorted, order, types_ordered

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def plot_connectivity_matrix(
    syn_sorted: pd.DataFrame,
    types_ordered: pd.Series,
    threshold: float = 0.0,           # entries > threshold are treated as connected
    cmap: str = "gray_r",
    title: str = "Connectivity (grouped)",
    ax=None
):
    """
    Plot a (pre-sorted) connectivity matrix with colored side strips for arbitrary groups.

    Parameters
    ----------
    syn_sorted : pd.DataFrame
        Square matrix already sorted the way you want.
    types_ordered : pd.Series
        Group labels aligned to syn_sorted's order (index == columns).
    threshold : float
        Values > threshold are shown as 1 (connected).
    cmap : str
        Colormap for the matrix.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axis to draw on; if None, a new figure is created.

    Returns
    -------
    ax, color_map
        The axis and a dict mapping {group_label: color} used in the strips/legend.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    # binarize for display
    # mat_disp = syn_sorted > threshold).astype(int)
    mat_disp = syn_sorted

    sns.heatmap(
        mat_disp,
        cmap=cmap,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        square=True,
        cbar_kws={"label": "normalized synapse"}
    )

    # ---- Build labels (in first-seen order) and colors ----
    # sanitize labels
    labels = pd.Index(pd.unique(types_ordered.fillna("Unlabeled").astype(str)))

    # palette: prefer tab10 (10 colors); if more needed, switch to tab20
    if len(labels) <= 10:
        base = plt.get_cmap("tab10")
        palette = [base(i) for i in range(10)]
    else:
        base = plt.get_cmap("tab20")
        palette = [base(i) for i in range(20)]

    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}

    # colors per row/column
    strip_colors = [color_map[str(t)] for t in types_ordered.fillna("Unlabeled").astype(str)]

    # left strip (rows)
    for i, c in enumerate(strip_colors):
        ax.add_patch(plt.Rectangle(
            xy=(-0.01, i), width=0.01, height=1, color=c, lw=0,
            transform=ax.get_yaxis_transform(), clip_on=False))

    # top strip (columns)
    for i, c in enumerate(strip_colors):
        ax.add_patch(plt.Rectangle(
            xy=(i, 1), height=0.01, width=1, color=c, lw=0,
            transform=ax.get_xaxis_transform(), clip_on=False))

    # dynamic legend
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color=color_map[lab], lw=4, label=str(lab))
        for lab in labels
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.25, 1), title="groups")

    ax.set_title(title)
    plt.tight_layout()
    return ax, color_map

# Step 1: sort
syn_sorted, order, types_ordered = sort_connectivity_matrix_exc_inh(
    normalized_mat, 
    filtered_lookuptable_df,
    id_col='FINAL NEURON ID',
    type_col='Cell Type'
)

# Step 2: plot
ax, color_map = plot_connectivity_matrix(syn_sorted, types_ordered,
                                         threshold=0, 
                                         title="Connectivity between proofread cells",
                                         cmap='grey_r')
plt.show()

#%%
# Step 1: sort
syn_sorted, order, types_ordered = sort_connectivity_matrix_exc_inh(
    normalized_mat_selected, 
    filtered_lookuptable_selected_df,
    id_col='FINAL NEURON ID',
    type_col='Cell Type'
)

# Step 2: plot
ax, color_map = plot_connectivity_matrix(syn_sorted, types_ordered,
                                         threshold=0, title="Connectivity between proofread cells",cmap='RdBu_r')
plt.show()

#%% sort based on ant/pos only

def sort_connectivity_matrix_by_function(
    syn_mat: pd.DataFrame,
    lookup_df: pd.DataFrame,
    id_col: str = 'FINAL NEURON ID',
    category_col: str = 'Functional Category',
    anterior_set = {
        'A Match Enhancement', 'A Match Suppression', 'A Modulation',
        'A Non-Match Enhancement',
        'A Non-Match Enhancement and Match Suppression', 'A Only'
    },
    posterior_set = {
        'P Match Suppression', 'P Modulation', 'P Non-Match Enhancement',
        'P Non-Match Enhancement and Match Suppression', 'P Only'
    },
    group_order = ('anterior', 'posterior', 'other')  # final sort order
):
    """
    Sort a square connectivity matrix by grouped Functional Category
    (anterior -> posterior -> other).

    Returns
    -------
    syn_sorted : pd.DataFrame
        Reordered connectivity matrix.
    order : list
        The neuron ID order used.
    groups_ordered : pd.Series
        Series of 'anterior'/'posterior'/'other' aligned to the order.
    """
    # --- ID dtype alignment (same logic as before) ---
    index_is_int = pd.api.types.is_integer_dtype(syn_mat.index)
    cols_are_int = pd.api.types.is_integer_dtype(syn_mat.columns)
    if index_is_int != cols_are_int:
        raise ValueError("syn_mat index and columns must be the same dtype (both int or both str).")

    def _to_dtype(x):
        if index_is_int:
            return pd.to_numeric(x, errors='coerce').dropna().astype(int)
        else:
            return x.astype(str).str.strip()

    ids_cast = _to_dtype(lookup_df[id_col].dropna())
    cats = lookup_df.loc[ids_cast.index, category_col].astype(str).str.strip()

    # Build ID -> raw category map (first occurrence if duplicates)
    id_to_cat = pd.Series(cats.values, index=ids_cast.values)
    id_to_cat = id_to_cat[~id_to_cat.index.duplicated(keep='first')]

    # Align to matrix IDs
    all_ids = syn_mat.index
    cats_aligned = id_to_cat.reindex(all_ids)

    # Map raw categories to groups
    def _group(c):
        if pd.isna(c):
            return 'other'
        if c in anterior_set:
            return 'anterior'
        if c in posterior_set:
            return 'posterior'
        return 'other'

    groups = cats_aligned.map(_group).fillna('other')

    # Build order by requested group_order
    masks = {g: (groups == g) for g in group_order}
    order = []
    for g in group_order:
        order.extend(list(all_ids[masks[g]]))
    # Include any unexpected labels not in group_order at the end (defensive)
    leftover = groups[~groups.isin(group_order)].index.tolist()
    if leftover:
        order.extend(leftover)

    # Reorder
    syn_sorted = syn_mat.loc[order, order]
    groups_ordered = groups.loc[order]

    return syn_sorted, order, groups_ordered

syn_sorted_func, order_func, groups_func = sort_connectivity_matrix_by_function(
    normalized_mat,
    filtered_lookuptable_df,
    id_col='FINAL NEURON ID',
    category_col='Functional Category'
)

# Then reuse your generic plotting function:
plot_connectivity_matrix(syn_sorted_func, groups_func, title='Connectivity by functional groups',cmap='RdBu_r');

#%%
syn_sorted_func, order_func, groups_func = sort_connectivity_matrix_by_function(
    normalized_mat_selected,
    filtered_lookuptable_selected_df,
    id_col='FINAL NEURON ID',
    category_col='Functional Category'
)

# Then reuse your generic plotting function:
plot_connectivity_matrix(syn_sorted_func, groups_func, title='Connectivity by functional groups',cmap='RdBu_r');


#%% replicate aaron's paper

# filter CCW, CW, excitatory and inhibitory
def sort_connectivity_matrix_function_x_type(
    syn_mat: pd.DataFrame,
    lookup_df: pd.DataFrame,
    id_col: str = 'FINAL NEURON ID',
    func_col: str = 'Functional Category',
    type_col: str = 'Cell Type',
    # functional groups
    anterior_set = {
        'A Match Enhancement', 'A Match Suppression', 'A Modulation',
        'A Non-Match Enhancement',
        'A Non-Match Enhancement and Match Suppression', 'A Only'
    },
    posterior_set = {
        'P Match Suppression', 'P Modulation', 'P Non-Match Enhancement',
        'P Non-Match Enhancement and Match Suppression', 'P Only'
    },
    # cell-type groups
    exc_labels=('exc', 'excitatory'),
    inh_labels=('inh', 'inhibitory'),
    # final ordering of the 2D bins
    pair_order=(
        'anterior-exc', 'anterior-inh',
        'posterior-exc', 'posterior-inh',
        'other-exc', 'other-inh',
        'other-other'  # safety tail for unlabeled types
    )
):
    """
    Sort a square connectivity matrix by (Functional Category × Cell Type).

    Returns
    -------
    syn_sorted : pd.DataFrame
        Reordered connectivity matrix.
    order : list
        The neuron ID order used.
    labels_ordered : pd.Series
        Combined labels like 'anterior-exc' aligned to the order.
    """

    # --- Ensure syn_mat index/columns dtype match; decide target dtype ---
    index_is_int = pd.api.types.is_integer_dtype(syn_mat.index)
    cols_are_int = pd.api.types.is_integer_dtype(syn_mat.columns)
    if index_is_int != cols_are_int:
        raise ValueError("syn_mat index and columns must have the same dtype (both int or both str).")

    def _to_dtype(x):
        if index_is_int:
            return pd.to_numeric(x, errors='coerce').dropna().astype(int)
        else:
            return x.astype(str).str.strip()

    # --- Build ID -> functional group and ID -> cell-type group mappings ---
    ids_cast = _to_dtype(lookup_df[id_col].dropna())

    func_raw = lookup_df.loc[ids_cast.index, func_col].astype(str).str.strip()
    type_raw = lookup_df.loc[ids_cast.index, type_col].astype(str).str.strip().str.lower()

    id_to_func = pd.Series(func_raw.values, index=ids_cast.values)
    id_to_func = id_to_func[~id_to_func.index.duplicated(keep='first')]

    id_to_type = pd.Series(type_raw.values, index=ids_cast.values)
    id_to_type = id_to_type[~id_to_type.index.duplicated(keep='first')]

    # align to syn_mat IDs
    all_ids = syn_mat.index
    func_aligned = id_to_func.reindex(all_ids)
    type_aligned = id_to_type.reindex(all_ids)

    # map to coarse groups
    def _func_group(x):
        if pd.isna(x): return 'other'
        return 'anterior' if x in anterior_set else ('posterior' if x in posterior_set else 'other')

    exc_set = {s.lower() for s in exc_labels}
    inh_set = {s.lower() for s in inh_labels}

    def _type_group(x):
        if pd.isna(x): return 'other'
        x = str(x).lower()
        return 'exc' if x in exc_set else ('inh' if x in inh_set else 'other')

    func_group = func_aligned.map(_func_group).fillna('other')
    type_group = type_aligned.map(_type_group).fillna('other')

    combined = (func_group + '-' + type_group).astype(str)

    # --- Build the final order following pair_order; append any leftovers defensively ---
    order = []
    for key in pair_order:
        mask = (combined == key)
        order.extend(list(all_ids[mask]))
    # leftovers (unexpected labels)
    leftovers = combined[~combined.isin(pair_order)].index.tolist()
    if leftovers:
        order.extend(leftovers)

    # --- Reorder outputs ---
    syn_sorted = syn_mat.loc[order, order]
    labels_ordered = combined.loc[order]

    return syn_sorted, order, labels_ordered

syn_sorted_fx, order_fx, labels_fx = sort_connectivity_matrix_function_x_type(
    normalized_mat,
    filtered_lookuptable_df,
    id_col='FINAL NEURON ID',
    func_col='Functional Category',
    type_col='Cell Type',
    pair_order=(
        'anterior-exc', 
        'posterior-exc', 'other-exc', 'anterior-inh','posterior-inh',
        'other-inh',
        'other-other'  # safety tail for unlabeled types
))

# Your generic plotting function will auto-make colors/legend from labels_fx
ax, color_map = plot_connectivity_matrix(
    syn_sorted_fx,
    labels_fx,
    threshold=0.0,
    title='Connectivity (anterior/posterior × exc/inh)',cmap='RdBu_r'
)

#%%
syn_sorted_fx, order_fx, labels_fx = sort_connectivity_matrix_function_x_type(
    normalized_mat_selected,
    filtered_lookuptable_selected_df,
    id_col='FINAL NEURON ID',
    func_col='Functional Category',
    type_col='Cell Type',
    pair_order=(
        'anterior-exc', 
        'posterior-exc', 'other-exc', 'anterior-inh','posterior-inh',
        'other-inh',
        'other-other'  # safety tail for unlabeled types
    )
)

# Your generic plotting function will auto-make colors/legend from labels_fx
ax, color_map = plot_connectivity_matrix(
    syn_sorted_fx,
    labels_fx,
    threshold=0.0,
    title='Connectivity (anterior/posterior × exc/inh)',cmap='RdBu_r'
)


#%%
def plot_connectivity_matrix_asymmetric(
    mat: pd.DataFrame,
    row_labels: pd.Series,
    col_labels: pd.Series,
    cmap='RdBu_r',
    title='Subset connectivity',
    cbar_label='normalized synapse'
):
    """
    Heatmap with independent row/column label strips.
    """
    # ensure alignment
    row_labels = row_labels.reindex(mat.index).fillna('Unlabeled').astype(str)
    col_labels = col_labels.reindex(mat.columns).fillna('Unlabeled').astype(str)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sns.heatmap(
        mat,
        cmap=cmap,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        square=False,
        cbar_kws={"label": cbar_label}
    )

    # Build palette from union of row+col labels
    all_labs = pd.Index(pd.unique(pd.concat([row_labels, col_labels], axis=0)))
    base = plt.get_cmap("tab20" if len(all_labs) > 10 else "tab10")
    palette = [base(i) for i in range(max(10, len(all_labs)))]
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(all_labs)}

    # Row strip (left)
    row_colors = [color_map[l] for l in row_labels]
    for i, c in enumerate(row_colors):
        ax.add_patch(plt.Rectangle(
            xy=(-0.01, i), width=0.01, height=1, color=c, lw=0,
            transform=ax.get_yaxis_transform(), clip_on=False
        ))

    # Column strip (top)
    col_colors = [color_map[l] for l in col_labels]
    for i, c in enumerate(col_colors):
        ax.add_patch(plt.Rectangle(
            xy=(i, 1), height=0.01, width=1, color=c, lw=0,
            transform=ax.get_xaxis_transform(), clip_on=False
        ))

    # Legend (only show unique labels actually present)
    present_labs = pd.unique(pd.Index(row_labels).append(pd.Index(col_labels)))
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color=color_map[lab], lw=4, label=str(lab))
        for lab in present_labs
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.25, 1), title="groups")
    ax.set_title(title)
    plt.tight_layout()
    return ax

pair_order=(
    'anterior-exc', 
    'posterior-exc', 'other-exc', 'anterior-inh','posterior-inh',
    'other-inh',
    'other-other')
non_match_group = lookuptable_df['FINAL NEURON ID'][lookuptable_df['Functional Category']=='Categorical Non-Match']
non_match_group_int = {int(x) for x in non_match_group if pd.notna(x)}
non_match_group = [item for item in non_match_group_int if item in selected_ids]
cols_in_order = list(syn_sorted_func.columns)   # functional order from your earlier sort
rows_in_order = [rid for rid in cols_in_order if rid in non_match_group]
rows_in_order += [rid for rid in non_match_group if rid not in set(cols_in_order)]
subset_mat = normalized_mat_selected.reindex(index=rows_in_order, columns=cols_in_order).fillna(0.0)
col_labels = labels_fx.reindex(cols_in_order).fillna('other-other')
col_labels = col_labels.where(col_labels.isin(pair_order), 'other-other')
row_labels = pd.Series(index=rows_in_order, data='non-match-group')

# 6) Call it
plot_connectivity_matrix_asymmetric(
    subset_mat,
    row_labels=row_labels,            # single color down Y
    col_labels=col_labels,            # functional colors across X (as before)
    title='Subset: non_match_group (rows) × functional order (cols)',
    cmap='RdBu_r'
)
plt.show()

#%%
def aggregate_connectivity_by_group(
    syn_sorted: pd.DataFrame,
    labels_ordered: pd.Series,
    aggfunc='mean'
):
    """
    Collapse a sorted connectivity matrix into group × group aggregates.

    Parameters
    ----------
    syn_sorted : pd.DataFrame
        Square connectivity matrix (already ordered).
    labels_ordered : pd.Series
        Labels for each row/column, aligned to syn_sorted.
    aggfunc : str or callable
        Aggregation function to apply (default: 'mean'; could also use 'sum').

    Returns
    -------
    group_mat : pd.DataFrame
        Matrix where rows/columns are groups and entries are aggregated values.
    """
    # Align
    syn_sorted = syn_sorted.copy()
    syn_sorted.index = labels_ordered
    syn_sorted.columns = labels_ordered

    # Group both axes
    group_mat = syn_sorted.groupby(level=0, axis=0).agg(aggfunc)   # rows
    group_mat = group_mat.groupby(level=0, axis=1).agg(aggfunc)   # cols

    return group_mat

def plot_group_connectivity(
    group_mat: pd.DataFrame,
    cmap="gist_heat_r",
    annot=True,
    fmt=".2f",      # use ".2f" for mean, "d" for counts
    vmin=None, vmax=None,
    title="Aggregate connectivity",
    order=None      # <-- new argument
):
    """
    Plot an aggregated group × group connectivity matrix,
    with super x-axis as 'postsynapse' and y-axis as 'presynapse'.

    Parameters
    ----------
    group_mat : pd.DataFrame
        Aggregated group matrix (square).
    cmap : str
        Colormap.
    annot : bool
        Whether to annotate cells.
    fmt : str
        Format for annotations.
    vmin, vmax : float
        Color scale limits.
    title : str
        Plot title.
    order : list or None
        Custom order of groups for rows/columns. If None, use group_mat order.
    """
    if order is not None:
        group_mat = group_mat.reindex(index=order, columns=order)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    sns.heatmap(
        group_mat,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'label': 'connectivity', 'location': 'right'},
        annot_kws={"fontsize": 6}
    )

    # Axis labels
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("postsynapse", labelpad=20, fontsize=12)
    ax.set_ylabel("presynapse", labelpad=20, fontsize=12)
    ax.set_title(title)

    plt.tight_layout()
    return ax

def normalize_group_matrix(group_mat: pd.DataFrame, axis=1):
    """
    Normalize a group-by-group matrix so rows (axis=1) or columns (axis=0) sum to 1.

    Parameters
    ----------
    group_mat : pd.DataFrame
        Aggregated group matrix.
    axis : int
        1 → normalize rows to sum 1 (default).
        0 → normalize columns to sum 1.

    Returns
    -------
    norm_mat : pd.DataFrame
    """
    if axis == 1:
        norm_mat = group_mat.div(group_mat.sum(axis=1).replace(0, 1), axis=0)
    elif axis == 0:
        norm_mat = group_mat.div(group_mat.sum(axis=0).replace(0, 1), axis=1)
    else:
        raise ValueError("axis must be 0 (columns) or 1 (rows)")
    return norm_mat
#%%
# First aggregate
group_means = aggregate_connectivity_by_group(syn_sorted_fx, labels_fx, aggfunc='mean')
group_row_norm = normalize_group_matrix(group_means)
# Then plot
custom_order = ["anterior-exc", "anterior-inh",
                "posterior-exc", "posterior-inh",
                "other-exc", "other-inh"]

custom_order = ["anterior-exc", 
                "posterior-exc", 
                "other-exc",
                "anterior-inh",
                "posterior-inh",
                 "other-inh"]


ax = plot_group_connectivity(
    group_row_norm*1000,
    annot=True,
    fmt=".2f",
   # vmin=0, vmax=1,
    title="Row-normalized connectivity",
    order=custom_order
)
plt.show()

