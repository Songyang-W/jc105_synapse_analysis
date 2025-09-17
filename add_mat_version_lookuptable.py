import pandas as pd
import numpy as np
import time
from requests import HTTPError
from caveclient import CAVEclient

# ========= paths =========
table_path = '/net/claustrum/mnt/data/Dropbox/Chen Lab Dropbox/Chen Lab Team Folder/Projects/Connectomics/Animals/jc105/LOOKUP_TABLE.xlsx'
output_excel = table_path.replace('.xlsx', '_with_versions.xlsx')   # write next to source
# output_csv   = table_path.replace('.xlsx', '_with_versions.csv')   # optional
ID_COL = 'FINAL NEURON ID'

# ========= load lookup table & extract IDs =========
lookuptable_df = pd.read_excel(table_path, sheet_name="MASTER_LIST",
                   dtype={
        "SHARD ID": "string",
        "FINAL NEURON ID": "string",
        "EM NAME (Nuclear)": "string",
    })
# Clean/standardize the column name if there might be stray spaces or case issues
if ID_COL not in lookuptable_df.columns:
    raise KeyError(f"Column '{ID_COL}' not found in {table_path}. Available: {list(lookuptable_df.columns)}")

# Keep only finite integers; drop NaNs
root_ids = (
    pd.to_numeric(lookuptable_df[ID_COL], errors='coerce')
      .dropna()
)

# ========= query materialization versions =========
client = CAVEclient("jchen_mouse_cortex")
available_version = client.materialize.get_versions()

# Trim column names just in case
lookuptable_df.rename(columns={c: c.strip() for c in lookuptable_df.columns}, inplace=True)
if ID_COL not in lookuptable_df.columns:
    raise KeyError(f"Column '{ID_COL}' not found. Available: {list(lookuptable_df.columns)}")

# Convert IDs to clean Python ints (drop NaNs)
id_series = pd.to_numeric(lookuptable_df[ID_COL], errors='coerce').dropna()
root_ids = [int(x) for x in id_series.tolist()]
root_ids_unique = sorted(set(root_ids))

print(f"Total rows in table: {len(lookuptable_df)}")
print(f"Non-null IDs: {len(root_ids)}; Unique IDs: {len(root_ids_unique)}")

# ========= query materialization versions =========
client = CAVEclient("jchen_mouse_cortex")
available_versions = client.materialize.get_versions()
print(f"Available versions from CAVE: {available_versions}")

rows = []  # will store (root_id, version)

for v in available_versions:
    try:
        client.materialize.version = int(v)
        syn_pre = client.materialize.synapse_query(
            pre_ids=root_ids_unique, remove_autapses=True,
            desired_resolution=[7.5, 7.5, 50]
        )
        syn_post = client.materialize.synapse_query(
            post_ids=root_ids_unique, remove_autapses=True,
            desired_resolution=[7.5, 7.5, 50]
        )
    except HTTPError:
        time.sleep(0.1)
        continue

    pre_ids_found  = syn_pre['pre_pt_root_id'].unique()   if len(syn_pre)  else np.array([], dtype=np.int64)
    post_ids_found = syn_post['post_pt_root_id'].unique() if len(syn_post) else np.array([], dtype=np.int64)
    exist_ids = np.union1d(pre_ids_found, post_ids_found)

    # Append rows for every (rid, version) we found
    for rid in exist_ids:
        rows.append((int(rid), int(v)))

# ========= build results DataFrame and merge =========
if rows:
    rv_df = pd.DataFrame(rows, columns=['root_id', 'version'])
    # aggregate versions per root_id
    agg_df = (rv_df.sort_values(['root_id','version'])
                  .groupby('root_id')['version']
                  .apply(lambda s: sorted(set(s.tolist())))
                  .reset_index(name='materialize_versions_list'))
    agg_df['materialize_versions_str'] = agg_df['materialize_versions_list'].apply(
        lambda lst: ",".join(map(str, lst)) if lst else ""
    )

    # Merge back: make sure join keys are the same dtype
    lookuptable_df['_root_id_int'] = pd.to_numeric(lookuptable_df[ID_COL], errors='coerce').astype('Int64')
    merged = lookuptable_df.merge(
        agg_df,
        left_on='_root_id_int',
        right_on='root_id',
        how='left'
    ).drop(columns=['root_id'])

    # Fill NaNs with empty list / empty string for rows with no hits
    merged['materialize_versions_list'] = merged['materialize_versions_list'].apply(lambda x: x if isinstance(x, list) else [])
    merged['materialize_versions_str'] = merged['materialize_versions_str'].fillna("")

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        merged.drop(columns=['_root_id_int']).to_excel(writer, index=False)

    print(f"Saved updated table with versions to: {output_excel}")
    print(f"IDs with >=1 version hit: {agg_df.shape[0]} / {len(root_ids_unique)}")
else:
    # No rows found at all—likely an ID/type mismatch or no synapses for given IDs
    lookuptable_df['materialize_versions_list'] = [[] for _ in range(len(lookuptable_df))]
    lookuptable_df['materialize_versions_str'] = ""
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        lookuptable_df.to_excel(writer, index=False)
    print("No (root_id, version) pairs were found. Wrote empty version columns.")
