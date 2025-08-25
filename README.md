# jc105_synapse_analysis

# Lookup Root IDs from Shard IDs

This script automates the process of mapping **shard IDs** to their corresponding **root IDs** using the [CAVEclient](https://github.com/seung-lab/CAVEclient) for the `jchen_mouse_cortex` project.

##  Features
- Load shard and neuron IDs from an Excel file (`lookup table.xlsx`, sheet `MASTER_LIST`).
- Query the **chunkedgraph service** to get the root ID for each shard ID.
- Append a new column (`root id`) to the DataFrame with the retrieved values.
- Filter and display rows that match a **target root ID** (default: `720575941088408585`).
- Optionally save the updated table back to Excel.

# Proofread Neuron Connectivity Analysis

This script analyzes synaptic connectivity between a set of **proofread neurons** in the `jchen_mouse_cortex` dataset using the [CAVEclient](https://github.com/seung-lab/CAVEclient).

## Features
- Query synapses among a given list of root IDs (`proofread_id`).
- Build a dense **connectivity matrix** and export as CSV.
- Visualize connectivity with a **heatmap** (binary connections).
- Inspect inputs to an individual neuron (total count + partner breakdown).
- Test for pairwise connectivity between specific cells.
- Report which neurons have **no synapses** after filtering.
- Utility functions:
  - `pre_post_synapse_count()`: input/output counts via CAVEclient.
  - `pre_post_synapse_count_from_df()`: counts from an existing dataframe.

## Requirements
```bash
pip install caveclient pandas matplotlib seaborn
