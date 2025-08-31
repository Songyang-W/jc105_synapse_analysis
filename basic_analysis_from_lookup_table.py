#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 17:00:43 2025

@author: songyangwang
"""
import pandas as pd

table_directory = '/Users/songyangwang/Downloads/LOOKUP_TABLE.xlsx'
df = pd.read_excel(table_directory, sheet_name="MASTER_LIST",
                   dtype={
        "SHARD ID": "string",
        "FINAL NEURON ID": "string",
        "EM NAME (Nuclear)": "string"
    })

#%%
import pandas as pd
import re
from caveclient import CAVEclient
from nglui import parser


client = CAVEclient("jchen_mouse_cortex")
# check individual cell annotatio
state_id = 5517900264767488
state_json = client.state.get_state_json(state_id)
state = parser.StateParser(state_json)

annotation_layer = state.annotation_dataframe()
ann_props = state.state.layers[-1].annotationProperties
tag_map = {i: prop.tag.upper() for i, prop in enumerate(ann_props)}

#%% pulling all annotation detail

def normalize_tags(tags, description, tag_map=None):
    """
    Combine numeric tags (0/1/2) with text mentions in description ('exit','stop','end').
    tag_map maps numeric codes -> label (e.g. {0:'EXIT',1:'END',2:'STOP'}).
    Works even if tag_map is None/empty, tags is NaN/None, or description is missing.
    """
    tag_map = tag_map or {}
    labels = set()

    # numeric tags
    if isinstance(tags, (list, tuple, set, pd.Series)):
        for t in tags:
            if t in tag_map:
                labels.add(tag_map[t])

    # description mentions
    if isinstance(description, str):
        desc = description.lower()
        if "exit" in desc:
            labels.add("EXIT")
        if "stop" in desc:
            labels.add("STOP")
        if "end" in desc:
            labels.add("END")

    return sorted(labels)  # stable order


def normalize_difficulty(description):
    """
    Standardize 'difficulties' from free text:
      - CONTRAST (matches 'contrast' or 'blur'; also fixes 'contract' -> 'contrast')
      - MISSING  (matches 'missing' and common misspelling 'mising')
      - THIN
      - OTHERS   (matches 'other'/'others')
      - None     if nothing relevant is found
    """
    if not isinstance(description, str):
        return None

    desc = description.lower()

    # quick spelling normalizations
    desc = desc.replace("mising", "missing").replace("contract", "contrast")

    if ("contrast" in desc) or ("blur" in desc):
        return "CONTRAST"
    if "missing" in desc:
        return "MISSING"
    if "thin" in desc:
        return "THIN"
    if "other" in desc:
        return "OTHERS"
    return None


# --- main collector ---
def collect_annotations(client, parser, status_ids: pd.Series) -> pd.DataFrame:
    """
    For each nglui_status id:
      - fetch state
      - parse annotation dataframe
      - build tag_map from annotationProperties if present (else empty)
      - add tag_labels and difficulties
    Returns a single concatenated dataframe (may be empty).
    """
    all_annos = []

    for raw_id in status_ids.dropna().unique():
        try:
            state_id = int(raw_id)
        except Exception:
            # non-numeric or bad id -> skip
            continue

        try:
            state_json = client.state.get_state_json(state_id)
            state = parser.StateParser(state_json)
        except Exception as e:
            print(f"⚠️ fetch/parse failed for {state_id}: {e}")
            continue

        try:
            # dataframe of annotations (may exist even if no annotationProperties)
            ann_df = state.annotation_dataframe()
            if ann_df is None or ann_df.empty:
                continue

            # ensure columns exist
            if "tags" not in ann_df.columns:
                ann_df["tags"] = [[] for _ in range(len(ann_df))]
            if "description" not in ann_df.columns:
                ann_df["description"] = None

            # find the annotation layer safely; do NOT assume layers[-1]
            ann_layer = None
            for lyr in getattr(state.state, "layers", []):
                if getattr(lyr, "type", None) == "annotation":
                    ann_layer = lyr
                    break

            # build tag_map if annotationProperties exists; else empty
            tag_map = {}
            if ann_layer is not None and hasattr(ann_layer, "annotationProperties"):
                props = getattr(ann_layer, "annotationProperties", None)
                if props:
                    # map by numeric index of properties: 0 -> first prop, etc.
                    tag_map = {i: getattr(p, "tag", "").upper() for i, p in enumerate(props)}

            # add normalized columns
            ann_df["tag_labels"] = ann_df.apply(
                lambda row: normalize_tags(row.get("tags"), row.get("description"), tag_map),
                axis=1
            )
            ann_df["difficulties"] = ann_df["description"].apply(normalize_difficulty)

            # keep state id for traceability
            ann_df["state_id"] = state_id

            all_annos.append(ann_df)

        except Exception as e:
            print(f"⚠️ processing failed for {state_id}: {e}")
            continue

    if all_annos:
        # Clear attrs in case any DF carries non-comparable attrs (avoids concat ValueError)
        for df in all_annos:
            df.attrs = {}
        return pd.concat(all_annos, ignore_index=True)

    # nothing collected
    return pd.DataFrame()

anno_df = collect_annotations(client, parser, df_add_status["nglui_status"])
# e.g., inspect:
anno_df[["layer","tags","description","tag_labels","difficulties","state_id"]].head()

#%%
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Histogram of tags (only EXIT, STOP, END) ---
all_tags = []
for tags in anno_df["tag_labels"].dropna():
    if isinstance(tags, list):
        all_tags.extend(tags)
    elif pd.notna(tags):
        all_tags.append(tags)

wanted_tags = ["EXIT", "STOP", "END"]
tag_counts = pd.Series(all_tags).value_counts()
tag_counts = tag_counts.reindex(wanted_tags).fillna(0).astype(int)

plt.figure(figsize=(6,4))
bars = plt.bar(tag_counts.index, tag_counts.values)
plt.title("Distribution of Tags")
plt.xlabel("Tag")
plt.ylabel("Count")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
             ha='center', va='bottom')
plt.show()


# --- Difficulties plot with label rename ---
wanted_diff_tags = ["MISSING", "CONTRAST", "THIN"]
diff_counts = anno_df["difficulties"].value_counts(dropna=True)
diff_counts = diff_counts.reindex(wanted_diff_tags).fillna(0).astype(int)

# Rename just for plotting
plot_labels = diff_counts.index.to_series().replace({"MISSING": "MISSING SECTIONS"})

plt.figure(figsize=(6,4))
bars = plt.bar(plot_labels, diff_counts.values)
plt.title("Distribution of Difficulties")
plt.xlabel("Difficulty")
plt.ylabel("Count")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
             ha='center', va='bottom')
plt.show()
