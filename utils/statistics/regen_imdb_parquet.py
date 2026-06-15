"""
Standalone, ADDITIVE regenerator for IMDB statistics from the processed parquet
files. This file does NOT modify any existing PRICE code; it only imports the
unchanged pure stat helpers (get_histogram, get_summary) from tools.py and
replicates the size/fanout logic from gen_size.py / gen_fanout.py verbatim.

It loads tables straight from parquet (so no postgres CSV / DDL plumbing is
needed) and writes the four statistics files under the names that
setup/features_tool.py reads:
    histogram40.pkl, summary40.pkl, fanout40.pkl, size.pkl

Run:
    python regen_imdb_parquet.py
"""
import os
import re
import time
import pickle

import numpy as np
import pandas as pd

from tools import get_histogram, get_summary, replace_comments

BIN_SIZE = 40
USAGE = "test"
DB = "imdb"
PARQUET_DIR = "/datadrive500/cardinality-estimation-data/processed_data/imdb"

current_dir = os.path.dirname(os.path.abspath(__file__))
stats_dir = f"{current_dir}/../../datas/statistics/{USAGE}/{DB}"
workload_file = f"{current_dir}/../../datas/workloads/{USAGE}/{DB}/workloads.sql"

t_all0 = time.time()

# ---- schema config (from abbrev_col_type.pkl, built from scratch earlier) ----
with open(f"{stats_dir}/abbrev_col_type.pkl", "rb") as f:
    _cfg = pickle.load(f)
abbrev = _cfg["abbrev"]                 # real table name -> alias
col_type = _cfg["col_type"]             # alias -> {'ctn':[...], 'dsct':[...]}
abbrev_inv = {v: k for k, v in abbrev.items()}

# ---- load tables straight from parquet, keyed by alias ----
print("------------------load tables from parquet------------------")
tables = {}
for real_name, alias in abbrev.items():
    df = pd.read_parquet(f"{PARQUET_DIR}/{real_name}.parquet")
    tables[alias] = df
    print(f"loaded {real_name} as {alias}: rows={len(df)} cols={list(df.columns)}")

# ---- histogram (continuous columns) : mirrors gen_histogram.py ----
print("------------------histogram------------------")
t0 = time.time()
histogram = {}
for alias in tables:
    histogram[alias] = {}
    for column in tables[alias].columns:
        if column in col_type[alias]["ctn"]:
            hist, bin_edges, len_col, min_v, max_v = get_histogram(tables, alias, column, BIN_SIZE)
            histogram[alias][column] = {
                "hist": hist, "bin_edges": bin_edges, "len": len_col,
                "min_value": min_v, "max_value": max_v,
            }
with open(f"{stats_dir}/histogram{BIN_SIZE}.pkl", "wb") as f:
    pickle.dump(histogram, f)
print(f"histogram done in {time.time()-t0:.2f}s")

# ---- summary (discrete columns) : mirrors gen_summary.py ----
print("------------------summary------------------")
t0 = time.time()
summary = {}
for alias in tables:
    summary[alias] = {}
    for column in tables[alias].columns:
        if column in col_type[alias]["dsct"]:
            keys, values = get_summary(tables, alias, column)
            summary[alias][column] = {"keys": keys, "values": values}
with open(f"{stats_dir}/summary{BIN_SIZE}.pkl", "wb") as f:
    pickle.dump(summary, f)
print(f"summary done in {time.time()-t0:.2f}s")

# ---- size : mirrors gen_size.py ----
print("------------------size------------------")
t0 = time.time()
size = {}
for alias in tables:
    size[alias] = {
        "size": tables[alias].shape[0],
        "num_cols": tables[alias].shape[1],
        "num_rows": tables[alias].shape[0],
    }
with open(f"{stats_dir}/size.pkl", "wb") as f:
    pickle.dump(size, f)
print(f"size done in {time.time()-t0:.2f}s")

# ---- fanout : mirrors gen_fanout.py exactly ----
print("------------------fanout------------------")
t0 = time.time()
joins = set([])
with open(workload_file, "r") as wf:
    for line in wf:
        line = replace_comments(line, "")
        line = line.split("||")[0].strip()
        if line.startswith("select"):
            if "where" not in line:
                continue
            candidates = line.strip().split("where")[1].strip(";").strip()
            candidates = re.split(r"(?i)\band\b", candidates)
        elif line.startswith("SELECT"):
            if "WHERE" not in line:
                continue
            candidates = line.strip().split("WHERE")[1].strip(";").strip()
            candidates = re.split(r"(?i)\band\b", candidates)
        else:
            raise ValueError("workload file must start with select or SELECT")
        candidates = [c.strip("(") for c in candidates]
        candidates = [c.strip(")") for c in candidates]
        candidates = [c.strip() for c in candidates if " = " in c and "." in c.split(" = ")[0] and "." in c.split(" = ")[1]]
        for c in candidates:
            left, right = c.split("=")[0].strip(), c.split("=")[1].strip()
            left = left.replace("(", "").replace(")", "").replace(";", "")
            right = right.replace("(", "").replace(")", "").replace(";", "")
            if left.split(".")[0] in abbrev_inv and right.split(".")[0] in abbrev_inv:
                if (left, right) not in joins and (right, left) not in joins:
                    joins.add((left, right))
print("joins:", joins)

fanout = {}
for join in joins:
    left, right = join[0], join[1]
    left_table, left_column = left.split(".")[0], left.split(".")[1]
    right_table, right_column = right.split(".")[0], right.split(".")[1]

    value_counts = pd.DataFrame(tables[right_table][right_column].value_counts())
    table_merge = pd.merge(left=tables[left_table][left_column], right=value_counts,
                           left_on=left_column, right_on=right_column, how="left")
    bin_edges = histogram[left_table][left_column]["bin_edges"]
    left_fanout = []
    assert len(bin_edges) == BIN_SIZE + 1
    for i in range(len(bin_edges) - 1):
        if i != BIN_SIZE - 1:
            mask = (table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] < bin_edges[i + 1])
        else:
            mask = (table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] <= bin_edges[i + 1])
        tmp = table_merge.loc[mask, "count"].sum()
        if tmp != 0:
            tmp = tmp / len(table_merge.loc[mask])
        left_fanout.append(tmp)

    value_counts = pd.DataFrame(tables[left_table][left_column].value_counts())
    table_merge = pd.merge(left=tables[right_table][right_column], right=value_counts,
                           left_on=right_column, right_on=left_column, how="left")
    bin_edges = histogram[right_table][right_column]["bin_edges"]
    right_fanout = []
    assert len(bin_edges) == BIN_SIZE + 1
    for i in range(len(bin_edges) - 1):
        if i != BIN_SIZE - 1:
            mask = (table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] < bin_edges[i + 1])
        else:
            mask = (table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] <= bin_edges[i + 1])
        tmp = table_merge.loc[mask, "count"].sum()
        if tmp != 0:
            tmp = tmp / len(table_merge.loc[mask])
        right_fanout.append(tmp)

    fanout[join] = [left_fanout, right_fanout]
    fanout[(join[1], join[0])] = [right_fanout, left_fanout]

with open(f"{stats_dir}/fanout{BIN_SIZE}.pkl", "wb") as f:
    pickle.dump(fanout, f)
print(f"fanout done in {time.time()-t0:.2f}s")

print(f"==== ALL STATS regenerated in {time.time()-t_all0:.2f}s ====")
