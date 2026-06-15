import os
import re
import pickle
import numpy as np
import pandas as pd

def replace_comments(text, replacement):
    pattern = r"/\*.*?\*/"
    modified_text = re.sub(pattern, replacement, text)
    return modified_text

def load_abbrev_coltype(folder_path, db, usage):
    col_type_path = f"{folder_path}/statistics/{usage}/{db}/abbrev_col_type.pkl"
    assert os.path.exists(col_type_path), f'{col_type_path} not exists'
    with open(col_type_path, 'rb') as f:
        data = pickle.load(f)
        abbrev = data['abbrev']
        col_type = data['col_type']
        print('abbrev: ', abbrev)
    for table in col_type:
        for c_type in col_type[table]:
            print(f"{table}.{c_type}: {col_type[table][c_type]}")
        print('-' * 50)
    print('-' * 100)
    return abbrev, col_type

def load_tbls_cols_types(folder_path, db):
    type_file = f"{folder_path}/datasets/{db}/postgres_create_{db}.sql"
    print(f'load col type file: {type_file}')
    assert os.path.exists(type_file)
    with open(type_file, 'r') as file:
        tbls_cols_types, decimal_tbls_cols = {}, {}
        with open(type_file, 'r') as file:
            for line in file:
                if "create table" in line:
                    tbl = line.split("create table")[1].split()[0].replace('"', '')
                    tbls_cols_types[tbl] = {}
                    decimal_tbls_cols[tbl] = []
                elif "integer" in line or "bigint" in line or "smallint" in line:
                    col = line.strip().split(" ")[0].replace('"', '')
                    tbls_cols_types[tbl][col] = pd.Int64Dtype()
                elif "character" in line or "varchar(" in line or "char(" in line: 
                    col = line.strip().split(" ")[0].replace('"', '')
                    tbls_cols_types[tbl][col] = pd.StringDtype()
                elif "decimal(" in line:
                    col = line.strip().split(" ")[0].replace('"', '')
                    decimal_tbls_cols[tbl].append(col)
                elif "double precision" in line:
                    col = line.strip().split(" ")[0].replace('"', '')
                    tbls_cols_types[tbl][col] = pd.Float64Dtype()
                else:
                    pass
    print(tbls_cols_types)     
    return tbls_cols_types, decimal_tbls_cols

TPCH_SKEWED_PARQUET_DIR = "/home/student.unimelb.edu.au/lrathuwadu/cardinality-estimation-data/original_data/tpch_skewed"
TPCH_UNIFORM_PARQUET_DIR = "/home/student.unimelb.edu.au/lrathuwadu/cardinality-estimation-data/original_data/tpch_uniform_sf1"
IMDB_PARQUET_DIR = "/datadrive500/cardinality-estimation-data/processed_data/imdb"

# Free-text / admin columns to drop from every table before any stats/feature
# work. None of these are referenced by the TPC-H workloads we run.
_DROP_COLS = {
    "lineitem": ["l_comment"],
    "orders":   ["o_comment", "o_clerk"],
    "customer": ["c_comment", "c_name", "c_address", "c_phone"],
    "part":     ["p_comment", "p_name"],
    "supplier": ["s_comment", "s_name", "s_address", "s_phone"],
    "partsupp": ["ps_comment"],
    "nation":   ["n_comment"],
    "region":   ["r_comment"],
}


def _load_parquet_table(parquet_path, tbl_dtypes, drop_cols=None):
    """Load a parquet file and coerce dtypes to match the postgres schema.
    Datetime columns get converted to int64 epoch days (matches schema bigint)."""
    df = pd.read_parquet(parquet_path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            epoch_days = (df[col].astype('datetime64[ns]').astype('int64') // (10 ** 9 * 86400))
            df[col] = epoch_days.astype(pd.Int64Dtype())
    if tbl_dtypes:
        for col, dt in tbl_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dt)
                except (TypeError, ValueError):
                    pass
    return df


def load_table_datas(folder_path, db, abbrev, tbls_cols_types):
    _parquet_dir = {
        "tpch_skewed": TPCH_SKEWED_PARQUET_DIR,
        "tpch_uniform": TPCH_UNIFORM_PARQUET_DIR,
        "imdb": IMDB_PARQUET_DIR,
    }.get(db)
    if _parquet_dir is not None:
        tables = {}
        print(f"loading parquet tables from {_parquet_dir}")
        for tablename in abbrev:
            path = f"{_parquet_dir}/{tablename}.parquet"
            assert os.path.exists(path), f"missing parquet: {path}"
            table = _load_parquet_table(
                path,
                tbls_cols_types.get(tablename),
                drop_cols=_DROP_COLS.get(tablename),
            )
            print(
                f"load table: {tablename} as {abbrev[tablename]}  rows={len(table)} "
                f"cols={list(table.columns)}"
            )
            tables[abbrev[tablename]] = table
        return tables

    load_file = f"{folder_path}/datasets/{db}/postgres_create_{db}.sql"
    tables = {}
    print("load table info from " + load_file)
    with open(load_file, 'r') as lf:
        for line in lf:
            if line.startswith('\copy') or line.startswith('\COPY'):
                tablename = line.split(' ')[1].strip("'")
                filename = line.split(' ')[3].strip("'")
                path = f"{folder_path}/datasets/{db}/{filename}"
                assert os.path.exists(path)
                table = pd.read_csv(path, sep='|', quotechar='"', escapechar='\\', dtype=tbls_cols_types[tablename], keep_default_na=False, na_values=['NULL'])

                assert tablename not in tables
                assert tablename in abbrev

                print('load table: ', tablename, ' as ', abbrev[tablename])
                tables[abbrev[tablename]] = table
    return tables

def get_histogram(tables, table, column, bin_size):
    len_column = tables[table][column].shape[0]
    tmp_column = tables[table][column].dropna()
    print('type of tmp_column: ', type(tmp_column))
    print('tmp_column: ', tmp_column)
    tmp_column = tmp_column.astype(float)
    if len(tmp_column) == 0:
        return np.zeros(bin_size), np.zeros(bin_size), 0, 0, 0
    hist, bin_edges = np.histogram(tmp_column, bins=bin_size, density=False)
    hist = hist.astype(float)
    min_value, max_value = min(tmp_column), max(tmp_column)
    return hist, bin_edges, len_column, min_value, max_value

def get_summary(tables, table, column):
    """ space saving summary algorithm"""
    K = 99999999
    summary = {}
    tmp_column = tables[table][column].dropna()
    print('current column:', column)
    for c in tmp_column:
        if c in summary:
            summary[c] += 1
        elif len(summary) < K:
            summary[c] = 1
        else:
            raise ValueError('summary length must be less than K')
    keys, values = list(summary.keys()), list(summary.values())
    assert len(keys) == len(values)
    return keys, values
