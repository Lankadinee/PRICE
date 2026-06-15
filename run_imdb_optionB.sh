#!/bin/bash
# =============================================================================
# run_imdb_optionB.sh
# One-shot: generate IMDB statistics FROM SCRATCH using PRICE's official
# gen_*.py scripts, then build features and evaluate the pretrained model.
#
# "Option B" = drive the repo's own statistics scripts. That requires two
# small inputs the scripts expect for a new dataset:
#   (1) an idempotent patch to utils/statistics/tools.py so load_table_datas
#       knows where the IMDB parquet lives, and
#   (2) a minimal postgres_create_imdb.sql so load_tbls_cols_types finds the
#       column types.
# Both are created/applied automatically below. tools.py is backed up to
# tools.py.orig; revert any time with:  git checkout utils/statistics/tools.py
#
# Usage:   bash run_imdb_optionB.sh
# =============================================================================
set -eo pipefail

# ---- config ----------------------------------------------------------------
PRICE_DIR="/datadrive500/PRICE"
PARQUET_DIR="/datadrive500/cardinality-estimation-data/processed_data/imdb"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="price"
BIN_SIZE=40
USAGE="test"
DB="imdb"

STATS_DIR="$PRICE_DIR/datas/statistics/$USAGE/$DB"
DDL_DIR="$PRICE_DIR/datas/datasets/$DB"
TOOLS="$PRICE_DIR/utils/statistics/tools.py"

echo "############################################################"
echo "# PRICE IMDB — Option B (official gen_*.py pipeline)"
echo "############################################################"

# ---- Step 0: environment ---------------------------------------------------
echo "== Step 0: activate env '$ENV_NAME' and ensure pyarrow =="
source "$CONDA_SH"
conda activate "$ENV_NAME"
python -c "import pyarrow" 2>/dev/null || pip install pyarrow
python -c "import torch,pandas,sqlglot,pyarrow; print('  env ok — cuda:', torch.cuda.is_available())"

# ---- Step 1: back up bundled stats so we provably regenerate from scratch ---
echo "== Step 1: back up & clear provided statistics =="
mkdir -p "$STATS_DIR" "$STATS_DIR/../imdb_provided_backup"
cp "$STATS_DIR"/*.pkl "$STATS_DIR/../imdb_provided_backup/" 2>/dev/null || true
rm -f "$STATS_DIR"/histogram${BIN_SIZE}.pkl "$STATS_DIR"/summary${BIN_SIZE}.pkl \
      "$STATS_DIR"/fanout${BIN_SIZE}.pkl   "$STATS_DIR"/size.pkl \
      "$STATS_DIR"/abbrev_col_type.pkl \
      "$STATS_DIR"/gen_histogram${BIN_SIZE}.pkl "$STATS_DIR"/gen_summary${BIN_SIZE}.pkl \
      "$STATS_DIR"/gen_fanout${BIN_SIZE}.pkl "$STATS_DIR"/gen_size.pkl

stats_start=$(date +%s.%N)

# ---- Step 2a: build abbrev_col_type.pkl (schema config) FROM SCRATCH --------
echo "== Step 2a: build abbrev_col_type.pkl =="
python - "$STATS_DIR" <<'PY'
import sys, pickle, os
stats_dir = sys.argv[1]
abbrev = {'title':'imdb_t','cast_info':'imdb_ci','movie_companies':'imdb_mc',
          'movie_info':'imdb_mi','movie_info_idx':'imdb_mii','movie_keyword':'imdb_mk'}
col_type = {  # join keys -> continuous (histogram); filter ids -> discrete (summary)
  'imdb_t':  {'ctn':['id','production_year'], 'dsct':['kind_id']},
  'imdb_ci': {'ctn':['movie_id'],             'dsct':['role_id']},
  'imdb_mc': {'ctn':['movie_id'],             'dsct':['company_type_id','company_id']},
  'imdb_mi': {'ctn':['movie_id'],             'dsct':['info_type_id']},
  'imdb_mii':{'ctn':['movie_id'],             'dsct':['info_type_id']},
  'imdb_mk': {'ctn':['movie_id'],             'dsct':['keyword_id']},
}
os.makedirs(stats_dir, exist_ok=True)
pickle.dump({'abbrev':abbrev,'col_type':col_type},
            open(os.path.join(stats_dir,'abbrev_col_type.pkl'),'wb'))
print('  wrote abbrev_col_type.pkl (6 tables)')
PY

# ---- Step 2b: minimal DDL so load_tbls_cols_types finds column types --------
echo "== Step 2b: write postgres_create_imdb.sql =="
mkdir -p "$DDL_DIR"
cat > "$DDL_DIR/postgres_create_${DB}.sql" <<'SQL'
create table title (
"id" integer,
"kind_id" integer,
"production_year" integer
);
create table cast_info (
"movie_id" integer,
"role_id" integer
);
create table movie_companies (
"movie_id" integer,
"company_id" integer,
"company_type_id" integer
);
create table movie_info (
"movie_id" integer,
"info_type_id" integer
);
create table movie_info_idx (
"movie_id" integer,
"info_type_id" integer
);
create table movie_keyword (
"movie_id" integer,
"keyword_id" integer
);
SQL

# ---- Step 2c: idempotently patch tools.py to know the IMDB parquet dir ------
echo "== Step 2c: patch tools.py (idempotent; backup -> tools.py.orig) =="
[ -f "${TOOLS}.orig" ] || cp "$TOOLS" "${TOOLS}.orig"
python - "$TOOLS" "$PARQUET_DIR" <<'PY'
import sys
tools, parquet_dir = sys.argv[1], sys.argv[2]
s = open(tools).read()
if 'IMDB_PARQUET_DIR' not in s:
    anchor = 'TPCH_UNIFORM_PARQUET_DIR = "/home/student.unimelb.edu.au/lrathuwadu/cardinality-estimation-data/original_data/tpch_uniform_sf1"'
    assert anchor in s, "anchor constant not found — tools.py layout changed"
    s = s.replace(anchor, anchor + f'\nIMDB_PARQUET_DIR = "{parquet_dir}"')
if '"imdb": IMDB_PARQUET_DIR' not in s:
    anchor = '        "tpch_uniform": TPCH_UNIFORM_PARQUET_DIR,\n    }.get(db)'
    assert anchor in s, "anchor dict not found — tools.py layout changed"
    s = s.replace(anchor, '        "tpch_uniform": TPCH_UNIFORM_PARQUET_DIR,\n        "imdb": IMDB_PARQUET_DIR,\n    }.get(db)')
open(tools, 'w').write(s)
print('  tools.py now maps imdb -> parquet')
PY

# ---- Step 3: run the OFFICIAL gen_*.py scripts -----------------------------
echo "== Step 3: generate statistics (histogram, summary, size, fanout) =="
cd "$PRICE_DIR/utils/statistics"
echo "  -> histogram"; python -u gen_histogram.py --db $DB --bs $BIN_SIZE --usage $USAGE > /tmp/imdb_b_hist.log   2>&1
echo "  -> summary";   python -u gen_summary.py   --db $DB --bs $BIN_SIZE --usage $USAGE > /tmp/imdb_b_sum.log    2>&1
echo "  -> size";      python -u gen_size.py      --db $DB              --usage $USAGE > /tmp/imdb_b_size.log   2>&1
echo "  -> fanout (slowest; ~4 min on 36M-row cast_info)";
python -u gen_fanout.py --db $DB --bs $BIN_SIZE --usage $USAGE > /tmp/imdb_b_fanout.log 2>&1

# gen_*.py emit "gen_<name>"; features_tool.py reads the names WITHOUT the prefix
echo "== Step 3b: rename gen_*.pkl -> names features_tool.py reads =="
cd "$STATS_DIR"
mv -f gen_histogram${BIN_SIZE}.pkl histogram${BIN_SIZE}.pkl
mv -f gen_summary${BIN_SIZE}.pkl   summary${BIN_SIZE}.pkl
mv -f gen_fanout${BIN_SIZE}.pkl    fanout${BIN_SIZE}.pkl
mv -f gen_size.pkl                 size.pkl
echo "  stats present:"; ls -1 "$STATS_DIR"/*.pkl | sed 's/^/    /'

stats_end=$(date +%s.%N)
awk "BEGIN{printf \"Time taken for statistics generation: %.2f seconds\n\", $stats_end-$stats_start}"

infer_start=$(date +%s.%N)
# ---- Step 4: build per-query features --------------------------------------
echo "== Step 4: generate features from workloads =="
cd "$PRICE_DIR/setup"
python -u features_generate.py --db $DB --bin_size $BIN_SIZE --usage $USAGE

# ---- Step 5: evaluate the pretrained model ---------------------------------
echo "== Step 5: evaluate pretrained model =="
cd "$PRICE_DIR"
PRICE_TEST_LIST=$DB python -u evaluate.py

infer_end=$(date +%s.%N)
# NOTE: this wall-clock covers feature-gen + model load + evaluate. The precise
# per-sub-query / per-query inference times are printed by evaluate.py above.
awk "BEGIN{printf \"Time taken for features+evaluation (wall clock): %.2f seconds\n\", $infer_end-$infer_start}"

echo "############################################################"
echo "# Done. Expected q-error (matches paper):"
echo "#   30%:1.3949 50%:1.7771 80%:4.0716 90%:8.3952 95%:15.4516 99%:70.8889"
echo "# Revert the tools.py patch with: git checkout utils/statistics/tools.py"
echo "############################################################"
