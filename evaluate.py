import os
import time
import datetime

import torch
import numpy as np
import torch.nn as nn

from model.encoder import RegressionModel
from utils.model.dataset import load_dataset_features, make_feature_datasets, make_test_feature_dataloaders
from utils.model.padding import features_padding
from utils.model.perror_input import generate_perror_input
from utils.model.qerror import get_qerror, interval_qerror
from utils.model.args import get_args

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TEST_LIST = os.environ.get('PRICE_TEST_LIST', 'imdb').split(',')

args = get_args()
print(args)
current_dir = os.path.dirname(os.path.abspath(__file__))

test_data, test_labels, test_pg_est_cards, \
test_n_join_cols, test_n_fanouts, test_n_tables, test_n_filter_cols, test_list_lens = load_dataset_features(bin_size=args.bin_size, dataset_list=TEST_LIST, train_or_test='test', usage='test')

max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col = max(test_n_join_cols), max(test_n_fanouts), max(test_n_tables), max(test_n_filter_cols)
test_data, test_padding_masks = features_padding(args.bin_size, args.table_dim, args.filter_dim,
                                             test_data, test_n_join_cols, test_n_fanouts, test_n_tables, test_n_filter_cols, 
                                             max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
print("dataset padding done!!")
test_datasets_list = make_feature_datasets(test_data, test_labels, test_pg_est_cards, test_padding_masks,
                                      test_n_join_cols, test_n_fanouts, test_n_tables, test_n_filter_cols,
                                      train_or_test='test', test_list_lens=test_list_lens)
from torch.utils.data import DataLoader
test_loaders_list = [DataLoader(ds, batch_size=1, shuffle=False) for ds in test_datasets_list]

# our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionModel(n_join_col=max_n_join_col, n_fanout=max_n_fanout, n_table=max_n_table, n_filter_col=max_n_filter_col,
                        hist_dim=args.bin_size, table_dim=args.table_dim, filter_dim=args.filter_dim,
                        query_hidden_dim=args.query_hidden_dim, final_hidden_dim=args.final_hidden_dim, output_dim=args.output_dim,
                        n_embd=args.n_embd, n_layers=args.n_layers, n_heads=args.n_heads, dropout_rate=args.dropout_rate).to(device)
n_gpus = torch.cuda.device_count()
device_ids = list(range(n_gpus)) if n_gpus > 0 else None
if device_ids:
    model = nn.DataParallel(model, device_ids=device_ids)
else:
    model = nn.DataParallel(model)

model_path = os.environ.get('PRICE_MODEL_PATH', f'{current_dir}/results/model_params.pth')
print(f"load model from {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.module  # unwrap DataParallel for clean per-query timing

n_params = sum(p.numel() for p in model.parameters())
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_file_bytes = os.path.getsize(model_path)
print(f"model parameters: {n_params:,}")
print(f"model size (in-memory, fp32): {model_size_bytes / (1024 * 1024):.2f} MB")
print(f"model size (on-disk checkpoint): {model_file_bytes / (1024 * 1024):.2f} MB")

criterion = nn.MSELoss()

print('--'*30)
model.eval()
import re
for idx, current_dataloader in enumerate(test_loaders_list):
    test_loss = 0
    total_infer_time = 0.0
    total_samples = 0
    all_outputs = []
    all_labels = []
    workloads_path = f'{current_dir}/datas/workloads/test/{TEST_LIST[idx]}/workloads.sql'
    with open(workloads_path, 'r') as _f:
        _lines = _f.readlines()
    parent_ids = []
    for _ln in _lines:
        _m = re.search(r'@(\d+)-(\d+)@', _ln)
        if _m:
            parent_ids.append(int(_m.group(1)))
    per_query_infer_time = {}
    # warm-up: run a few forward passes so cuDNN/CUDA kernels are cached before timing
    with torch.no_grad():
        _warmup_iter = iter(current_dataloader)
        for _w in range(min(20, len(current_dataloader))):
            try:
                _d, _lbl, _pg, _pm, _njc, _nf, _nt, _nfc = next(_warmup_iter)
            except StopIteration:
                break
            _d = _d.to(torch.float).to(device)
            _njc = _njc.to(torch.float).to(device).view(-1, 1)
            _nf = _nf.to(torch.float).to(device).view(-1, 1)
            _nt = _nt.to(torch.float).to(device).view(-1, 1)
            _nfc = _nfc.to(torch.float).to(device).view(-1, 1)
            _pg = _pg.to(torch.float).to(device).view(-1, 1)
            _pg = torch.log(_pg + 1) + 1
            _pm = _pm.to(device)
            _ = model(_d, _pg, _pm, _njc, _nf, _nt, _nfc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    for i, (data, label, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col) in enumerate(current_dataloader):
        data = data.to(torch.float).to(device)
        n_join_col = n_join_col.to(torch.float).to(device).view(-1, 1)
        n_fanout = n_fanout.to(torch.float).to(device).view(-1, 1)
        n_table = n_table.to(torch.float).to(device).view(-1, 1)
        n_filter_col = n_filter_col.to(torch.float).to(device).view(-1, 1)
        pg_est_card = pg_est_card.to(torch.float).to(device).view(-1, 1)
        pg_est_card = torch.log(pg_est_card + 1) + 1
        padding_mask = padding_mask.to(device)
        label = torch.log(label.to(torch.float).to(device) + 1) + 1
        label = label.view(1, -1)

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.time()
            output = model(data, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col).view(1, -1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _elapsed = time.time() - infer_start
            total_infer_time += _elapsed
            total_samples += len(data)
            if i < len(parent_ids):
                _pid = parent_ids[i]
                per_query_infer_time[_pid] = per_query_infer_time.get(_pid, 0.0) + _elapsed
            loss = criterion(output, label)
            test_loss += loss.item() * len(data)
            all_outputs.append(output)
            all_labels.append(label)
    test_loss = test_loss / len(current_dataloader.dataset)
    output = torch.cat(all_outputs, dim=1)
    label = torch.cat(all_labels, dim=1)
    print(f"{TEST_LIST[idx]} loss: {test_loss}")
    print(f"{TEST_LIST[idx]} inference time: total {total_infer_time * 1000:.2f} ms over {total_samples} sub-queries, "
          f"avg {total_infer_time / max(total_samples, 1) * 1000:.3f} ms/sub-query")
    _n_top = len(per_query_infer_time)
    _top_times_ms = sorted([v * 1000 for v in per_query_infer_time.values()])
    _sum_top_ms = sum(_top_times_ms)
    _avg_top_ms = _sum_top_ms / max(_n_top, 1)
    _median_top_ms = _top_times_ms[_n_top // 2] if _n_top else 0.0
    _p95_top_ms = _top_times_ms[int(0.95 * (_n_top - 1))] if _n_top else 0.0
    _max_top_ms = _top_times_ms[-1] if _n_top else 0.0
    _min_top_ms = _top_times_ms[0] if _n_top else 0.0
    print(f"{TEST_LIST[idx]} inference time per top-level query ({_n_top} queries): "
          f"total {_sum_top_ms:.2f} ms, avg {_avg_top_ms:.3f} ms/query, "
          f"min {_min_top_ms:.3f} ms, median {_median_top_ms:.3f} ms, "
          f"p95 {_p95_top_ms:.3f} ms, max {_max_top_ms:.3f} ms")
    q_error = get_qerror(output, label, cuda=True, do_scale=True, percentile_list=[30, 50, 80, 90, 95, 99])
    print(f'{TEST_LIST[idx]} q-error: 30%:', q_error[0], '  50%:', q_error[1], '  80%:', q_error[2], '  90%:', q_error[3], '  95%:', q_error[4], '  99%:', q_error[5])
    interval_qerror(output, label, cuda=True, do_scale=True)

    # to generate p-error input file
    output1 = output[0].detach().cpu().numpy()
    workloads_test_file_path = f'{current_dir}/datas/workloads/test/{TEST_LIST[idx]}/workloads.sql'
    workloads_all_file_path = f'{current_dir}/datas/workloads/test/{TEST_LIST[idx]}/workloads_all.sql'
    out_path = f'{current_dir}/results/{TEST_LIST[idx]}_perror_input.sql'
    generate_perror_input(output1, out_path, workloads_test_file_path, workloads_all_file_path, True)

print('done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
