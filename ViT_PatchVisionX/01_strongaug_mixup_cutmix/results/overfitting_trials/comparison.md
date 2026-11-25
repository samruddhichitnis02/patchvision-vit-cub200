| run | finished | epochs_run | best_val_acc1 | best_val_epoch | smoothed_val_acc1 | min_val_loss | min_val_loss_epoch | clean_train_acc_at_best | gap_at_best | val_attn_in_box | train_min | sec_per_epoch | test_acc1 (record only) | test_ece (record only) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | True | 246 | 42.33 | 186 | 40.80 | 2.72 | 114 | 99.50 | 59.83 | 0.35 | 48.42 | 11.81 | 39.52 | 2.49 |
| full | True | 213 | 48.50 | 153 | 48.38 | 2.32 | 140 | 98.17 | 49.83 | 0.66 | 71.20 | 20.06 | 47.39 | 5.95 |
| ovf1_droppath0.3 | True | 300 | 51.67 | 246 | 51.35 | 2.14 | 149 | 99.83 | 48.33 | 0.67 | 96.83 | 19.37 | 52.07 | 10.77 |
| ovf2_wd0.1 | True | 208 | 50.83 | 148 | 50.07 | 2.26 | 120 | 99.33 | 49.17 | 0.64 | 69.09 | 19.93 | 50.03 | 7.38 |
| ovf3_autoaug | True | 239 | 50.83 | 179 | 50.57 | 2.14 | 167 | 99.00 | 48.50 | 0.66 | 78.67 | 19.75 | 50.26 | 8.09 |
| ovf4_dropout0.1 | True | 253 | 51.00 | 193 | 50.82 | 2.23 | 120 | 99.83 | 48.83 | 0.63 | 86.19 | 20.44 | 48.08 | 5.70 |