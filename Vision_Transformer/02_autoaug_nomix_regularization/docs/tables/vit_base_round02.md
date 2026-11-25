| Run | What changes | Best epoch | **Val top-1** | vs r0 (photos) | Gap (pts) | Test top-1 | Test top-5 | Test macro-F1 | Test ECE |
|---|---|---|---|---|---|---|---|---|---|
| r0_reference | nothing (new base recipe) | 141 | 39.17 % |  | 61.3 | 36.24 % | 61.81 % | 35.68 % | 11.08 % |
| r1_droppath0.2 | drop-path 0.1 → 0.2 | 99 | 40.17 % | +6 | 57.8 | 37.33 % | 63.96 % | 37.02 % | 9.46 % |
| **r2_droppath0.4** | drop-path 0.1 → 0.4 | 299 | **42.67 %** | +21 | 57.3 | 39.30 % | 63.48 % | 38.72 % | 7.25 % |
| r3_droppath0.5 | drop-path 0.1 → 0.5 | 258 | 42.67 % | +21 | 57.2 | 38.97 % | 65.88 % | 38.54 % | 9.41 % |
| r4_labelsmooth0.2 | label smoothing 0.1 → 0.2 | 205 | 39.83 % | +4 | 60.2 | 36.24 % | 61.49 % | 35.64 % | 2.61 % |
| r5_wd0.1 | weight decay 0.05 → 0.1 | 233 | 38.00 % | -7 | 62.3 | 35.61 % | 59.86 % | 34.88 % | 4.47 % |
| r6_wd0.2 | weight decay 0.05 → 0.2 | 137 | 40.33 % | +7 | 60.3 | 36.90 % | 62.46 % | 36.44 % | 10.18 % |
| r7_sam | SAM optimizer (rho 0.05) | 122 | 37.67 % | -9 | 63.0 | 34.28 % | 59.99 % | 33.84 % | 10.42 % |
| r8_combined | winners of r1-r7 combined | *pending* | *pending* | | | | | | |
