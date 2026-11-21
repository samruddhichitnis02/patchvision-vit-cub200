| Run | What changes | Best epoch | **Val top-1** | vs r0 | Gap (pts) | Test top-1 | Test top-5 | Test macro-F1 | Test ECE |
|---|---|---|---|---|---|---|---|---|---|
| r0_reference | nothing (new base recipe) | 165 | 45.83 % |  | 54.2 | 46.03 % | 71.87 % | 45.65 % | 4.74 % |
| r1_droppath0.2 | drop-path 0 -> 0.2 | 194 | 46.50 % | +0.67 | 53.8 | 46.62 % | 72.02 % | 46.39 % | 3.67 % |
| r2_droppath0.4 | drop-path 0 -> 0.4 | 216 | 45.00 % | -0.83 | 55.2 | 44.62 % | 71.16 % | 44.64 % | 2.33 % |
| r3_droppath0.5 | drop-path 0 -> 0.5 | 276 | 43.17 % | -2.67 | 56.7 | 43.55 % | 70.42 % | 43.71 % | 1.99 % |
| r4_labelsmooth0.2 | label smoothing 0.1 -> 0.2 | 237 | 48.50 % | +2.67 | 52.0 | 47.89 % | 71.56 % | 47.47 % | 23.01 % |
| r5_wd0.1 | weight decay 0.05 -> 0.1 | 199 | 47.67 % | +1.83 | 52.8 | 47.89 % | 71.97 % | 47.49 % | 9.86 % |
| **r6_wd0.2** | weight decay 0.05 -> 0.2 | 211 | **50.50 %** | +4.67 | 50.0 | 50.28 % | 73.85 % | 49.92 % | 12.62 % |
| r7_sam | SAM optimizer (rho 0.05) | 150 | 47.33 % | +1.50 | 52.7 | 46.82 % | 71.80 % | 46.60 % | 4.64 % |
| r8_combined | winners of r1-r7 combined | *pending* | *pending* | | | | | | |
