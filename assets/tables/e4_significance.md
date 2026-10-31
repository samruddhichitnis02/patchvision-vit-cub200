| Comparison (A vs B) | A minus B, top-1 points | Photos only A gets right | Photos only B gets right | p-value (exact paired test) | Significant at 5%? |
|---|---|---|---|---|---|
| E4 drop-path 0.3 + auto-aug (ovf5) vs PatchVision-X (full) | +5.63 | 658 | 332 | <0.0001 | yes |
| E4 drop-path 0.3 + auto-aug (ovf5) vs drop-path 0.3 (ovf1) | +0.97 | 481 | 425 | 0.0676 | no |
| E4 all four (ovf6) vs PatchVision-X (full) | +5.06 | 606 | 313 | <0.0001 | yes |
| E4 all four (ovf6) vs drop-path 0.3 (ovf1) | +0.40 | 437 | 414 | 0.4508 | no |
| E4 drop-path 0.3 + auto-aug (ovf5) vs E4 all four (ovf6) | +0.57 | 396 | 363 | 0.2454 | no |
| E4 drop-path 0.3 + auto-aug (ovf5) vs CNN large | +7.18 | 854 | 438 | <0.0001 | yes |
| E4 drop-path 0.3 + auto-aug (ovf5) vs CNN small | +9.39 | 950 | 406 | <0.0001 | yes |