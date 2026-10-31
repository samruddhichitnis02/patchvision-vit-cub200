| Comparison (A vs B) | A minus B, top-1 points | photos only A gets right | photos only B gets right | p-value (exact paired test) | significant at 5%? |
|---|---|---|---|---|---|
| PatchVision-X vs CNN small | +3.76 | 742 | 524 | 0.0000 | yes |
| PatchVision-X vs CNN large | +1.55 | 668 | 578 | 0.0117 | yes |
| CNN large vs CNN small | +2.21 | 478 | 350 | 0.0000 | yes |
| PatchVision-X vs ViT base | +7.85 | 789 | 334 | 0.0000 | yes |

All three of PatchVision-X, CNN small and CNN large are right on 1829 photos and all three are wrong on 2295 photos (of 5794).