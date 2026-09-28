# Yaskira baseline vs owner Human Gold — 2026-09-27

Offline interval comparison of saved baseline outputs from runs 36321269066 and 36324899917 against `CUTSELL_YASKIRA_01_10_HUMAN_GOLD.md`. No provider call or render was performed. Metrics use the owner's rounded whole-second labels and saved exact engine boundaries, so they are diagnostic approximations, not frame-accurate acceptance scores.

| Video | Gold recall | Precision | Approved content lost | Unwanted content retained |
| ---: | ---: | ---: | ---: | ---: |
| 01 | 89.5% | 99.0% | 12.06 s | 1.00 s |
| 02 | 95.0% | 97.5% | 4.80 s | 2.33 s |
| 03 | 91.2% | 100.0% | 1.72 s | 0.00 s |
| 04 | 87.7% | 99.1% | 4.66 s | 0.29 s |
| 05 | 94.9% | 97.6% | 1.97 s | 0.92 s |
| 06 | 100.0% | 98.1% | 0.00 s | 0.43 s |
| 07 | pending exact RAW alignment | pending | pending | known retained repeat 5:27.56–5:42.56 |
| 08 | 0.0% | 0.0% | 27.00 s | 52.00 s |
| 09 | 53.9% | 89.2% | 33.65 s | 4.78 s |
| 10 | 91.2% | 64.0% | 3.18 s | 18.47 s |

## Priority

1. **Wrong winning take / scene selection:** Video08 selected earlier takes (36.79–116.55) and missed the sole approved 120–147 interval. This is the clearest next general defect.
2. **False deletion of distinct approved scenes:** Video09 lost 33.65 seconds of Human Gold; later V17 improved parts of this but still omitted the 97–106 demonstration and retained boundary debris.
3. **Excess retention across multiple attempts:** Video10 retained 18.47 seconds outside Human Gold.
4. **False deletion inside mostly continuous story:** Video01 lost about 12 seconds even though the owner approved the source except 58–59.
5. Videos02–06 are much closer. Exact acceptance needs boundary-aware tolerances because owner labels are rounded.

## Next bounded correction

Historical proposal: use Video08 to diagnose one general winning-take failure. The former two-cycle limit is superseded by `CUTSELL_EDITORIAL_CALIBRATION_MASTER.md`: investigation has no fixed cycle count, and each paid run needs a recorded hypothesis or diagnostic/reproducibility objective within authorization. Regression checks include 05, 06 and 09, followed by a complete 01–10 pass on the same version.
