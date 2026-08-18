# Why the leakage-controlled observed-IMV model performs better

## Conclusion

The day-shift-1 improvement is real on the available outputs, but it does not show that leakage is harmless. On exactly the same 523,008 test timestamps, day shift 1 lowers macro RMSE from 536.312 to 454.712 (15.21%), lowers MAE from 227.109 to 186.970 (17.67%), and raises NSE from 0.920917 to 0.940788.

Leakage only makes more information available; it does not guarantee that a finite LSTM trained once will generalize better. The evidence supports a combination of hydrologic response lag, removal of a noisy same-day shortcut, and a better optimization outcome. A single seed cannot yet distinguish their causal contributions.

## Checks that rule out simple artifacts

- Day 0 contains 525,312 reconstructed test rows. Day 1 contains 523,008 because its first day has no prior-day IMV. Restricting both models to the 523,008 shared timestamps leaves the conclusion unchanged.
- All 144 stored metric values per experiment were recomputed from the reconstructed CSVs and matched the JSON summaries exactly.
- Observed targets on the overlap match to CSV/scaler serialization precision.
- Both configurations use the same model architecture, feature count, data splits, optimizer settings, and seed. The intended differences are the processed-data directory and IMV date alignment.
- Shift 1 improves reconstructed MAE in all 12 watersheds and RMSE in 9 of 12. It also improves MAE in low, middle, and high observed-flow bands, so the result is not driven by one watershed or one flow regime.

## Common-timestamp test metrics

| Metric | Day shift 0 | Day shift 1 | Shift-1 change | Shift-1 watershed wins |
|---|---:|---:|---:|---:|
| RMSE | 536.312 | 454.712 | -15.21% | 9 / 12 |
| MAE | 227.109 | 186.970 | -17.67% | 12 / 12 |
| NSE | 0.920917 | 0.940788 | +0.019871 | 9 / 12 |
| KGE | 0.866694 | 0.880447 | +0.013753 | 7 / 12 |

Absolute bias is the notable exception: the original summary gives 73.262 for day 0 and 77.393 for day 1, so day 0 is slightly better on that metric.

## Why day shift 1 can perform better

### 1. Prior-day IMVs can align better with outlet response

On the test split, the macro mean correlations with current hourly streamflow are stronger at lag 1 for several hydrologically important inputs:

| IMV | Same-day correlation | Prior-day correlation |
|---|---:|---:|
| SUPY | 0.1800 | 0.2750 |
| WYIE | -0.1490 | -0.2126 |
| AGW | 0.7770 | 0.7905 |
| TWS | 0.6451 | 0.6485 |

This is consistent with routing and storage delays: water supplied or stored on day D-1 can be more informative about outlet flow on day D than a completed summary for day D repeated over all its hours. This is evidence for a plausible mechanism, not a causal proof.

### 2. Leakage can create a poor shortcut

The same-day daily row is temporally coarse and partly future-looking for early hours. Although it contains extra information, repeating one completed daily aggregate across 24 targets can encourage the LSTM to learn a shortcut that is noisy or phase-misaligned at hourly resolution. Lagging the features removes that shortcut and can act as useful regularization.

### 3. The shifted model optimized better

The selected day-1 checkpoint reached validation loss 0.04809 at epoch 64. The selected day-0 checkpoint reached 0.05423 at epoch 25. The lower day-1 validation loss is consistent with its test improvement. It does not by itself prove whether the cause is physical lag or training dynamics.

### 4. The gain is broad across flow regimes

| Observed-flow band | Day-0 MAE | Day-1 MAE | Mean within-watershed change | Shift-1 wins |
|---|---:|---:|---:|---:|
| Low (at or below P50) | 80.37 | 56.83 | -31.36% | 11 / 12 |
| Middle (P50 to P90) | 212.94 | 187.96 | -13.03% | 10 / 12 |
| High (above P90) | 1017.39 | 833.62 | -15.19% | 10 / 12 |

## The sign and timestamp semantics are correct

The implementation advances each daily IMV timestamp by one day:

```python
daily_values["date"] = daily_values["date"] + pd.to_timedelta(day_shift, unit="D")
```

Consequently, source value D is joined to target date D+1, which means target date D+1 receives the previous day's value. The plus sign is therefore correct.

The archived Kettle River data contain paired hourly and daily IMV outputs. Aggregating hourly flux values by their own calendar date matches the daily row much better than assigning the aggregate to either adjacent day. For SUPY, mean absolute mismatch is 0.0128 on the same date, versus 0.0986 one date earlier and 0.1009 one date later. PET and ET show the same pattern. Thus day 0 truly includes information accumulated during the current day; day shift 1 is not merely correcting a one-day label error.

## What remains uncertain and how to prove the cause

Only one seed is available for each alignment. The result therefore establishes performance for these runs, not that lag 1 is universally superior. The baseline directory also contains multiple training logs, so run provenance should be tightened.

The decisive next experiment is:

1. Train day shifts 0, 1, and 2 with 5–10 identical paired seeds.
2. Evaluate every run on the identical timestamp intersection.
3. Save each checkpoint and training history in a seed-specific directory.
4. Report paired mean differences and confidence intervals.
5. Ablate feature groups, especially SUPY/WYIE/AGW/TWS versus PET/ET/LZS/SNOW.

No chart is included because there are only two treatments and 12 watersheds; exact tables retain the audit trail and expose small differences more clearly.

## Sources

- `experiments/hourly_global_streamflow_observed_IMVs/test_results/analysis_results/test/summary_metrics.json`
- `experiments/hourly_global_streamflow_observed_IMVs_day_shift_1/test_results/analysis_results/test/summary_metrics.json`
- Per-watershed reconstructed test CSVs under both experiments' `test_results` directories
- `data_processed` and `data_processed/day_shift_1`
- `data/flow_data_20260127_bak/KettleRiverModels_outlet_hist_scaled_Daily_outputs.csv`
- `data/flow_data_20260127_bak/KettleRiverModels_outlet_hist_scaled_Hourly_outputs.csv`
- `config_global.yaml`
