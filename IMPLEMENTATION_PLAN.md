# Implementation Plan and Status - AI Predictive Trim & Fuse Validation (MCU)

**Dataset:** `data/mcu_trim_fuse.csv` only (8,420 dies x 42 columns; source: `AI Trim & Fuse/Dataset/.../data/`). The 64-bit `fuse_map_hex` column is inside this table. The fuse bit layout (documented in the dataset's data dictionary) is embedded in `src/data.py`. No other CSV is used.

**Status:** audit, trim models, drift/transfer, temperature sensor, fuse validator, anomaly screen and Streamlit UI are built; 49 tests pass. Summary (reports/00_implementation_summary.docx), diagrams and prompt log are written. Still to do: rehearsal and any UI polish.

## Why this table
- The PDF's own rollout step 1 (MCU oscillator and bandgap trim): simplest, high volume, quickest proof of savings.
- Clean (no nulls, no duplicates) with clear labels: optimal trim codes (exhaustive-search truth), post-package final values, latent-defect flag, fuse words.
- Three derivatives with 15 / 3 / 2 wafers, so transfer learning (MCU-A to MCU-C) and drift-aware targets are real, testable questions.
- Covers five of the PDF's nine techniques from one table: predictive trim, transfer learning, drift-aware targets, fuse-map validation, anomaly screening (plus the temperature-sensor insertion removal).
- Other tables in the folder (PMIC, sensor, SoC, memory repair, OTP requests) are wider or need solvers; not used.

## Objective
Replace the 6-8 iteration binary-search trim (28.7 ms/die) with one predicted code plus a verification step, keep escapes at zero, block bad fuse words before OTP is blown, and flag latent-defect dies.

## Audit findings (reports/01_data_quality_report.md)
- No nulls, duplicates or out-of-range codes. Fuse words decode exactly to the trim columns (0 mismatches); reserved bits all 0; debug lock always set.
- Blanks/flags that look odd but are fine: `ts_cal2_*` is measured in production only for MCU-C; `final_pass` also requires sort pass (explains 29 apparent mismatches).
- All 15 dies with the bandgap code at the DAC edge are latent defects. Latent defects fail final test 97% vs 1.2% for others.
- MCU-C package shifts the bandgap about 3.6 mV (A 1.8, B 1.0) and fails final 9.4% (A 0.7%).
- Heavy tails in `bg_untrim_V` (18 extreme values) are signal; no clipping.

## What was built and what it showed
| Module | Technique | Key result |
|---|---|---|
| `src/data.py` (`clean_mcu`, `validate_mcu`) | Cleaning step: strip/normalise text, parse types, drop duplicates and rows with missing required values (never impute), drop out-of-range DAC codes; count (not remove) outliers and fuse-vs-column mismatches | Log in the audit report: every drop count is 0 on this file, so cleaned data equals raw; each action is proven on deliberately dirty data in the tests. |
| `src/trim_models.py` | Predictive single-shot trim, wafer-grouped 5-fold on MCU-A | Total trim 28.7 ms to 8.1 ms (72% saved). Within +/-1 code: bandgap 99.4% (ridge), LDO 100% (ridge), HSI 80.4% (gradient boosting). Linear models win where the relation is linear; HSI looks capped by unobservable per-die gain. |
| `src/drift.py` | Drift-aware targets + transfer MCU-A to MCU-C | MCU-C bandgap final pass 90.7% to 99.4%. Gain is mostly a per-derivative constant offset (about 10 dies are enough); radius adds little. |
| `src/transfer.py` | Trim-code transfer MCU-A to MCU-B/C (how many tuning dies a new derivative needs) | MCU-B needs no tuning (7.9 ms/die untuned). MCU-C is 9.3 ms untuned and 8.4 ms with 10 tuning dies (pooled); training on the new derivative alone needs 200+ dies and is still worse (10.4 ms). |
| `src/ts_cal.py` | Hot-point (110 C) from 30 C + PCM, trained on MCU-C | Ridge on cal1 + PCM: 95% within 2 C on MCU-A/B. cal1-only and tree models generalise worse. |
| `src/fuse_check.py` | Rule-based fuse validator (no ML) | 368/368 injected errors caught, 0 false alarms on 1,132 clean words; every single-bit flip caught when the sort log is available; trim-field flips need the sort log. |
| `src/fuse_gen.py` | Fuse-map generator from derivative spec + sort codes | Regenerates 8,420/8,420 stored words identically; all pass the validator; rejects out-of-range codes. |
| `src/baseline_compare.py` | Dataset-README baseline vs ours, plus bandgap Cpk | Trim results match or slightly beat the baseline. MCU-C bandgap Cpk 0.31 to 1.12 (0.45 to 1.56 excluding latent defects). |
| `src/anomaly.py` | Latent-defect screen (PCM-residual z-score) | Flags all 30 latent defects, no false flags. Separation is unrealistically clean (synthetic 50-80 mV defect offset). |
| `src/engine.py`, `app.py` | Streamlit UI, leave-wafer-out predictions | Overview, die explorer with wafer map, fuse checker with bit flipping, report viewer. |

Tester decision order per die: fuse violation = BLOCK; anomaly z >= 6 = HOLD; predicted code verified = PASS; off by 1 = one local step; further off = full binary-search FALLBACK.

## Validation method
- Split by `wafer_id` (GroupKFold); dies on a wafer are correlated, so random die splits would overstate results.
- Transfer tests tune on one MCU-C wafer and test on the other (both directions, 20 draws).
- Always compared against a baseline (median code, conventional search, nominal slope, constant shift).
- Labels (`latent_defect`, `final_pass`, optimal codes) are never used as features (a test enforces this).

## Model choice and offline vs online
- Ridge and histogram gradient boosting (scikit-learn), chosen per trim from measured results. Small tabular data, fast, explainable, exportable to a lookup table or ONNX for a tester. A neural net would add nothing.
- **Offline**: trained and run locally, no API at inference. Deterministic, milliseconds, data stays in the fab, version-lockable (needed for automotive). Claude was used as the development assistant only.

## Safeguards
Fuses are one-time programmable: never program on a prediction alone; always verify; fall back to conventional search on low confidence; hard-limit codes to the valid range; security fuses (debug lock, RDP) use rule checks only, never ML.

## Known limitations (say these before they are asked)
- Synthetic data: results show the method works, not production numbers. Time savings are modelled from the dataset's iteration counts and per-iteration costs, not measured on a tester.
- HSI trim stays near 80% within +/-1 code, so about 20% of dies fall back to a full search.
- MCU-C trim predictions in the UI are untuned (9.3 ms/die); about 10 tuning dies bring it to 8.4 ms (reports/04_transfer_results.md). HSI on MCU-C is weaker (77% untuned, about 78-80% tuned).
- The anomaly result is clean because the generator makes defects obvious.
- The fuse-validator test set is generated by our own corruption code, so its error types are ours.

## Code layout
```
trimfuse_project/
  data/mcu_trim_fuse.csv
  requirements.txt  DATASET.md
  src/  audit.py data.py trim_models.py drift.py transfer.py ts_cal.py fuse_check.py fuse_gen.py anomaly.py baseline_compare.py engine.py
  app.py            Streamlit UI (python -m streamlit run app.py)
  tests/            49 pytest tests
  reports/          audit + one results file per module
  prompts/          prompt_log.md
  diagrams/         architecture.png, data_flow.png
```
Data flow: CSV -> validate (data.py) -> features -> models (leave-wafer-out) -> tester-flow simulation (predict, verify, fallback) + fuse rule check + anomaly screen -> metrics and UI.

## Remaining milestones
Done: summary, diagrams, prompt log.
To do: add your own prompts to the log, rehearse to 45 minutes.
