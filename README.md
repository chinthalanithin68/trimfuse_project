# AI-Optimized Trim & Fuse - MCU

Predictive trim, drift-aware targets, fuse-map validation and generation, and latent-defect screening for an MCU family, built on
`data/mcu_trim_fuse.csv` only (synthetic, 8,420 dies, 3 derivatives). See `DATASET.md` for the data and `IMPLEMENTATION_PLAN.md` for scope,
status and limitations.

## Run it
```bash
python -m pip install -r requirements.txt       # Python 3.14 tested
python -m pytest -q tests                        # 49 tests
python -m streamlit run app.py                   # UI at http://localhost:8501 (first load trains models, ~30 s)
```
Reproduce every report (each script writes its own file in `reports/`):
```bash
python src/audit.py            # data-quality audit + cleaning log
python src/trim_models.py      # predictive trim
python src/drift.py            # drift-aware targets, Cpk inputs
python src/transfer.py         # trim-code transfer to MCU-B/C (slow, several minutes)
python src/ts_cal.py           # temperature-sensor hot point
python src/fuse_check.py       # fuse-map validator
python src/fuse_gen.py         # fuse-map generator + roundtrip
python src/anomaly.py          # latent-defect screen
python src/baseline_compare.py # dataset-README baseline vs ours, Cpk
```
**Deployment:** the app is a local, offline Streamlit app. `requirements.txt` pins the tested versions, there is no network or API call at
run time, and a server run is `python -m streamlit run app.py --server.port 8501 --server.headless true`. It has not been deployed to a hosted
service, and no model is exported to a tester (the plan notes ONNX / lookup tables as the route).

## Where each evaluation point is covered
| Evaluation point | Evidence |
|---|---|
| Requirement understanding | `IMPLEMENTATION_PLAN.md` (objective, techniques); `reports/00_implementation_summary.docx` section 1 |
| Implementation summary (1-2 pages) | `reports/00_implementation_summary.docx` (diagrams on a last appendix page) |
| Dataset cleaning | `src/data.py` `clean_mcu()` (duplicates, missing values, ranges, types, outlier policy, fuse-vs-column check) and `validate_mcu()`; log in `reports/01_data_quality_report.md` section 10; proven on dirty data in `tests/test_data_and_trim.py` |
| Data-quality checks | `src/audit.py` -> `reports/01_data_quality_report.md` (nulls, duplicates, ranges, fuse decode, label consistency, patterns) |
| Model training and validation | `src/trim_models.py`, `drift.py`, `transfer.py`, `ts_cal.py`, `anomaly.py`: wafer-grouped splits, baselines, label/feature-role test |
| Model selection | `reports/02_trim_results.md` (ridge vs gradient boosting vs median, per trim) and `reports/00_implementation_summary.docx` section 5 |
| Offline vs online | Offline: scikit-learn, local, no API at run time (summary section 5) |
| End-to-end flow | `prompts/prompt_log.md` (requirements -> prompts -> code -> tests -> run); `tests/` (49 tests); the Run section above |
| UI | `app.py`: Overview, Die explorer with wafer map, Fuse checker and generator, Method & reports |
| Code and data-flow walkthrough | `diagrams/data_flow.png`, `diagrams/architecture.png`; module table in `IMPLEMENTATION_PLAN.md` |
| Results and limitations | `reports/` (one file per module), `reports/09_baseline_comparison.md`; limitations in the plan and summary |

## Reading order
1. `README.md` (this file), then `DATASET.md` for the data and `IMPLEMENTATION_PLAN.md` for scope, results per module and limitations.
2. `reports/00_implementation_summary.docx`: the 1-2 page summary with the diagrams.
3. `reports/01` to `09`: results in pipeline order: data quality and cleaning, predictive trim, drift, transfer to MCU-B/C, temperature sensor, fuse validator,
   fuse generator, anomaly screen, comparison with the README baseline.
4. `src/` in pipeline order: `data.py`, `audit.py`, `trim_models.py`, `drift.py`, `transfer.py`, `ts_cal.py`, `fuse_check.py`, `fuse_gen.py`, `anomaly.py`,
   `baseline_compare.py`, `engine.py` (UI backend), then `app.py`.
5. `tests/`, then `prompts/prompt_log.md`.

## Layout
```
data/mcu_trim_fuse.csv      the dataset
src/                        data, clean/validate, models, fuse rules, engine (UI backend)
app.py                      Streamlit UI
tests/                      pytest suite
reports/                    generated results + 00_implementation_summary.docx
diagrams/                   architecture and data-flow PNGs
prompts/prompt_log.md       prompt log
```
Models are offline (scikit-learn, local, no API at inference). Claude was used as the development assistant.
