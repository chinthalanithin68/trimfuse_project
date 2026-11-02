# AI-Assisted Development Prompt Log
## MCU Manufacturing Test Optimization

Seven consolidated prompts that describe how this project is built from the provided data. They are a reconstruction of the development
workflow, not a literal transcript (see the Transparency Statement at the end).

---

## Prompt 1: Requirements Understanding and Dataset Selection

**Context:**
I am developing an MCU manufacturing test optimization system based on the provided design document, dataset README, data dictionary and the
`AI Trim & Fuse` dataset kit.

**Constraints:**
- Understand the complete requirements before implementation.
- Use only `mcu_trim_fuse.csv` as the approved dataset; do not introduce other tables without approval.
- Cover the three derivatives (MCU-A base, MCU-B, MCU-C automotive).
- Do not assume that documented features are already implemented.

**Task:**
Analyse the requirements and prepare a structured implementation plan covering predictive trim, temperature-sensor calibration, drift-aware
targets, transfer to small derivatives, fuse-map generation, fuse validation and anomaly screening. Identify the modules, evaluation metrics,
dependencies and deliverables, and list what the README baseline already proves.

**Done when:**
- The dataset choice and the reasons for it are documented.
- Requirements are mapped to modules and an implementation sequence.
- Assumptions and open questions are listed.

---

## Prompt 2: Dataset Understanding, Cleaning and Validation

**Context:**
The selected dataset is `mcu_trim_fuse.csv` (8,420 dies, 42 columns). A reliable data pipeline is needed before any model work.

**Constraints:**
- Use only the approved CSV and keep the original file unchanged.
- Model inputs come only from the dictionary's `feature` columns; identifiers, targets and post-test values are never inputs.
- Keep `wafer_id` for grouped splitting.
- Cleaning must be logged, never silent, and must never impute a trim code or fuse bit.

**Task:**
1. `data.py`: loading, a cleaning step (strip and normalise text, parse types, drop duplicate rows and duplicate die IDs, drop rows with a missing required value,
   drop out-of-range DAC codes, count but keep outliers), strict validation, fuse-word decoding, wafer-grouped splits.
2. `audit.py`: missing values, duplicates, ranges, label consistency, fuse-decode check, data patterns, and the cleaning log.
3. `DATASET.md`: dataset description, column roles, quirks and limitations.
4. Tests that prove each cleaning action on deliberately dirty data and that no label column is used as a model input.

**Done when:**
- `reports/01_data_quality_report.md` has a count for every check and the cleaning log.
- The cleaned data equals the raw data on the shipped file, and the tests show the cleaning works on dirty data.
- The feature-role test passes.

---

## Prompt 3: Predictive Trim Model Development

**Context:**
Machine-learning models should predict the optimal MCU trim codes so the tester can apply one code and verify instead of running a search loop.

**Constraints:**
- Use only the dictionary's feature columns.
- Split by `wafer_id` (5-fold grouped, out-of-fold predictions); no wafer in both train and test.
- Compare a median baseline, ridge regression and gradient boosting; choose per trim from held-out results.

**Task:**
Implement `trim_models.py` for bandgap, LDO and HSI trim codes. Report exact accuracy, within-one-code accuracy and mean absolute code error.
Simulate the tester flow (apply the predicted code and verify; off by one costs one step; further off falls back to a full search) and report iterations and
milliseconds against the conventional search. Compare against the README baseline (bandgap 97.9%, LDO 99.9%, HSI 80.4% within one code).

**Done when:**
- All three trims are evaluated with the chosen model and the reason for it.
- `reports/02_trim_results.md` shows every model and the time saved.
- Results are reproducible (fixed seeds).

---

## Prompt 4: Drift, Transfer Learning and Temperature Calibration

**Context:**
Derivatives differ in package and drift. Models trained on the base derivative must be tested on the small ones.

**Constraints:**
- Keep wafer-grouped evaluation; tune on one wafer and test on the others, never on test data.
- Distinguish zero-shot from tuned results; do not assume MCU-A models carry over to MCU-B or MCU-C without measuring it.
- Keep temperature-sensor evaluation separate from trim-code evaluation.

**Task:**
1. `drift.py`: predict the sort-to-final shift, trim to target minus shift, report final-test pass rate.
2. `transfer.py`: MCU-A models on MCU-B and MCU-C with 0 to 200 tuning dies (zero-shot, linear fix, pooled, own model); report accuracy and trim time.
3. `ts_cal.py`: predict the 110 C temperature-sensor reading from the 30 C reading and process data; train on MCU-C, test on MCU-A/B; report error and share within 2 C.
4. `baseline_compare.py`: compare against the README baseline, add bandgap Cpk before and after the drift-aware target, and state where the test setups differ.

**Done when:**
- Drift, transfer and temperature results each have a baseline next to them.
- Every improvement quoted is backed by a table in `reports/`.

---

## Prompt 5: Fuse Safety, Fuse-Map Generation and Anomaly Screening

**Context:**
Fuses are one-time programmable, so fuse words must be checked before programming, and abnormal dies should be held for review.

**Constraints:**
- Use only the approved dataset; do not reintroduce the separate fuse-schema and fuse-audit CSV files (embed the fuse layout in code).
- Keep the validator rule-based (no ML); state clearly that its test errors are injected by our own code, not the README audit benchmark.
- Do not invent encoding rules; take them from the dictionary and verify against the data.

**Task:**
1. `fuse_check.py`: rules from the fuse layout plus a cross-check against the sort-log codes; test with six injected error types and a flip of each of the 64 bits.
2. `fuse_gen.py`: build `fuse_map_hex` from the derivative spec and trim codes; regenerate all 8,420 stored words and compare; reject out-of-range values, unknown derivatives and missing fields.
3. `anomaly.py`: flag dies whose untrimmed readings disagree with what their process data predicts (robust z-score), scored against `latent_defect` without using it as a feature; compare with a rule and an Isolation Forest.

**Done when:**
- Errors caught, false alarms, the bit-flip table and the generator roundtrip count are in the reports.
- The anomaly result is reported with its caveat that the synthetic defects are unrealistically obvious.

---

## Prompt 6: Application Integration and UI Development

**Context:**
The implemented modules should be usable together through one interface for the demo.

**Constraints:**
- Keep logic out of the UI file: a backend module builds every result, the app only displays it.
- Predictions shown for a die must come from models that never saw that die's wafer.
- Every number on screen must come from the code or the generated reports, not be typed in.
- Widgets need unique keys (two identical selectboxes crashed the app once).

**Task:**
1. `engine.py`: run the trim, drift and anomaly models leave-wafer-out, apply the fuse rules, and return one row per die with a decision (block, hold, pass, one step, full search). The models are retrained on load; none are saved to disk.
2. `app.py` (Streamlit): an overview of time saved, accuracy and decisions; a die explorer with a wafer map; a fuse checker and generator; and a viewer for the generated reports (trim, drift, transfer, temperature sensor, fuse, anomaly, baseline comparison).

**Done when:**
- The app starts and every tab and widget branch runs in headless tests.
- Displayed values match the reports.

---

## Prompt 7: Documentation, Testing and Final Delivery

**Context:**
The project needs documentation, tests and deliverables suitable for a technical evaluation.

**Constraints:**
- Do not claim unbuilt features as done; do not invent results.
- State limitations plainly: synthetic data, modelled (not measured) time savings, weak HSI trim, the anomaly separation being a property of the generator.
- Keep documents consistent with the code and the reports.

**Task:**
1. `README.md`: overview, run commands, a deployment note, and a table mapping each evaluation point to its evidence.
2. `IMPLEMENTATION_PLAN.md`: scope, module table with results, validation method, limitations.
3. A one- to two-page Word summary with data-flow and architecture diagrams on an appendix page.
4. `requirements.txt` and this prompt log.
5. Run the full test suite and the headless app tests, and check that every number in the documents matches its report.

**Done when:**
- `python -m pytest -q tests` passes and the app starts.
- Documents match the implementation and the limitations are disclosed.

---

## Development Principles
1. Use one approved CSV dataset: `mcu_trim_fuse.csv`.
2. Split by `wafer_id`; never by random die.
3. Use only dictionary-defined feature columns as model inputs.
4. Prevent training/test leakage.
5. Fix random seeds so experiments repeat.
6. Report only verified metrics, each next to a baseline.
7. Document decisions and limitations.
8. Keep working functionality intact; every module ends with a report and tests.

## Transparency Statement
The project was developed through about 14 interactive steps, including two scope corrections (starting on the wrong dataset, then using two extra CSV files)
and one data-leakage correction (a label column used as a model input). These seven prompts are a consolidated reconstruction of that workflow, not a
literal transcript. They have not been replayed from scratch to establish exact reproducibility.
