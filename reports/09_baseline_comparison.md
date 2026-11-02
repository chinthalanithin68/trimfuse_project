# Dataset README baseline vs this project

| use_case                                   | readme                                                        | ours                                                                                           |
|:-------------------------------------------|:--------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| MCU bg predictive trim                     | 97.9% within +/-1; 6.6 -> 1.35 iterations                     | 99.4% within +/-1; 6.6 -> 1.31 iterations (ridge)                                              |
| MCU hsi predictive trim                    | 80.4% within +/-1; 7.6 -> 2.7 iterations                      | 80.4% within +/-1; 7.6 -> 2.69 iterations (hgb)                                                |
| MCU ldo predictive trim                    | 99.9% within +/-1; 5.6 -> 1.11 iterations                     | 100.0% within +/-1; 5.6 -> 1.14 iterations (ridge)                                             |
| MCU-C drift transfer (bandgap shift error) | 1.73 -> 0.86 mV (50 random dies)                              | 1.80 -> 0.78 mV (50 dies of one wafer, tested on the other)                                    |
| MCU-C bandgap final-test pass              | 90.9% -> 99.4%                                                | 90.7% -> 99.4% (tuned, other wafer)                                                            |
| Fuse-map validation                        | 406/406 injected errors, 0 false alarms (provided audit file) | 368/368 injected errors, 0 false alarms (our own injection on real words; audit file not used) |
| Fuse-map generation                        | not in README baseline                                        | 8,420/8,420 words regenerated identically (see 07_fuse_gen_results.md)                            |

The README baseline predicts all three trims with gradient-boosted trees. We pick ridge for bandgap and LDO because it measured better. The README tunes MCU-C on 50 random dies from both wafers; our test is stricter (tune on one wafer, test on the other).

## Final-test centring: bandgap Cpk (spec 1.200 V +/- 5 mV)

Drift-aware values come from leave-wafer-out predictions (see engine.py). Cpk >= 1.33 is the usual capability target.

| derivative   | dies                     |   cpk_trim_to_nominal |   cpk_drift_aware |   pass_nominal |   pass_drift_aware |
|:-------------|:-------------------------|----------------------:|------------------:|---------------:|-------------------:|
| MCU-A        | all dies                 |                  0.88 |              1.38 |          99.65 |              99.76 |
| MCU-A        | excluding latent defects |                  1.06 |              1.64 |          99.92 |             100    |
| MCU-B        | all dies                 |                  1.22 |              1.51 |          99.92 |              99.92 |
| MCU-B        | excluding latent defects |                  1.33 |              1.66 |         100    |             100    |
| MCU-C        | all dies                 |                  0.31 |              1.12 |          90.74 |              99.41 |
| MCU-C        | excluding latent defects |                  0.45 |              1.56 |          91.28 |             100    |

Latent defects add 5-15 mV of shift that the model cannot predict from sort data, which is why they are shown separately (they are held by the anomaly screen).