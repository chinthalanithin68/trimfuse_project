# Drift-aware targets and transfer learning

## 1. Drift model inside MCU-A (wafer-grouped out-of-fold)

| trim   | model   | unit        |   mae_naive |   mae_model |   pass_conv |   pass_drift_aware |
|:-------|:--------|:------------|------------:|------------:|------------:|-------------------:|
| bg     | ridge   | mV          |      0.702  |      0.69   |      0.9965 |             0.9976 |
| bg     | hgb     | mV          |      0.702  |      0.7214 |      0.9965 |             0.9978 |
| hsi    | ridge   | % of 16 MHz |      0.096  |      0.0958 |      1      |             1      |
| hsi    | hgb     | % of 16 MHz |      0.096  |      0.0994 |      1      |             1      |
| ldo    | ridge   | mV          |      1.2364 |      1.2288 |      1      |             1      |
| ldo    | hgb     | mV          |      1.2364 |      1.2644 |      1      |             1      |

## 2. Zero-shot: MCU-A model applied to MCU-B and MCU-C

| derivative   | trim   | unit        |   mae_zero_shot |   mae_naive_mean_of_A |   pass_conv |   pass_zero_shot |
|:-------------|:-------|:------------|----------------:|----------------------:|------------:|-----------------:|
| MCU-B        | bg     | mV          |          0.9702 |                0.9728 |      0.9992 |           0.9992 |
| MCU-B        | hsi    | % of 16 MHz |          0.1031 |                0.1028 |      1      |           1      |
| MCU-B        | ldo    | mV          |          1.4324 |                1.4317 |      1      |           1      |
| MCU-C        | bg     | mV          |          1.7964 |                1.8033 |      0.9074 |           0.9941 |
| MCU-C        | hsi    | % of 16 MHz |          0.2351 |                0.2352 |      1      |           1      |
| MCU-C        | ldo    | mV          |          2.9543 |                2.9602 |      1      |           1      |

## 3. Transfer to MCU-C: tune on n dies of one wafer, test on the other (both directions, 20 draws)


### bg - MAE of shift (mV)

|   n_tune |   c_only |   mean_offset |   pooled_hgb |   scale_correct |   zero_shot |
|---------:|---------:|--------------:|-------------:|----------------:|------------:|
|        0 |  nan     |       nan     |      nan     |         nan     |       1.796 |
|       10 |    0.889 |         0.801 |        1.774 |           0.792 |       1.796 |
|       25 |    0.804 |         0.799 |        0.799 |           0.771 |       1.796 |
|       50 |    0.773 |         0.789 |        0.802 |           0.758 |       1.796 |
|      100 |    0.759 |         0.789 |        0.824 |           0.745 |       1.796 |

### bg - final-spec pass rate with drift-aware target

|   n_tune |   c_only |   mean_offset |   pooled_hgb |   scale_correct |   zero_shot |
|---------:|---------:|--------------:|-------------:|----------------:|------------:|
|        0 | nan      |      nan      |     nan      |        nan      |      0.9941 |
|       10 |   0.9904 |        0.9941 |       0.994  |          0.9941 |      0.9941 |
|       25 |   0.9941 |        0.9941 |       0.9936 |          0.9941 |      0.9941 |
|       50 |   0.9941 |        0.9941 |       0.9932 |          0.9941 |      0.9941 |
|      100 |   0.9941 |        0.9941 |       0.992  |          0.9941 |      0.9941 |

### hsi - MAE of shift (% of 16 MHz)

|   n_tune |   c_only |   mean_offset |   pooled_hgb |   scale_correct |   zero_shot |
|---------:|---------:|--------------:|-------------:|----------------:|------------:|
|        0 |  nan     |       nan     |      nan     |         nan     |       0.235 |
|       10 |    0.116 |         0.102 |        0.233 |           0.109 |       0.235 |
|       25 |    0.103 |         0.1   |        0.103 |           0.1   |       0.235 |
|       50 |    0.1   |         0.099 |        0.105 |           0.099 |       0.235 |
|      100 |    0.098 |         0.099 |        0.107 |           0.098 |       0.235 |

### hsi - final-spec pass rate with drift-aware target

|   n_tune |   c_only |   mean_offset |   pooled_hgb |   scale_correct |   zero_shot |
|---------:|---------:|--------------:|-------------:|----------------:|------------:|
|        0 |      nan |           nan |          nan |             nan |           1 |
|       10 |        1 |             1 |            1 |               1 |           1 |
|       25 |        1 |             1 |            1 |               1 |           1 |
|       50 |        1 |             1 |            1 |               1 |           1 |
|      100 |        1 |             1 |            1 |               1 |           1 |

### ldo - MAE of shift (mV)

|   n_tune |   c_only |   mean_offset |   pooled_hgb |   scale_correct |   zero_shot |
|---------:|---------:|--------------:|-------------:|----------------:|------------:|
|        0 |  nan     |       nan     |      nan     |         nan     |       2.954 |
|       10 |    1.577 |         1.4   |        2.923 |           1.422 |       2.954 |
|       25 |    1.375 |         1.346 |        1.375 |           1.318 |       2.954 |
|       50 |    1.326 |         1.331 |        1.399 |           1.299 |       2.954 |
|      100 |    1.294 |         1.326 |        1.418 |           1.283 |       2.954 |

### ldo - final-spec pass rate with drift-aware target

|   n_tune |   c_only |   mean_offset |   pooled_hgb |   scale_correct |   zero_shot |
|---------:|---------:|--------------:|-------------:|----------------:|------------:|
|        0 |      nan |           nan |          nan |             nan |           1 |
|       10 |        1 |             1 |            1 |               1 |           1 |
|       25 |        1 |             1 |            1 |               1 |           1 |
|       50 |        1 |             1 |            1 |               1 |           1 |
|      100 |        1 |             1 |            1 |               1 |           1 |

Conventional (trim to nominal) MCU-C final-spec pass: bg 90.7%, hsi 100.0%, ldo 100.0%