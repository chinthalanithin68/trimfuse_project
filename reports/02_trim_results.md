# Predictive trim results (MCU-A, wafer-grouped 5-fold, out-of-fold)

| trim   | feature_set   | model   |   exact |   within1 |   mae_codes |   conv_iters |   ai_iters |   conv_ms |   ai_ms |
|:-------|:--------------|:--------|--------:|----------:|------------:|-------------:|-----------:|----------:|--------:|
| bg     | base          | median  |   0.074 |     0.208 |       4.639 |        6.59  |      5.884 |     7.907 |   7.061 |
| bg     | base          | ridge   |   0.725 |     0.994 |       0.282 |        6.59  |      1.306 |     7.907 |   1.567 |
| bg     | base          | hgb     |   0.743 |     0.959 |       0.345 |        6.59  |      1.46  |     7.907 |   1.752 |
| hsi    | base          | median  |   0.022 |     0.052 |      19.08  |        7.595 |      7.667 |    15.19  |  15.334 |
| hsi    | base          | ridge   |   0.202 |     0.558 |       1.571 |        7.595 |      4.452 |    15.19  |   8.905 |
| hsi    | base          | hgb     |   0.484 |     0.804 |       0.799 |        7.595 |      2.694 |    15.19  |   5.389 |
| hsi    | chained       | median  |   0.022 |     0.052 |      19.08  |        7.595 |      7.667 |    15.19  |  15.334 |
| hsi    | chained       | ridge   |   0.202 |     0.557 |       1.571 |        7.595 |      4.458 |    15.19  |   8.916 |
| hsi    | chained       | hgb     |   0.481 |     0.802 |       0.805 |        7.595 |      2.707 |    15.19  |   5.413 |
| ldo    | base          | median  |   0.156 |     0.451 |       2.082 |        5.594 |      4.042 |     5.594 |   4.042 |
| ldo    | base          | ridge   |   0.861 |     1     |       0.139 |        5.594 |      1.139 |     5.594 |   1.139 |
| ldo    | base          | hgb     |   0.873 |     0.991 |       0.142 |        5.594 |      1.164 |     5.594 |   1.164 |
| ldo    | chained       | median  |   0.156 |     0.451 |       2.082 |        5.594 |      4.042 |     5.594 |   4.042 |
| ldo    | chained       | ridge   |   0.86  |     1     |       0.14  |        5.594 |      1.14  |     5.594 |   1.14  |
| ldo    | chained       | hgb     |   0.873 |     0.991 |       0.141 |        5.594 |      1.162 |     5.594 |   1.162 |

## Best model per trim (lowest verified-flow time)

| trim   | feature_set   | model   |   exact |   within1 |   mae_codes |   conv_iters |   ai_iters |   conv_ms |   ai_ms |
|:-------|:--------------|:--------|--------:|----------:|------------:|-------------:|-----------:|----------:|--------:|
| bg     | base          | ridge   |   0.725 |     0.994 |       0.282 |        6.59  |      1.306 |     7.907 |   1.567 |
| hsi    | base          | hgb     |   0.484 |     0.804 |       0.799 |        7.595 |      2.694 |    15.19  |   5.389 |
| ldo    | base          | ridge   |   0.861 |     1     |       0.139 |        5.594 |      1.139 |     5.594 |   1.139 |

Total trim time per die: conventional 28.7 ms -> predictive 8.1 ms (72% saved). Dies: 6315, wafers: 15.