# Latent-defect screen (8420 dies, 30 latent defects = 0.36%)

Scores are computed from sort-time data only; `latent_defect` is used to score.

Metric note: recall_at_k flags the k highest-scoring dies where k = number of true latent defects.

| method                                   |   auc |    ap |   recall_at_k |   final_fail_in_top_k |
|:-----------------------------------------|------:|------:|--------------:|----------------------:|
| residual ridge: max |z| over 4 readings  | 1     | 1     |           1   |                 0.967 |
| residual ridge: bandgap only             | 1     | 1     |           1   |                 0.967 |
| residual hgb: max |z| over 4 readings    | 1     | 1     |           1   |                 0.967 |
| rule: bandgap code at DAC edge (0 or 63) | 0.75  | 0.502 |           0.5 |                 0.6   |
| IsolationForest on raw features          | 0.918 | 0.023 |           0   |                 0     |

## Operating points for the ridge residual screen (max |z|)

|   z_threshold |   flagged |   flagged_pct |   caught |   recall |   precision |   final_fail_among_flagged |
|--------------:|----------:|--------------:|---------:|---------:|------------:|---------------------------:|
|             4 |        30 |         0.356 |       30 |    1     |           1 |                      0.967 |
|             6 |        30 |         0.356 |       30 |    1     |           1 |                      0.967 |
|             8 |        30 |         0.356 |       30 |    1     |           1 |                      0.967 |
|            10 |        29 |         0.344 |       29 |    0.967 |           1 |                      0.966 |
|            15 |        12 |         0.143 |       12 |    0.4   |           1 |                      1     |

Final-test fail rate overall: 1.52%; among true latent defects: 97%