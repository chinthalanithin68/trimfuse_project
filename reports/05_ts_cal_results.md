# Temperature-sensor hot-point prediction (trained on MCU-C, 842 dies, 2 wafers)

| test_set                     | model                    | features     |   mae_counts |   rmse_counts |   mae_degC |   within_1C |   within_2C |
|:-----------------------------|:-------------------------|:-------------|-------------:|--------------:|-----------:|------------:|------------:|
| MCU-A (never measured hot)   | nominal slope (no model) | cal1         |        4.193 |         5.182 |      1.288 |       0.482 |       0.786 |
| MCU-A (never measured hot)   | ridge                    | cal1_only    |        9.311 |        10.221 |      2.862 |       0.082 |       0.253 |
| MCU-A (never measured hot)   | hgb                      | cal1_only    |        9.412 |        10.372 |      2.893 |       0.085 |       0.253 |
| MCU-A (never measured hot)   | ridge                    | cal1_pcm     |        2.641 |         3.306 |      0.816 |       0.669 |       0.951 |
| MCU-A (never measured hot)   | hgb                      | cal1_pcm     |        5.539 |         6.683 |      1.699 |       0.318 |       0.632 |
| MCU-A (never measured hot)   | ridge                    | cal1_pcm_pos |        2.654 |         3.323 |      0.82  |       0.667 |       0.95  |
| MCU-A (never measured hot)   | hgb                      | cal1_pcm_pos |        5.322 |         6.395 |      1.633 |       0.328 |       0.653 |
| MCU-B (never measured hot)   | nominal slope (no model) | cal1         |        3.926 |         4.832 |      1.216 |       0.514 |       0.82  |
| MCU-B (never measured hot)   | ridge                    | cal1_only    |        8.615 |         9.668 |      2.658 |       0.128 |       0.325 |
| MCU-B (never measured hot)   | hgb                      | cal1_only    |        8.624 |         9.698 |      2.662 |       0.136 |       0.325 |
| MCU-B (never measured hot)   | ridge                    | cal1_pcm     |        2.606 |         3.28  |      0.81  |       0.675 |       0.952 |
| MCU-B (never measured hot)   | hgb                      | cal1_pcm     |        4.476 |         5.543 |      1.381 |       0.421 |       0.747 |
| MCU-B (never measured hot)   | ridge                    | cal1_pcm_pos |        2.621 |         3.293 |      0.815 |       0.676 |       0.951 |
| MCU-B (never measured hot)   | hgb                      | cal1_pcm_pos |        4.446 |         5.483 |      1.372 |       0.426 |       0.761 |
| MCU-C held-out wafer MC1-W02 | nominal slope (no model) | cal1         |        7.43  |         8.191 |      2.402 |       0.131 |       0.423 |
| MCU-C held-out wafer MC1-W02 | ridge                    | cal1_only    |        3.015 |         3.817 |      0.974 |       0.648 |       0.91  |
| MCU-C held-out wafer MC1-W02 | hgb                      | cal1_only    |        2.975 |         3.732 |      0.96  |       0.589 |       0.893 |
| MCU-C held-out wafer MC1-W02 | ridge                    | cal1_pcm     |        2.758 |         3.405 |      0.89  |       0.62  |       0.943 |
| MCU-C held-out wafer MC1-W02 | hgb                      | cal1_pcm     |        2.772 |         3.485 |      0.892 |       0.644 |       0.933 |
| MCU-C held-out wafer MC1-W02 | ridge                    | cal1_pcm_pos |        2.96  |         3.675 |      0.956 |       0.587 |       0.91  |
| MCU-C held-out wafer MC1-W02 | hgb                      | cal1_pcm_pos |        3.016 |         3.799 |      0.973 |       0.58  |       0.881 |

True slope per die is used only to convert count errors to degrees C for scoring.