# Fuse-map validator results

Test set: 1,500 real fuse words from mcu_trim_fuse.csv, about 25% corrupted by the 6 error types in `inject_error` (seeded, regenerated on each run).

Words: 1500, injected errors: 368 (24.5%).

- Errors caught: 368/368
- False alarms on clean words: 0/1132
- Fired rule names the injected error type: 100.0%

Injected error type vs rule(s) fired:

| error_type             |   DEBUG_LOCK |   FLASH_SIZE |   PERIPH_EN |   RDP |   RESERVED |   SORT_MISMATCH |
|:-----------------------|-------------:|-------------:|------------:|------:|-----------:|----------------:|
| missing_debug_lock     |          111 |            0 |           0 |     0 |          0 |               0 |
| periph_mask_mismatch   |            0 |            0 |          61 |     0 |          0 |               0 |
| rdp_missing_automotive |            0 |            0 |           0 |    11 |          0 |               0 |
| reserved_bit_set       |            0 |            0 |           0 |     0 |         54 |               0 |
| trim_field_corrupt     |            0 |            0 |           0 |     0 |          0 |              61 |
| wrong_flash_size       |            0 |           70 |           0 |     0 |          0 |               0 |

## Without the sort log (schema rules only)

- Errors caught: 307/368; missed types: {'trim_field_corrupt': 61}

## Single-bit-flip mutation test (any of 64 bits flipped on 300 good words)

With sort log:

| field      |   detected |
|:-----------|-----------:|
| BG_TRIM    |          1 |
| DEBUG_LOCK |          1 |
| FLASH_SIZE |          1 |
| HSI_TRIM   |          1 |
| LDO_TRIM   |          1 |
| PERIPH_EN  |          1 |
| RDP        |          1 |
| RESERVED   |          1 |
| TEMP_GRADE |          1 |
| TS_CAL1    |          1 |

Schema rules only:

| field      |   detected |
|:-----------|-----------:|
| BG_TRIM    |          0 |
| DEBUG_LOCK |          1 |
| FLASH_SIZE |          1 |
| HSI_TRIM   |          0 |
| LDO_TRIM   |          0 |
| PERIPH_EN  |          1 |
| RDP        |          1 |
| RESERVED   |          1 |
| TEMP_GRADE |          1 |
| TS_CAL1    |          0 |