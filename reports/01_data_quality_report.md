# Data-Quality Audit - MCU trim & fuse dataset (mcu_trim_fuse.csv)

## 1. Shape and schema

- mcu_trim_fuse: (8420, 42); fuse layout (embedded): (10, 7)
- dtypes: {'float64': 19, 'int64': 11, 'str': 7, 'bool': 5}

## 2. Missing values

- columns with nulls: none
- blank strings in object columns: 0

## 3. Duplicates and keys

- duplicate rows: 0; duplicate die_id: 0
- duplicate (wafer_id, die_x, die_y): 0

## 4. Structure

```
            dies  wafers        pkg         grade
derivative                                       
MCU-A       6315      15    [QFN48]  [industrial]
MCU-B       1263       3    [QFN32]  [industrial]
MCU-C        842       2  [LQFP100]  [automotive]
```
- lots: 5, wafers: 20, wafers spanning >1 derivative: 0
- derivative-to-package mapping unique: {'MCU-A': 1, 'MCU-B': 1, 'MCU-C': 1}
- radius_mm vs sqrt(x^2+y^2): max abs diff 0.005 mm
- ts_cal2_measured_on_tester True only for: ['MCU-C'] (share of that derivative: 1.00)

## 5. Range checks (trim codes vs DAC bits, targets)

- bg_trim_code_opt: min=0, max=63, valid range 0..63, out of range=0, at range edge (0 or max)=15
- hsi_trim_code_opt: min=6, max=125, valid range 0..127, out of range=0, at range edge (0 or max)=0
- ldo_trim_code_opt: min=9, max=29, valid range 0..31, out of range=0, at range edge (0 or max)=0
- bg_trimmed_sort_V: mean abs error vs target 1.2 = 0.00055, max=0.03839, dies beyond +/-0.005: 10
- hsi_trimmed_sort_MHz: mean abs error vs target 16 = 0.01626, max=0.04805, dies beyond +/-0.16: 0
- ldo_trimmed_sort_V: mean abs error vs target 1.1 = 0.00125, max=0.00326, dies beyond +/-0.022: 0
- sram_fail_bits range 0..4; conv_trim_time_ms 26.2..34.6
- conv iterations (bg/hsi/ldo) range: 6-8, 7-9, 5-7

## 6. Fuse word cross-check (decode fuse_map_hex and compare to columns)

- BG_TRIM decoded vs bg_trim_code_opt: mismatches = 0
- HSI_TRIM decoded vs hsi_trim_code_opt: mismatches = 0
- LDO_TRIM decoded vs ldo_trim_code_opt: mismatches = 0
- TS_CAL1 decoded vs ts_cal1_30C_counts: mismatches = 0
- RESERVED bits non-zero in main table: 0; DEBUG_LOCK=0: 0

## 7. Label consistency

- sort_pass=False: 34; final_pass=False: 128; final fail though sort pass: 94
- final_pass recomputed from spec vs stored: mismatches=29 (among sort-pass dies: 1)
- final fail rate by derivative: {'MCU-A': 0.0073, 'MCU-B': 0.0024, 'MCU-C': 0.0938}
- latent defects: 30 (0.36%); final fail among them: 0.97 vs 0.012 for others

## 8. Outliers (IQR x 3) on features

- pcm_vth_n_mV: extreme outliers = 0
- pcm_vth_p_mV: extreme outliers = 0
- pcm_rs_ohm_sq: extreme outliers = 0
- pcm_tox_A: extreme outliers = 0
- pcm_mobility_rel: extreme outliers = 0
- bg_untrim_V: extreme outliers = 18
- hsi_untrim_MHz: extreme outliers = 0
- ldo_untrim_V: extreme outliers = 0
- ts_cal1_30C_counts: extreme outliers = 0
- latent defects vs trim-code edge rate: 15 defective dies with BG code at edge

## 9. Pattern checks (what the models should exploit)

- corr(bg_untrim_V, bg_trim_code_opt) = -0.997
- corr(hsi_untrim_MHz, hsi_trim_code_opt) = -0.996
- corr(ldo_untrim_V, bg_untrim_V) = 0.605 (coupling)
- corr(ts_cal1, ts_cal2) = 0.969
- bandgap sort-to-final shift (mV) by derivative: {'MCU-A': 1.8, 'MCU-B': 0.98, 'MCU-C': 3.59}; corr(shift, radius) = 0.136

## 10. Cleaning step applied (data.clean_mcu, run on the raw file)

| action | count |
|---|---|
| rows_in | 8420 |
| duplicate_rows_dropped | 0 |
| duplicate_die_id_dropped | 0 |
| rows_dropped_missing_required | 0 |
| missing_required_by_column | {} |
| missing_in_other_columns_kept | {} |
| rows_dropped_code_out_of_range | 0 |
| extreme_outlier_values_kept | 18 |
| fuse_word_vs_trim_column_mismatches_kept | 0 |
| rows_out | 8420 |

Policy: duplicates are dropped (first kept); a row with a missing required value is dropped, never imputed, because a guessed trim code or fuse bit is unsafe; out-of-range DAC codes are dropped; text is stripped and fuse words normalised to 0x + 16 upper-case hex digits; outliers and fuse-vs-column mismatches are only counted. On this file every drop count is 0, so the cleaned data equals the raw data; tests/test_data_and_trim.py proves each action on deliberately dirty data.