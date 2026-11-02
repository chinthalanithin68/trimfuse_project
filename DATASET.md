# Dataset: `data/mcu_trim_fuse.csv`

Synthetic MCU wafer-sort and final-test data from the "AI-Optimized Trim & Fuse for Derivative Chips" kit. No real product, fab or
customer data. 8,420 dies x 42 columns, no missing values or duplicates (see `reports/01_data_quality_report.md`).

| Derivative | Package | Wafers | Dies | Role in the project |
|---|---|---|---|---|
| MCU-A | QFN48, industrial | 15 | 6,315 | base derivative, most data |
| MCU-B | QFN32, industrial | 3 | 1,263 | small derivative |
| MCU-C | LQFP100, automotive | 2 | 842 | small derivative, largest package drift, only one measured hot in production |

## Column roles (from the dataset's data dictionary)
| Role | Columns | Used as |
|---|---|---|
| feature (known at test time) | `die_x`, `die_y`, `x_mm`, `y_mm`, `radius_mm`; `pcm_vth_n_mV`, `pcm_vth_p_mV`, `pcm_rs_ohm_sq`, `pcm_tox_A`, `pcm_mobility_rel`; `bg_untrim_V`, `hsi_untrim_MHz`, `ldo_untrim_V`, `ts_cal1_30C_counts`, `sram_fail_bits` | model inputs (the only allowed inputs; a test enforces it) |
| label (ground truth) | `bg_trim_code_opt`, `hsi_trim_code_opt`, `ldo_trim_code_opt`, `*_trimmed_sort_*`, `*_final_*`, `ts_cal2_110C_counts`, `ts_slope_counts_per_C_true`, `sram_repairable`, `sort_pass`, `final_pass`, `latent_defect`, `fuse_map_hex` | targets and scoring only |
| meta | `lot_id`, `wafer_id`, `derivative`, `package`, `temp_grade`, `*_conv_iterations`, `conv_trim_time_ms`, `ts_cal2_measured_on_tester` | grouping, baselines, time accounting |
| id | `die_id` | key |

## Things to know about the data
- Trim codes are the true best codes from an exhaustive search; the DAC widths are 6 bit (bandgap), 7 bit (HSI), 5 bit (LDO).
- `fuse_map_hex` is a 64-bit word: trims at bits 0-29, then flash size, peripheral mask, grade, debug lock, read-out protection and 21 reserved bits.
- `final_pass` requires `sort_pass`. `ts_cal2_*` is measured in production only for MCU-C.
- About 0.36% of dies are injected latent defects (30 dies); 97% of them fail final test.
- Always split by `wafer_id`: dies on one wafer are correlated.
