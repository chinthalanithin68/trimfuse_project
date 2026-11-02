# Fuse-map generator results

- Words regenerated from derivative spec + sort codes: 8420
- Identical to the stored `fuse_map_hex`: 8420/8420
- Generated words that pass the validator (with the sort log): 8420/8420

## Invalid inputs

- bandgap code 64 (6-bit field): rejected: BG_TRIM=64 does not fit in 6 bits
- negative LDO code: rejected: LDO_TRIM=-1 does not fit in 5 bits
- unknown derivative: rejected: unknown derivative 'MCU-Z'
- missing HSI code: rejected: codes must be exactly ('BG_TRIM', 'HSI_TRIM', 'LDO_TRIM', 'TS_CAL1')