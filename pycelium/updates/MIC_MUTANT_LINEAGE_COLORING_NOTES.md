# MIC mutant lineage colouring

This update colours MIC-mutant breakout lineages consistently across all visual outputs.

## What changed

- Wildtype/non-mutant hyphae use a shared dark wildtype colour when `drug_mic_lineage_coloring_enabled = true`.
- Every time a new daughter section acquires a fresh MIC mutation, that daughter starts a new colour lineage.
- All descendants of that mutant inherit the same colour until a later MIC mutation creates another mutant sub-lineage.
- The 2D overlay PNG, 2D/3D PNGs, Plotly 3D HTML, OBJ export, final CSV, time-series CSV, and 2D MP4 now all keep those colours consistent.
- The 2D MP4 stores per-frame segment and tip colours so new breakout lineages can be seen emerging from within the colony.

## New config options

```json
"drug_mic_lineage_coloring_enabled": true,
"drug_mic_wildtype_color": [0.1, 0.1, 0.1],
"drug_mic_palette_saturation": 0.8,
"drug_mic_palette_value": 0.95
```

## New CSV metadata

- `mic_mutation_lineage_id`
- `mic_mutation_origin_section_id`
- `visual_color_source`
- `r`, `g`, `b` (time-series export)

`mic_mutation_lineage_id = 0` corresponds to the shared wildtype colour lineage.
