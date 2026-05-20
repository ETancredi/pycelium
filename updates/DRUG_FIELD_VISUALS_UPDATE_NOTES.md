# Drug-field visualisation update

This update improves antifungal-field visual outputs.

## What changed

1. `drug_field_initial.png` and `drug_field_final.png` now use a single-hue
   colour scale: white = no drug, dark blue = high drug.

2. A new combined output can be generated:

   ```text
   mycelium_drug_overlay.png
   ```

   This plots the final mycelium over the final antifungal field, using black
   hyphae with a white outline so the network remains visible on both white and
   dark-blue regions.

3. New config options:

   ```json
   "generate_mycelium_drug_overlay_png": true,
   "drug_plot_max_concentration": null
   ```

   Set `drug_plot_max_concentration` to a number such as `8.0` when you want
   multiple runs to share the same drug colour scale. Leave it as `null` to
   auto-scale each run to its own maximum concentration.
