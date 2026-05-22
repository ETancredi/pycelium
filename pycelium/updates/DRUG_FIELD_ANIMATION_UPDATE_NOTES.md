# Drug-field animation update

This update changes the 2D growth MP4 from a tip-dot movie into a contextual
network movie.

## What changed

- `core/mycel.py` now records lightweight per-step 2D network snapshots when
  `generate_mycelium_growth_mp4` is enabled. Each frame stores visible hyphal
  line segments and living tip positions.
- `main.py` now records antifungal concentration snapshots during 2D drug runs,
  so the MP4 can show diffusion through time.
- `vis/animate_growth.py` has a new `animate_network_growth_2d()` function. It
  draws the antifungal heatmap behind the colony and overlays connected hyphal
  line segments with the same black/white-halo style used by
  `mycelium_drug_overlay.png`.
- The existing `generate_mycelium_growth_mp4` toggle still controls MP4 output.
  In 2D mode it now writes the upgraded `mycelium_growth_2d.mp4`. In 3D mode the
  older point-based 3D animation path is retained.

## New optional settings

```json
"mycelium_growth_mp4_interval_ms": 100,
"mycelium_growth_mp4_dpi": 150,
"mycelium_growth_mp4_show_drug_colorbar": true
```

These are optional because `Options` provides defaults. Existing configs should
continue to run unchanged.
