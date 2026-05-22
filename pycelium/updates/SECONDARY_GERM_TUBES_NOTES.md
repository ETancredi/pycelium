# Secondary germ-tube emergence logic

This update adds optional delayed secondary germ tubes to Pycelium, so a spore can begin with one primary germ tube and later produce one or more additional germ tubes from the original origin.

## Biological idea

Aspergillus conidia often germinate first in one direction, then after a lag can initiate another germ tube from the same spore body. This is different from ordinary lateral branching because the new tip starts at the original germination point, not at the current hyphal apex.

## New Options fields

```python
secondary_germ_tubes_enabled: bool = False
secondary_germ_tube_min_step: int = 2
secondary_germ_tube_max_step: int = 10
secondary_germ_tube_min_count: int = 1
secondary_germ_tube_max_count: int = 4
secondary_germ_tube_extra_probability: float = 0.35
secondary_germ_tube_extra_probability_decay: float = 0.5
secondary_germ_tube_timing_shape: float = 1.0
secondary_germ_tube_angle_degrees: float = 180.0
secondary_germ_tube_angle_spread: float = 25.0
```

## How it works

When `secondary_germ_tubes_enabled` is true, the model schedules delayed germ-tube emergence at seeding time.

1. At least `secondary_germ_tube_min_count` delayed germ tubes are scheduled.
2. Extra delayed germ tubes are added using a geometric-like probability tail, so 1 is most likely, 2 is less likely, 3 is less likely again, and so on.
3. Emergence steps are drawn from `secondary_germ_tube_min_step` to `secondary_germ_tube_max_step` with linearly increasing weights by default.
4. Each delayed germ tube starts at the original spore origin.
5. Its orientation is approximately `secondary_germ_tube_angle_degrees` relative to the primary germ tube, with ± `secondary_germ_tube_angle_spread` degrees of jitter.

## Output metadata

`mycelium_final.csv` and `mycelium_time_series.csv` now include:

- `germ_tube_order`: 0 for the primary germ tube, 1+ for delayed secondary germ tubes.
- `germ_tube_role`: `primary_germ_tube`, `secondary_germ_tube`, or `branch`.
- `emergence_step`: the scheduled step at which a germ-tube lineage appeared.
- `emergence_time`: the simulation time when it appeared.

## Example config

An example enabled config is included at:

```bash
config/default_configs/param_config_secondary_germ_tubes.json
```

Run it with:

```bash
python3 -m launcher.run --mode cli \
  --config config/default_configs/param_config_secondary_germ_tubes.json \
  --steps 30
```
