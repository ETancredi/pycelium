# Drug field control fix notes

This update addresses a control-test problem where intended uniform-drug configs could still behave like the square outside-in setup if the legacy square-source boolean was left enabled.

## What changed

1. Added `drug_initial_condition` to `core/options.py`.

Recommended values:

- `uniform`: use `drug_initial_background_concentration` everywhere.
- `vertical_sections`: use `drug_initial_x_edges` and `drug_initial_concentrations`.
- `square_perimeter`: use the outside-in square-frame settings.
- `legacy`: fall back to the older boolean switches `drug_use_vertical_sections` and `drug_use_square_perimeter`.

2. Updated `main.py` so the explicit `drug_initial_condition` decides which initial condition is applied.

3. Added console diagnostics at run start, for example:

```text
🧪 Drug field initialised mode=uniform shape=(101, 101) min=8 max=8 mean=8 centre=8 alpha=0.05
```

This lets you immediately catch a bad config before waiting for output files.

4. Added initial field exports:

- `drug_field_initial.npy`
- `drug_field_initial.csv`
- `drug_field_initial.png`

These are exported before any diffusion or mycelial growth, so they show the actual starting condition.

5. Added corrected test configs under:

```text
config/test_configs/
```

The key control configs are:

```text
param_config_drug_control_none.json
param_config_drug_control_1xMIC_uniform.json
param_config_drug_control_8xMIC_uniform.json
param_config_drug_square_origin_1xMIC.json
```

## Expected behaviour

With the included test configs and the same fixed seed:

- No drug control: normal growth.
- Uniform 1xMIC: uniform field at concentration 1.0; growth multiplier around 0.5 under the current Hill response.
- Uniform 8xMIC: uniform field at concentration 8.0; with hill coefficient 8, growth multiplier is about 5.96e-8 and the run should essentially stop immediately.
- Square origin 1xMIC: centre starts at 1.0, perimeter starts at 8.0, and drug diffuses inward from the square frame.
