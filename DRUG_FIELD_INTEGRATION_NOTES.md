# Drug field integration notes

This branch adds a first-pass diffusing antifungal field to Pycelium.

## 1. New module: `compute/drug_field.py`

Adds `DrugField2D`, a vectorised NumPy finite-difference field. The field stores concentration as a 2D array indexed as `concentration[y_index, x_index]`, but it can be sampled using normal Pycelium `MPoint` x/y coordinates.

Implemented features:

- one diffusion update per Pycelium growth step;
- explicit 2D diffusion with a five-point Laplacian stencil;
- numerical stability check for `alpha = D * dt / dx^2 <= 0.25`;
- no-flux, absorbing, and constant boundary modes;
- optional first-order drug decay;
- megaplate-style vertical section initialisation;
- square outside-in perimeter-source initialisation;
- optional maintained square perimeter reservoir for continuous inward diffusion;
- bilinear sampling at continuous tip coordinates;
- Hill-style growth inhibition based on local concentration and MIC;
- final field exports to `.npy`, `.csv`, and `.png`.

## 2. New Options fields: `core/options.py`

Adds drug-field parameters to the main `Options` dataclass. These include field size, grid spacing, diffusion coefficient, boundary mode, optional vertical sections, optional square outside-in perimeter fields, wildtype MIC, Hill coefficient, minimum growth multiplier, and export toggles.

The default `Options` value keeps `drug_field_enabled = False` so existing simulations behave as before unless the drug field is explicitly enabled.

## 3. Updated configs

`config/param_config.json` now contains all drug-field keys, with `drug_field_enabled` set to `false` for backward compatibility.

`config/param_config_drug_test.json` is now a small demonstration config with the drug field enabled and square outside-in perimeter diffusion turned on. `config/param_config_drug_square_test.json` is an explicitly named copy of the same square-field test config.

Example command:

```bash
python -m launcher.run --mode cli --config config/param_config_drug_test.json --steps 50
```

## 4. Updated simulation setup: `main.py`

`setup_simulation()` now creates a `DrugField2D` object when `opts.drug_field_enabled` is true.

If `opts.drug_use_vertical_sections` is true, the initial field is divided into x-oriented drug bands using `drug_initial_x_edges` and `drug_initial_concentrations`.

If `opts.drug_use_square_perimeter` is true, the field is reset to `drug_square_interior_concentration`, an outer square frame of width `drug_square_perimeter_width` is set to `drug_square_perimeter_concentration`, and the frame is optionally maintained every diffusion substep when `drug_square_maintain_perimeter` is true. This produces diffusion from all four edges toward the centre.

The resulting object is stored in the `components` dictionary as `components["drug_field"]`.

## 5. Updated simulation step ordering: `main.py`

`step_simulation()` now performs:

```text
1. diffuse drug field once, if enabled
2. rebuild field aggregator from current mycelial sections
3. compute new tip orientations
4. grow/branch/prune mycelium with local drug response
5. update density/statistics/checkpoints
```

This means tips always respond to the antifungal concentration after the latest diffusion step.

## 6. Updated growth response: `core/mycel.py`

`Mycel.step()` now accepts an optional `drug_field` argument.

Each active tip calculates:

```text
effective_growth_rate = options.growth_rate * drug_growth_multiplier
```

where:

```text
drug_growth_multiplier = min + (1 - min) / (1 + (local_concentration / MIC)^hill)
```

The resulting `effective_growth_rate` is passed into `Section.grow()`. This preserves the existing `Section.grow()` length-scaled growth logic, because length scaling is applied after drug inhibition.

## 7. Section-level MIC placeholder: `core/section.py`

Each `Section` now carries `section.drug_mic`.

Current behaviour:

- seed sections start with `opts.drug_wildtype_mic`;
- child sections inherit their parent's `drug_mic`;
- no MIC mutation is applied yet.

This is intentionally ready for the next pass, where new branches can mutate MIC multiplicatively to create resistant breakout lineages.

## 8. Updated exports: `io_utils/exporter.py` and `main.py`

Final segment CSVs now include:

- `drug_mic`
- `drug_concentration`
- `drug_growth_multiplier`

Tip time-series CSVs also include those fields while retaining the original `time`, `x`, `y`, and `z` columns required by the animation code.

When enabled, the final drug field is exported as:

```text
drug_field_final.npy
drug_field_final.csv
drug_field_final.png
```

## 9. Validation performed

Validated with:

```bash
python -m compileall -q .
python -m launcher.run --mode cli --config config/param_config_drug_test.json --steps 5
python -m launcher.run --mode cli --config config/param_config_drug_square_test.json --steps 5
python -m launcher.run --mode cli --config config/param_config.json --steps 2
```

Both the drug-enabled and default drug-disabled configurations completed successfully.
