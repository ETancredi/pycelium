# MIC response update notes

This update replaces the original multiplier-only drug response with an MIC-centred pharmacodynamic response.

## Why this changed

The first drug-field implementation used:

```text
multiplier = min + (1 - min) / (1 + (A / MIC)^k)
```

That curve treats `MIC` like a half-inhibition point: when `A == MIC`, growth is halfway between full growth and the configured floor. That was useful for testing drug-field plumbing, but it does **not** match the biological interpretation we now want.

## New model

The default model is now:

```text
Ψ = Ψmax - ((Ψmax - Ψmin) * (A / MIC)^k)
           / ((A / MIC)^k - (Ψmin / Ψmax))
```

where:

```text
Ψmax = normal drug-free growth rate, taken from options.growth_rate
Ψmin = high-drug asymptotic growth rate, set by drug_min_growth_rate
A    = local antifungal concentration sampled from the drug field
MIC  = tip-specific MIC, currently section.drug_mic
k    = drug_hill_coefficient
```

Important property:

```text
A == MIC  ->  Ψ == 0
```

This requires `Ψmax > 0` and `Ψmin < 0`. For the default `growth_rate = 1.0`, a sensible first value is:

```json
"drug_min_growth_rate": -1.0
```

## New options

```json
"drug_response_model": "pharmacodynamic",
"drug_min_growth_rate": -1.0,
"drug_nonpositive_growth_action": "stall"
```

`drug_nonpositive_growth_action` controls what happens when the pharmacodynamic model returns zero or negative growth:

```text
stall -> the tip remains alive but does not extend
kill  -> the tip is marked dead and removed from active growth
```

The old multiplier curve is still available with:

```json
"drug_response_model": "hill_multiplier"
```

In that legacy mode, `drug_min_growth_multiplier` is still used.

## Export changes

The mycelium CSV exports now include:

```text
drug_raw_growth_rate       signed Ψ from the response equation; can be negative
drug_effective_growth_rate non-negative rate actually passed to Section.grow()
drug_growth_multiplier     applied effective rate / growth_rate after clamping
```

This makes it possible to distinguish between high-drug inhibition and ordinary slow growth.

## Tested controls

With the supplied test configs:

```bash
python -m launcher.run --mode cli --config config/test_configs/param_config_drug_control_1xMIC_uniform.json --steps 5
python -m launcher.run --mode cli --config config/test_configs/param_config_drug_control_8xMIC_uniform.json --steps 5
python -m launcher.run --mode cli --config config/test_configs/param_config_drug_square_origin_1xMIC.json --steps 5
```

Expected console checks include:

```text
1xMIC uniform: A=1 MIC=1 raw_growth=0 applied_growth=0
8xMIC uniform: A=8 MIC=1 raw_growth≈-1 applied_growth=0
square origin 1xMIC: A=1 MIC=1 raw_growth=0 applied_growth=0
```

With the default `stall` action, these runs continue for the requested number of steps, but biomass remains zero. Set `drug_nonpositive_growth_action` to `kill` if you want active tips to be removed and autostop to trigger instead.
