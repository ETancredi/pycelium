# MIC mutation no-density-boost update

This update removes the explicit density-dependent MIC mutation probability model.

## Rationale

The model now treats internal colony density as an indirect driver of resistance emergence:

- dense internal regions keep generating more new hyphal sections and branches;
- each new daughter section has the same baseline MIC mutation probability;
- therefore dense/growing regions naturally create more mutation opportunities without adding a separate density threshold, saturation point, or boost term.

## Removed settings

The following obsolete config keys are no longer part of `Options`:

- `drug_mic_mutation_density_boost`
- `drug_mic_mutation_density_threshold`
- `drug_mic_mutation_density_saturation`

Older JSON configs containing these keys are still loadable. `config/sim_config.py` discards those three keys before constructing `Options`.

## Retained MIC mutation model

For every new daughter section or delayed secondary germ tube:

```text
p_mutation = min(drug_mic_mutation_base_probability, drug_mic_mutation_max_probability)

if mutation occurs:
    delta ~ Laplace(0, drug_mic_mutation_scale)
    child_MIC = parent_MIC * exp(delta)
```

The mutation is still heritable, multiplicative, and symmetric on the log scale, so small MIC changes remain common and large susceptible/resistant shifts remain rare.

## CSV changes

The previous diagnostic column `drug_mic_mutation_density` has been removed from:

- `mycelium_final.csv`
- `mycelium_time_series.csv`

The remaining mutation columns are:

- `drug_mic`
- `drug_mic_parent`
- `drug_mic_mutated_from_parent`
- `drug_mic_mutated_from_wildtype`
- `drug_mic_mutation_delta_log`
- `drug_mic_mutation_probability`
