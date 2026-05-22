# MIC mutation / breakout-event update

This update adds heritable MIC-like tolerance mutations to the antifungal field model.

## Biological idea

When a new hyphal lineage is born, normally through branching, it inherits its parent's MIC.  If MIC mutation is enabled, that new daughter section has a configurable probability of changing MIC.  The probability has a baseline component and a density-dependent boost so crowded internal colony regions can become mutation-rich reservoirs.

The MIC effect size is multiplicative and Laplace distributed on the log scale:

```text
delta ~ Laplace(0, drug_mic_mutation_scale)
new_MIC = parent_MIC * exp(delta)
```

This gives many small MIC changes and exponentially fewer large jumps.  Positive deltas increase resistance; negative deltas increase susceptibility.

## New Options fields

```python
drug_mic_mutations_enabled: bool = False
drug_mic_mutation_base_probability: float = 0.0
drug_mic_mutation_density_boost: float = 0.0
drug_mic_mutation_density_threshold: Optional[float] = None
drug_mic_mutation_density_saturation: Optional[float] = None
drug_mic_mutation_max_probability: float = 1.0
drug_mic_mutation_scale: float = 0.25
drug_mic_min: float = 1e-6
drug_mic_max: float = 1e6
```

Defaults are backward-compatible: MIC mutation is off unless `drug_mic_mutations_enabled` is set to true.

## Density-dependent probability

The effective mutation probability is:

```text
p = base_probability + density_boost * density_signal
```

where `density_signal` ramps from 0 to 1 between `drug_mic_mutation_density_threshold` and `drug_mic_mutation_density_saturation`.

If `drug_mic_mutation_density_threshold` is `None`, the boost starts from density 0.  If `drug_mic_mutation_density_saturation` is `None`, the boost saturates at the existing `density_threshold`.

This avoids the common trap where density only boosts mutation after the normal branching-density threshold has already blocked branch formation.

## New CSV columns

Both `mycelium_final.csv` and `mycelium_time_series.csv` now include:

```text
drug_mic_parent
drug_mic_mutated_from_parent
drug_mic_mutated_from_wildtype
drug_mic_mutation_delta_log
drug_mic_mutation_probability
drug_mic_mutation_density
```

These allow later R analysis of which sections are true MIC mutants, whether they became more resistant or more susceptible, and whether mutation probability was elevated in dense regions.

## Example config

A test config was added at:

```text
config/test_configs/param_config_drug_square_mic_mutations_breakout_test.json
```

The probabilities in this config are intentionally higher than realistic biological rates so that short simulations are likely to produce MIC diversity and candidate breakout lineages.
