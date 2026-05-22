# Fungistatic vs. fungicidal drug behaviour

This update adds a high-level switch for drug outcome behaviour:

- `drug_effect_type = "fungistatic"` keeps inhibited tips alive but stalled in place.
- `drug_effect_type = "fungicidal"` kills inhibited tips, so they lose their live tip marker and can be distinguished from living/stalled tips.
- `drug_effect_type = "legacy"` preserves the older `drug_nonpositive_growth_action` behaviour.

For fungicidal runs, `drug_fungicidal_condition` controls what counts as lethal:

- `"raw_growth_nonpositive"` kills when the drug-response model returns `raw_growth_rate <= 0`.
- `"concentration_at_or_above_mic"` kills as soon as the local concentration is at or above that tip's MIC.

Additional CSV fields exported:

- `drug_killed_by_drug`
- `drug_death_time`
