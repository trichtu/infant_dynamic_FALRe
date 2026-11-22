# Infant functional connectivity analysis


## Trajectory estimation

`trajectory_estimation/` fits developmental trajectories for connectivity and behavior using generalized additive mixed models and summarizes connectivity change rates by network distance.

```powershell
python trajectory_estimation/estimate_connectome_trajectory.py
python trajectory_estimation/estimate_behavior_trajectories.py
```

## Dynamic decomposition

`dynamic_decomposition/` decomposes sliding-window functional connectivity into recurring states, derives an axis from each selected state, and plots the relationship between state expression and axis amplitude.

```powershell
python dynamic_decomposition/dFNC_decomposition.py
python dynamic_decomposition/axe_state_relationship.py
```

To regenerate `eigenvector.mat`, run from `dynamic_decomposition/`:

```powershell
python -c "from axis_pattern import axe_pattern; axe_pattern()"
```

## FLARe scores

`FLARe_score/` estimates state and axis time courses with constrained ICA, then calculates state fluctuation asymmetry (FLARe) and axis amplitudes.

```powershell
python FLARe_score/FLARe_calculation.py
python FLARe_score/axes_amplitude_calculation.py
```


## Prediction of long-term change rates

`prediction_on_long_term_CR/` uses FLARe scores to predict future and previous change rates in connectivity, targets reached, and eye looking with infant-level cross-validation.

Run from `prediction_on_long_term_CR/` with an existing prediction cache. Use a different `--repeat-id` for each repeat and run with `--aggregate` after two or more repeats.

```powershell
python connectivity_FLARe_prediction.py --output connectivity_results --repeat-id 0
python behavior_FLARe_prediction.py --output behavior_results --repeat-id 0
```

