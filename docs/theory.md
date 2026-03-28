# Delta Framework - Theoretical Foundations

## The Problem

Given a model `theta_old` trained on `D_old`, and new data `D_new` arriving,
we want a new model `theta_new` that is close to the ideal full retraining solution:

```text
theta* = argmin_theta L(theta; D_old U D_new)
```

but during the update we only want to use `D_new`.

## Core Mathematical Insight

The full-dataset gradient can be decomposed as:

```text
grad L(theta; D_full) =
    [n_old * grad L(theta; D_old) + n_new * grad L(theta; D_new)] / n_total
```

We cannot compute `grad L(theta; D_old)` directly because `D_old` is unavailable.

So we approximate the old-task gradient around `theta_old` using a second-order expansion:

```text
grad L(theta; D_old) ~= H_old * (theta - theta_old)
```

where `H_old` is the Hessian of the old-task loss near `theta_old`.

Substituting this into the combined objective gives the delta-style update:

```text
theta_new = argmin_theta [
    ((n_old + n_new) / n_new) * L(theta; D_new)
    + (n_old / (n_old + n_new)) * (theta - theta_old)^T H_old (theta - theta_old)
]
```

The first term is the new-data training objective.
The second term is the old-task preservation term.

## Bias Correction

The factor:

```text
(n_old + n_new) / n_new
```

is meant to correct recency bias, because training on only `D_new`
would otherwise overweight the new data relative to the full-data objective.

In the current practical implementation, we use a more stable default:

```text
ce_scale = 1.0
```

because the derived scaling can interact badly with neural training on harder tasks.

## Hessian Approximation via KFAC

The old-task Hessian is approximated using KFAC:

```text
H_l ~= G_l kron A_l
```

where:

```text
A_l = E[a_l a_l^T]
G_l = E[g_l g_l^T]
```

This gives a compact second-order approximation that is much cheaper than a full Hessian.

## Fisher-Regularized Drift Penalty

Using Fisher/KFAC, the preservation term becomes:

```text
Omega(theta, theta_ref) =
    (theta - theta_ref)^T F (theta - theta_ref)
```

In the implementation, we normalize by Fisher trace for scale stability:

```text
Omega_norm = Omega / (trace(F) + eps)
```

## Theorem 1 - Convex Approximation

For locally smooth and strongly convex layers:

```text
||theta_delta - theta*||_2 <= (L / mu) * epsilon_hessian
```

where:

- `L` is a smoothness constant
- `mu` is a local strong-convexity constant
- `epsilon_hessian` is the approximation error of the KFAC/Hessian surrogate

This is reported in the framework as `epsilon_param`.

## Theorem 2 - Prediction-Side Diagnostic

The framework also uses a prediction-side bound:

```text
KL_bound = L_pred_hat * ||theta_delta - theta_ref||
```

and a normalized diagnostic:

```text
KL_bound_normalized = KL_bound / sqrt(n_params)
```

This is used as a practical indicator of how far the updated model moved
from the old reference model in prediction space.

## Distribution Shift Handling

The framework distinguishes 3 cases:

| Shift Type | Detection | Update Behavior |
|---|---|---|
| `none` | low embedding shift, stable labels | standard delta update |
| `covariate` | embedding distribution changes | continue delta update, report shift |
| `concept` | shared-class label shift detected | reduce trust in equivalence, warn that full retraining may be needed |

Shift detection is implemented using:

- embedding distribution comparison
- label-distribution testing on shared classes

## Calibration Preservation

Calibration is tracked with Expected Calibration Error:

```text
ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
```

The framework reports:

- `ece_before`
- `ece_after`
- `ece_delta`

This matters because a model can remain accurate but become poorly calibrated.

## Compute Savings

The framework compares the delta update cost with estimated full retraining cost.

If:

```text
full retrain cost ~= O(n_total)
delta update cost ~= O(n_new)
```

then the delta update can be much cheaper when `n_new << n_old`.

The actual empirical quantity reported by the framework is:

```text
compute_ratio = full_retrain_time / delta_update_time
```

## What The Current Code Uses

The present implementation uses the following practical design:

- `ce_scale = 1.0`
- bounded `ewc_scale`
- EWC warmup
- active-class masked CE
- seen-class-only KD
- Fisher-trace normalization
- task-time head expansion
- shift detection
- equivalence-style diagnostics

These choices were made because they are more stable in real continual-learning runs
than the most aggressive theoretical scaling rules.

## References

1. Wu et al. "DeltaGrad: Rapid retraining of machine learning models", ICML 2020.
2. Martens and Grosse. "Optimizing Neural Networks with Kronecker-factored Approximate Curvature", ICML 2015.
3. Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks", PNAS 2017.
4. van de Ven. "On the Computation of the Fisher Information in Continual Learning", 2025.
5. Sugiyama et al. "Covariate Shift Adaptation by Importance Weighted Cross Validation", JMLR 2007.

# Update 1 - 2026-03-29 02:08:57
# Update 8 - 2026-03-28 23:48:10
# Update 24 - 2026-03-28 18:36:38
# Update 18 @ 2026-03-29 07:23:11
# Update 20 @ 2026-03-29 01:54:10
# Update 27 @ 2026-03-29 02:36:32
# Update 28 @ 2026-03-28 16:36:40
# Update 32 @ 2026-03-28 13:54:29
# Update 35 @ 2026-03-29 05:30:20
# Update 10 @ 2026-03-29 03:23:00
# Update 19 @ 2026-03-28 21:02:44
# Update 12 @ 2026-03-28 18:28:15
# Update 31 @ 2026-03-28 11:16:26
# Update 34 @ 2026-03-29 02:17:27