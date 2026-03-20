# MELD Theory

MELD frames continual learning as a *bounded-drift control problem*. For each
incoming task, the system predicts an upper bound on allowable representation
drift before training (`epsilon_max`), executes a constrained update, and then
measures realized drift (`epsilon_actual`) from replay-free statistics.

## 1) Replay-free snapshot state

For each past class `c`, MELD stores:

- Embedding Gaussian statistics: mean `mu_c`, diagonal variance `sigma_c^2`
- A small embedding-anchor set and corresponding classifier logits
- Parameter reference `theta_old`
- Curvature approximation: diagonal Fisher for all params, plus K-FAC factors
  `(A_l, G_l)` for selected final linear layers

No raw image tensors are retained or replayed.

## 2) Pre-training safety bound

Let `F` be the curvature proxy and `eta` the planned peak learning rate for the
task update. The safety oracle computes:

`epsilon_max = lambda_max(F) * eta * sqrt(T * d)`

where:

- `lambda_max(F)` is estimated from the max of diagonal Fisher and K-FAC proxy
  eigenvalues gathered in the snapshot
- `T` is effective optimization steps for the task
- `d` is embedding dimensionality

If `epsilon_max > bound_tolerance`, MELD enters `BOUND_EXCEEDED`, skips the
delta update, and recommends full retraining.

## 3) Delta update objective

For non-skipped tasks, MELD optimizes:

`L_total = L_ce + lambda_geo(t) * L_geo + lambda_ewc * L_ewc`

with:

- `L_ce`: task classification loss (mixup on task 0; CutMix after task 0)
- `L_geo`: embedding-anchor geometry consistency + KD on stored anchor logits
- `L_ewc`: quadratic trust-region penalty around `theta_old` using diagonal
  Fisher and K-FAC on selected layers

`lambda_geo(t)` decays over epochs based on configured geometry decay and
snapshot curvature scale.

## 4) Post-training verification

After update, MELD recomputes snapshot statistics and measures:

`epsilon_actual = mean_c ||mu_c_after - mu_c_before||_2 / max(||mu_c_before||_2, eps)`

The deploy policy then uses `(epsilon_max, epsilon_actual, drift_score)` to
choose `SAFE_DELTA`, `CAUTIOUS_DELTA`, `BOUND_VIOLATED`, `SHIFT_CRITICAL`, or
`BOUND_EXCEEDED`.

## 5) Operational interpretation

- `epsilon_max` is a conservative *risk estimate before training*
- `epsilon_actual` is a realized *drift audit after training*
- The gap (`epsilon_max - epsilon_actual`) indicates bound tightness
- A skip decision is safety-preserving: deployment can continue on the prior
  model while triggering full retrain on the background path
