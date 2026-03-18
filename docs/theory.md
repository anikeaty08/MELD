# MELD Theory

MELD treats class-incremental learning as a safety verification problem rather than a purely optimization problem.

## Pre-training safety bound

The default oracle computes:

`epsilon_max = lambda_max(F) * eta * sqrt(T * d)`

- `lambda_max(F)` is approximated with the maximum entry of the diagonal Fisher estimate.
- `eta` is the learning rate.
- `T` is the number of planned optimization steps.
- `d` is the embedding dimension.

If the predicted drift exceeds the configured tolerance, MELD skips the delta update and recommends a full retrain.

## Post-training verification

The realized drift is computed as the mean normalized class-mean displacement:

`epsilon_actual = mean_c ||mu_after - mu_before||_2 / ||mu_before||_2`

## Geometry-preserving update

The default updater minimizes:

`L = CE(new data) + lambda_1(t) * KL(old manifold || current manifold) + lambda_2 * EWC`

This preserves old-class geometry without replay data while still letting the backbone adapt to new distributions.
