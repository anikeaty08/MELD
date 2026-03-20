"""Geometry-constrained updater for MELD."""

from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from ..interfaces.base import ManifoldUpdater, TaskSnapshot, TrainArtifacts


def _gaussian_kl_diag(
    mean_before: Tensor,
    var_before: Tensor,
    mean_after: Tensor,
    var_after: Tensor,
) -> Tensor:
    safe_after  = torch.clamp(var_after,  min=1e-6)
    safe_before = torch.clamp(var_before, min=1e-6)
    term = (
        torch.log(safe_after / safe_before)
        + (safe_before + (mean_before - mean_after).pow(2)) / safe_after
        - 1.0
    )
    return 0.5 * term.sum()


def _kd_loss(student_logits: Tensor, teacher_logits: Tensor, temperature: float = 2.0) -> Tensor:
    """Soft knowledge distillation — Hinton et al. 2015."""
    T = temperature
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)


def _mixup(x: Tensor, y: Tensor, alpha: float = 0.2) -> tuple[Tensor, Tensor, Tensor, float]:
    """Mixup augmentation — Zhang et al. 2018."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _cutmix(x: Tensor, y: Tensor, alpha: float = 1.0) -> tuple[Tensor, Tensor, Tensor, float]:
    """CutMix augmentation — Yun et al. 2019.

    Returns: mixed_x, y_a, y_b, lam_adjusted
    """
    if alpha <= 0:
        lam = 1.0
        y_a, y_b = y, y
        return x, y_a, y_b, lam

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch = x.size(0)
    idx = torch.randperm(batch, device=x.device)
    y_a, y_b = y, y[idx]

    _, _, H, W = x.shape
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))

    cx = torch.randint(0, W, (1,), device=x.device).item()
    cy = torch.randint(0, H, (1,), device=x.device).item()

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    x_mixed = x.clone()
    x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on exact pixel ratio.
    lam_adjusted = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    return x_mixed, y_a, y_b, lam_adjusted


class GeometryConstrainedUpdater(ManifoldUpdater):
    def __init__(self, geometry_refresh_interval: int = 50) -> None:
        self.geometry_refresh_interval = geometry_refresh_interval

    def update(
        self,
        model: nn.Module,
        new_data_loader: DataLoader,
        snapshot: TaskSnapshot | None,
        config: object,
    ) -> tuple[nn.Module, TrainArtifacts]:
        device = next(model.parameters()).device
        epochs = int(config.epochs)
        lr = float(config.lr)
        lam_geo = float(getattr(config, "lambda_geometry", 0.5))
        lam_ewc = float(getattr(config, "lambda_ewc", 0.3))
        lam_kd = float(getattr(config, "lambda_kd", 1.0))
        kd_temp = float(getattr(config, "kd_temperature", 2.0))
        geo_dec = float(getattr(config, "geometry_decay", 0.3))

        # Optional training stabilizers (all configurable via attributes on `config`).
        # If not provided, enable early stopping with conservative defaults.
        # This is mainly to avoid wasting compute on later epochs once the
        # model stops improving.
        early_patience = int(getattr(config, "early_stopping_patience", 5))
        val_fraction = float(getattr(config, "early_stopping_val_fraction", 0.1))
        min_delta = float(getattr(config, "early_stopping_min_delta", 0.0))
        enable_grad_projection = bool(getattr(config, "enable_grad_projection", True))

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(getattr(config, "momentum", 0.9)),
            weight_decay=float(getattr(config, "weight_decay", 5e-4)),
            nesterov=True,
        )

        # Per-task LR schedule: cosine annealing with warm restarts.
        # We create the scheduler anew inside this per-task `update()` call.
        steps_per_epoch = max(1, len(new_data_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch,
            T_mult=1,
            eta_min=lr / 25.0,
        )

        # Label smoothing CE — reduces overconfidence.
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # EWC setup (diagonal Fisher for EWC fallback; K-FAC is used inside `_ewc_loss`).
        if snapshot is not None and snapshot.parameter_reference:
            reference = [torch.from_numpy(a).to(device) for a in snapshot.parameter_reference]
            fisher = torch.from_numpy(snapshot.fisher_diagonal).to(device)
        else:
            reference, fisher = [], torch.empty(0, device=device)

        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        param_names = [n for n, _ in named_params]
        params = [p for _, p in named_params]
        fisher_splits: list[Tensor] = []
        cursor = 0
        for p in params:
            n = p.numel()
            fisher_splits.append(
                fisher[cursor : cursor + n].view_as(p) if fisher.numel() else torch.zeros_like(p)
            )
            cursor += n

        # Optional early stopping: split the task train dataset into train/val subsets.
        train_loader = new_data_loader
        val_loader = None
        if early_patience > 0 and val_fraction > 0.0:
            ds = new_data_loader.dataset
            ds_len = len(ds)
            if ds_len > 2:
                val_size = max(1, int(ds_len * val_fraction))
                train_size = ds_len - val_size
                if train_size > 0:
                    train_idx = list(range(train_size))
                    val_idx = list(range(train_size, ds_len))
                    train_ds = Subset(ds, train_idx)
                    val_ds = Subset(ds, val_idx)
                    train_loader = DataLoader(
                        train_ds,
                        batch_size=new_data_loader.batch_size,
                        shuffle=True,
                        num_workers=new_data_loader.num_workers,
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=new_data_loader.batch_size,
                        shuffle=False,
                        num_workers=new_data_loader.num_workers,
                    )
                    steps_per_epoch = max(1, len(train_loader))
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=steps_per_epoch,
                        T_mult=1,
                        eta_min=lr / 25.0,
                    )

        start = time.time()
        lambda_schedule: list[float] = []
        geo_hist: list[float] = []
        ewc_hist: list[float] = []
        ce_hist: list[float] = []
        loss_hist: list[float] = []
        cached_geo = torch.tensor(0.0, device=device)
        train_acc_hist: list[float] = []
        projected_steps = 0
        total_steps = 0

        best_val = float("inf")
        bad_epochs = 0
        epochs_run = 0

        for epoch in range(epochs):
            model.train()
            decay = geo_dec / (1.0 + max(snapshot.fisher_eigenvalue_max if snapshot else 0, 1e-6) * 100.0)
            lam_g = lam_geo * math.exp(-decay * epoch)
            lambda_schedule.append(lam_g)

            ce_t = 0.0
            geo_t = 0.0
            ewc_t = 0.0
            correct = 0
            seen = 0

            for bidx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Mixup on task 0 (snapshot is None), CutMix afterwards.
                if snapshot is None:
                    inputs, ta, tb, lam_m = _mixup(inputs, targets, alpha=0.2)
                else:
                    inputs, ta, tb, lam_m = _cutmix(inputs, targets, alpha=1.0)

                embeddings = model.embed(inputs)
                logits = model.classifier(embeddings)

                ce = lam_m * criterion(logits, ta) + (1 - lam_m) * criterion(logits, tb)
                preds = logits.argmax(dim=1)
                correct += int((preds == targets).sum().item())
                seen += int(targets.numel())

                geo = torch.tensor(0.0, device=device)
                ewc = torch.tensor(0.0, device=device)

                if snapshot is not None and snapshot.class_means:
                    if bidx % self.geometry_refresh_interval == 0:
                        cached_geo = self._geometry_loss(model, snapshot, device, embeddings.dtype, lam_kd, kd_temp)
                    geo = cached_geo if bidx % self.geometry_refresh_interval == 0 else cached_geo.detach()
                    ewc = self._ewc_loss(params, param_names, reference, fisher_splits, snapshot) * lam_ewc

                reg_loss = lam_g * geo + ewc
                total_loss = ce + reg_loss

                optimizer.zero_grad(set_to_none=True)

                if enable_grad_projection and reg_loss.requires_grad:
                    ce_grads = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
                    reg_grads = torch.autograd.grad(reg_loss, params, retain_graph=False, allow_unused=True)

                    # PCGrad-style projection: if gradients conflict, project reg away from ce.
                    for p, gce, grec in zip(params, ce_grads, reg_grads):
                        if gce is None and grec is None:
                            continue
                        if gce is None:
                            p.grad = grec
                            continue
                        if grec is None:
                            p.grad = gce
                            continue
                        dot = (gce * grec).sum()
                        if dot.item() < 0:
                            projected_steps += 1
                            gce_norm2 = (gce * gce).sum() + 1e-12
                            grec_proj = grec - (dot / gce_norm2) * gce
                            p.grad = gce + grec_proj
                        else:
                            p.grad = gce + grec
                else:
                    total_loss.backward()
                total_steps += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                ce_t += float(ce.item())
                geo_t += float(geo.item())
                ewc_t += float(ewc.item())
                loss_hist.append(float(total_loss.item()))

            n = max(1, len(train_loader))
            ce_hist.append(ce_t / n)
            geo_hist.append(geo_t / n)
            ewc_hist.append(ewc_t / n)
            train_acc_hist.append(float(correct / max(1, seen)))
            epochs_run = epoch + 1

            # Early stopping check.
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        embeddings = model.embed(inputs)
                        logits = model.classifier(embeddings)
                        vloss = criterion(logits, targets)
                        val_loss_sum += float(vloss.item())
                        val_n += 1
                val_loss = val_loss_sum / max(1, val_n)

                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_patience:
                        break

        return model, TrainArtifacts(
            epochs_run=epochs_run if epochs_run > 0 else epochs,
            lambda_schedule=lambda_schedule[:epochs_run] if epochs_run > 0 else lambda_schedule,
            geometry_loss_per_epoch=geo_hist[:epochs_run] if epochs_run > 0 else geo_hist,
            ewc_loss_per_epoch=ewc_hist[:epochs_run] if epochs_run > 0 else ewc_hist,
            ce_loss_per_epoch=ce_hist[:epochs_run] if epochs_run > 0 else ce_hist,
            train_accuracy_per_epoch=train_acc_hist[:epochs_run] if epochs_run > 0 else train_acc_hist,
            projected_step_fraction=float(projected_steps / max(1, total_steps)),
            wall_time_seconds=time.time() - start,
            loss_history=loss_hist,
        )

    def _geometry_loss(
        self,
        model: nn.Module,
        snapshot: TaskSnapshot,
        device: torch.device,
        dtype: torch.dtype,
        lambda_kd: float,
        kd_temp: float,
    ) -> Tensor:
        # Zero-replay: operate purely on stored anchor embeddings/logits.
        # Gradients flow through the classifier head via KD; the Gaussian stats
        # term is constant (anchors come from stored snapshot).
        losses: list[Tensor] = []
        for cid in snapshot.class_ids:
            anch_np = snapshot.class_anchors.get(cid)
            logit_np = snapshot.class_anchor_logits.get(cid)
            if anch_np is None or logit_np is None or len(anch_np) == 0:
                continue

            anchors = torch.from_numpy(anch_np).to(device=device, dtype=dtype)
            teacher_logits = torch.from_numpy(logit_np).to(device=device, dtype=dtype)

            # Distill the classifier head towards stored anchor behavior.
            current_logits = model.classifier(anchors)
            kd = _kd_loss(current_logits, teacher_logits, kd_temp) * lambda_kd

            # Optional geometry proxy in embedding space (constant w.r.t. model).
            before_mean = torch.from_numpy(snapshot.class_means[cid]).to(device=device, dtype=dtype)
            before_var = torch.from_numpy(snapshot.class_covs[cid]).to(device=device, dtype=dtype)
            anchor_mean = anchors.mean(dim=0)
            anchor_var = anchors.var(dim=0, unbiased=False) + 1e-6
            stats = _gaussian_kl_diag(before_mean, before_var, anchor_mean, anchor_var)

            losses.append(stats + kd)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

    def _ewc_loss(
        self,
        params: list[Tensor],
        param_names: list[str],
        reference: list[Tensor],
        fisher_splits: list[Tensor],
        snapshot: TaskSnapshot,
    ) -> Tensor:
        if not reference:
            return torch.tensor(0.0, device=params[0].device if params else "cpu")
        pen = torch.tensor(0.0, device=params[0].device)

        kfac_A = snapshot.kfac_factors_A
        kfac_G = snapshot.kfac_factors_G
        kfac_param_names = set(snapshot.kfac_weight_param_names)

        for p, name, ref, f in zip(params, param_names, reference, fisher_splits):
            if name in kfac_param_names and name in kfac_A and name in kfac_G:
                # K-FAC quadratic form: Tr(G dW A dW^T)
                A = torch.from_numpy(kfac_A[name]).to(device=p.device)
                G = torch.from_numpy(kfac_G[name]).to(device=p.device)
                dW = p - ref
                pen = pen + (G @ dW @ A * dW).sum()
            else:
                pen = pen + (f * (p - ref).pow(2)).sum()
        return pen


class FullRetrainUpdater(ManifoldUpdater):
    """Plain full retrain baseline: no mixup and no label smoothing.

    This exists so the "full retrain" comparator is not unfairly helped by
    MELD's delta-training regularizers.
    """

    def update(
        self,
        model: nn.Module,
        new_data_loader: DataLoader,
        snapshot: TaskSnapshot | None,
        config: object,
    ) -> tuple[nn.Module, TrainArtifacts]:
        device = next(model.parameters()).device
        epochs = int(config.epochs)
        lr = float(config.lr)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(getattr(config, "momentum", 0.9)),
            weight_decay=float(getattr(config, "weight_decay", 5e-4)),
            nesterov=True,
        )

        total_steps = max(1, epochs * len(new_data_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

        # Plain CE: no label smoothing.
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        ce_hist: list[float] = []
        loss_hist: list[float] = []
        train_acc_hist: list[float] = []

        for _epoch in range(epochs):
            model.train()
            ce_t = 0.0
            correct = 0
            seen = 0

            for inputs, targets in new_data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                embeddings = model.embed(inputs)
                logits = model.classifier(embeddings)
                ce = criterion(logits, targets)
                preds = logits.argmax(dim=1)
                correct += int((preds == targets).sum().item())
                seen += int(targets.numel())
                ce.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                ce_t += float(ce.item())
                loss_hist.append(float(ce.item()))

            n = max(1, len(new_data_loader))
            ce_hist.append(ce_t / n)
            train_acc_hist.append(float(correct / max(1, seen)))

        return model, TrainArtifacts(
            epochs_run=epochs,
            lambda_schedule=[],
            geometry_loss_per_epoch=[],
            ewc_loss_per_epoch=[],
            ce_loss_per_epoch=ce_hist,
            train_accuracy_per_epoch=train_acc_hist,
            projected_step_fraction=None,
            wall_time_seconds=time.time() - start,
            loss_history=loss_hist,
        )