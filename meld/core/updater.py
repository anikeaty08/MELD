"""Geometry-constrained updater for MELD."""

from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

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
        device  = next(model.parameters()).device
        epochs  = int(config.epochs)
        lr      = float(config.lr)
        lam_geo = float(getattr(config, "lambda_geometry",  0.5))
        lam_ewc = float(getattr(config, "lambda_ewc",       0.3))
        lam_kd  = float(getattr(config, "lambda_kd",        1.0))
        kd_temp = float(getattr(config, "kd_temperature",   2.0))
        geo_dec = float(getattr(config, "geometry_decay",   0.3))

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(getattr(config, "momentum",     0.9)),
            weight_decay=float(getattr(config, "weight_decay", 5e-4)),
            nesterov=True,
        )

        # OneCycleLR — proven +5-8% over flat LR on CIFAR
        total_steps = epochs * len(new_data_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

        # Label smoothing CE — reduces overconfidence, +1-2% accuracy
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # EWC setup
        if snapshot is not None and snapshot.parameter_reference:
            reference = [torch.from_numpy(a).to(device) for a in snapshot.parameter_reference]
            fisher    = torch.from_numpy(snapshot.fisher_diagonal).to(device)
        else:
            reference, fisher = [], torch.empty(0, device=device)

        params = [p for p in model.parameters() if p.requires_grad]
        fisher_splits: list[Tensor] = []
        cursor = 0
        for p in params:
            n = p.numel()
            fisher_splits.append(
                fisher[cursor: cursor + n].view_as(p) if fisher.numel()
                else torch.zeros_like(p)
            )
            cursor += n

        start = time.time()
        lambda_schedule, geo_hist, ewc_hist, ce_hist, loss_hist = [], [], [], [], []
        cached_geo = torch.tensor(0.0, device=device)

        for epoch in range(epochs):
            model.train()
            decay = geo_dec / (1.0 + max(snapshot.fisher_eigenvalue_max if snapshot else 0, 1e-6) * 100.0)
            lam_g = lam_geo * math.exp(-decay * epoch)
            lambda_schedule.append(lam_g)
            ce_t = geo_t = ewc_t = 0.0

            for bidx, (inputs, targets) in enumerate(new_data_loader):
                inputs  = inputs.to(device)
                targets = targets.to(device)

                # Mixup augmentation
                inputs, ta, tb, lam_m = _mixup(inputs, targets, alpha=0.2)

                optimizer.zero_grad(set_to_none=True)
                embeddings = model.embed(inputs)
                logits     = model.classifier(embeddings)

                # CE loss with mixup
                ce = lam_m * criterion(logits, ta) + (1 - lam_m) * criterion(logits, tb)

                geo = torch.tensor(0.0, device=device)
                ewc = torch.tensor(0.0, device=device)

                if snapshot is not None and snapshot.class_means:
                    if bidx % self.geometry_refresh_interval == 0:
                        cached_geo = self._geometry_loss(
                            model, snapshot, device, embeddings.dtype, lam_kd, kd_temp
                        )
                    geo = cached_geo if bidx % self.geometry_refresh_interval == 0 \
                        else cached_geo.detach()
                    ewc = self._ewc_loss(params, reference, fisher_splits) * lam_ewc

                loss = ce + lam_g * geo + ewc
                loss.backward()

                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                ce_t  += float(ce.item())
                geo_t += float(geo.item())
                ewc_t += float(ewc.item())
                loss_hist.append(float(loss.item()))

            n = max(1, len(new_data_loader))
            ce_hist.append(ce_t / n)
            geo_hist.append(geo_t / n)
            ewc_hist.append(ewc_t / n)

        return model, TrainArtifacts(
            epochs_run=epochs,
            lambda_schedule=lambda_schedule,
            geometry_loss_per_epoch=geo_hist,
            ewc_loss_per_epoch=ewc_hist,
            ce_loss_per_epoch=ce_hist,
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
        losses: list[Tensor] = []
        model.eval()
        with torch.no_grad():
            for cid in snapshot.class_ids:
                anch_np = snapshot.class_anchors.get(cid)
                logit_np = snapshot.class_anchor_logits.get(cid)
                if anch_np is None or logit_np is None or len(anch_np) == 0:
                    continue
                anchors         = torch.from_numpy(anch_np).to(device=device, dtype=dtype)
                teacher_logits  = torch.from_numpy(logit_np).to(device=device, dtype=dtype)
                current_logits  = model.classifier(anchors)
                before_mean     = torch.from_numpy(snapshot.class_means[cid]).to(device=device, dtype=dtype)
                before_var      = torch.from_numpy(snapshot.class_covs[cid]).to(device=device,  dtype=dtype)
                anchor_mean     = anchors.mean(dim=0)
                anchor_var      = anchors.var(dim=0, unbiased=False) + 1e-6
                stats = _gaussian_kl_diag(before_mean, before_var, anchor_mean, anchor_var)
                kd    = _kd_loss(current_logits, teacher_logits, kd_temp) * lambda_kd
                losses.append(stats + kd)
        model.train()
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

    def _ewc_loss(self, params, reference, fisher_splits) -> Tensor:
        if not reference:
            return torch.tensor(0.0, device=params[0].device if params else "cpu")
        pen = torch.tensor(0.0, device=params[0].device)
        for p, ref, f in zip(params, reference, fisher_splits):
            pen = pen + (f * (p - ref).pow(2)).sum()
        return pen