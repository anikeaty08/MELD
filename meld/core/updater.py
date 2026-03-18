"""Geometry-constrained updater for MELD."""

from __future__ import annotations

import math
import time

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..interfaces.base import ManifoldUpdater, TaskSnapshot, TrainArtifacts


def _gaussian_kl_diag(
    mean_before: Tensor,
    var_before: Tensor,
    mean_after: Tensor,
    var_after: Tensor,
) -> Tensor:
    safe_after = torch.clamp(var_after, min=1e-6)
    safe_before = torch.clamp(var_before, min=1e-6)
    term = torch.log(safe_after / safe_before) + (safe_before + (mean_before - mean_after).pow(2)) / safe_after - 1.0
    return 0.5 * term.sum()


def _distribution_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    student_log_probs = torch.log_softmax(student_logits, dim=1)
    teacher_probs = torch.softmax(teacher_logits, dim=1)
    return torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


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
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config.lr),
            momentum=float(getattr(config, "momentum", 0.9)),
            weight_decay=float(getattr(config, "weight_decay", 5e-4)),
        )
        criterion = nn.CrossEntropyLoss()
        start = time.time()
        lambda_schedule: list[float] = []
        geometry_loss_per_epoch: list[float] = []
        ewc_loss_per_epoch: list[float] = []
        ce_loss_per_epoch: list[float] = []
        loss_history: list[float] = []

        if snapshot is not None and snapshot.parameter_reference:
            reference = [torch.from_numpy(array).to(device=device) for array in snapshot.parameter_reference]
            fisher = torch.from_numpy(snapshot.fisher_diagonal).to(device=device)
        else:
            reference = []
            fisher = torch.empty(0, device=device)

        params = [param for param in model.parameters() if param.requires_grad]
        fisher_splits = []
        cursor = 0
        for param in params:
            numel = param.numel()
            fisher_splits.append(fisher[cursor : cursor + numel].view_as(param) if fisher.numel() else torch.zeros_like(param))
            cursor += numel

        for epoch in range(int(config.epochs)):
            model.train()
            decay_rate = self._decay_rate(snapshot, config)
            lambda_geometry = float(config.lambda_geometry) * math.exp(-decay_rate * epoch)
            lambda_schedule.append(lambda_geometry)
            ce_total = 0.0
            geometry_total = 0.0
            ewc_total = 0.0
            cached_geometry = torch.tensor(0.0, device=device)

            for batch_index, (inputs, targets) in enumerate(new_data_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                embeddings = model.embed(inputs)
                logits = model.classifier(embeddings)
                ce_loss = criterion(logits, targets)

                if snapshot is not None and snapshot.class_means:
                    if batch_index % self.geometry_refresh_interval == 0:
                        cached_geometry = self._geometry_loss(model, snapshot, device, embeddings.dtype)
                        geometry_loss = cached_geometry
                    else:
                        geometry_loss = cached_geometry.detach()
                    ewc_loss = self._ewc_loss(params, reference, fisher_splits) * float(config.lambda_ewc)
                else:
                    geometry_loss = torch.tensor(0.0, device=device)
                    ewc_loss = torch.tensor(0.0, device=device)

                loss = ce_loss + lambda_geometry * geometry_loss + ewc_loss
                loss.backward()
                optimizer.step()

                ce_total += float(ce_loss.item())
                geometry_total += float(geometry_loss.item())
                ewc_total += float(ewc_loss.item())
                loss_history.append(float(loss.item()))

            batches = max(1, len(new_data_loader))
            ce_loss_per_epoch.append(ce_total / batches)
            geometry_loss_per_epoch.append(geometry_total / batches)
            ewc_loss_per_epoch.append(ewc_total / batches)

        artifacts = TrainArtifacts(
            epochs_run=int(config.epochs),
            lambda_schedule=lambda_schedule,
            geometry_loss_per_epoch=geometry_loss_per_epoch,
            ewc_loss_per_epoch=ewc_loss_per_epoch,
            ce_loss_per_epoch=ce_loss_per_epoch,
            wall_time_seconds=time.time() - start,
            loss_history=loss_history,
        )
        return model, artifacts

    def _decay_rate(self, snapshot: TaskSnapshot | None, config: object) -> float:
        if snapshot is None:
            return float(getattr(config, "geometry_decay", 0.5))
        fisher_scale = max(snapshot.fisher_eigenvalue_max, 1e-6)
        return float(getattr(config, "geometry_decay", 0.5)) / (1.0 + fisher_scale * 100.0)

    def _geometry_loss(
        self,
        model: nn.Module,
        snapshot: TaskSnapshot,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        losses = []
        for class_id in snapshot.class_ids:
            anchors_np = snapshot.class_anchors.get(class_id)
            anchor_logits_np = snapshot.class_anchor_logits.get(class_id)
            if anchors_np is None or anchor_logits_np is None or len(anchors_np) == 0:
                continue
            anchors = torch.from_numpy(anchors_np).to(device=device, dtype=dtype)
            teacher_logits = torch.from_numpy(anchor_logits_np).to(device=device, dtype=dtype)
            current_logits = model.classifier(anchors)
            before_mean = torch.from_numpy(snapshot.class_means[class_id]).to(device=device, dtype=dtype)
            before_var = torch.from_numpy(snapshot.class_covs[class_id]).to(device=device, dtype=dtype)
            anchor_mean = anchors.mean(dim=0)
            anchor_var = anchors.var(dim=0, unbiased=False) + 1e-6
            stats_loss = _gaussian_kl_diag(before_mean, before_var, anchor_mean, anchor_var)
            logits_loss = _distribution_kl(current_logits, teacher_logits)
            losses.append(stats_loss + logits_loss)
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _ewc_loss(self, params: list[Tensor], reference: list[Tensor], fisher_splits: list[Tensor]) -> Tensor:
        if not reference:
            return torch.tensor(0.0, device=params[0].device if params else "cpu")
        penalty = torch.tensor(0.0, device=params[0].device)
        for param, ref, fisher in zip(params, reference, fisher_splits):
            penalty = penalty + (fisher * (param - ref).pow(2)).sum()
        return penalty
