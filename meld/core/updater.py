"""Geometry-constrained updater for MELD."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from ..interfaces.base import ManifoldUpdater, TaskSnapshot, TrainArtifacts
from .weighter import KLIEPWeighter


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
    def __init__(
        self,
        geometry_refresh_interval: int = 50,
        manifold_samples_per_class: int = 8,
        weighter: KLIEPWeighter | None = None,
    ) -> None:
        self.geometry_refresh_interval = geometry_refresh_interval
        self.manifold_samples_per_class = manifold_samples_per_class
        self.weighter = weighter or KLIEPWeighter()

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
        cutmix_alpha = float(getattr(config, "cutmix_alpha", 0.0))
        enable_importance_weighting = bool(getattr(config, "enable_importance_weighting", True))

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
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(getattr(config, "label_smoothing", 0.0)),
            reduction="none",
        )

        # EWC setup (diagonal Fisher for EWC fallback; K-FAC is used inside `_ewc_loss`).
        if snapshot is not None and snapshot.parameter_reference:
            reference = [torch.from_numpy(a).to(device) for a in snapshot.parameter_reference]
            fisher = torch.from_numpy(snapshot.fisher_diagonal).to(device)
        else:
            reference, fisher = [], torch.empty(0, device=device)

        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        params = [p for _, p in named_params]
        protected_names = set(snapshot.protected_parameter_names) if snapshot is not None else set()
        ewc_named_params = [
            (name, param)
            for name, param in named_params
            if not protected_names or name in protected_names
        ]
        ewc_param_names = [name for name, _ in ewc_named_params]
        ewc_params = [param for _, param in ewc_named_params]
        fisher_splits: list[Tensor] = []
        cursor = 0
        for p in ewc_params:
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
                    generator = torch.Generator()
                    generator.manual_seed(int(getattr(config, "seed", 0)))
                    permutation = torch.randperm(ds_len, generator=generator).tolist()
                    train_idx = permutation[:train_size]
                    val_idx = permutation[train_size:]
                    train_ds = Subset(ds, train_idx)
                    val_ds = Subset(ds, val_idx)
                    train_loader = DataLoader(
                        train_ds,
                        batch_size=new_data_loader.batch_size,
                        shuffle=True,
                        num_workers=new_data_loader.num_workers,
                        pin_memory=bool(getattr(new_data_loader, "pin_memory", False)),
                        persistent_workers=bool(new_data_loader.num_workers > 0),
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=new_data_loader.batch_size,
                        shuffle=False,
                        num_workers=new_data_loader.num_workers,
                        pin_memory=bool(getattr(new_data_loader, "pin_memory", False)),
                        persistent_workers=bool(new_data_loader.num_workers > 0),
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
        use_importance_weighting = snapshot is not None and enable_importance_weighting
        if use_importance_weighting:
            self._fit_importance_weighter(model, new_data_loader, snapshot, device)

        for epoch in range(epochs):
            model.train()
            if snapshot is not None and bool(getattr(config, "freeze_bn_stats", True)):
                model.apply(_freeze_batch_norm_stats)
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

                # Task 0 should learn the base classes cleanly; replay-free
                # regularization starts from later tasks.
                if snapshot is None:
                    inputs, ta, tb, lam_m = _mixup(inputs, targets, alpha=0.0)
                else:
                    inputs, ta, tb, lam_m = _cutmix(inputs, targets, alpha=cutmix_alpha)

                embeddings = model.embed(inputs)
                logits = model.classifier(embeddings)

                ce_terms = lam_m * criterion(logits, ta) + (1 - lam_m) * criterion(logits, tb)
                if use_importance_weighting:
                    sample_weights = self.weighter.score(embeddings).detach()
                    ce = (sample_weights * ce_terms).mean()
                else:
                    ce = ce_terms.mean()
                preds = logits.argmax(dim=1)
                correct += int((preds == targets).sum().item())
                seen += int(targets.numel())

                geo = torch.tensor(0.0, device=device)
                ewc = torch.tensor(0.0, device=device)

                if snapshot is not None and snapshot.class_means:
                    if bidx % self.geometry_refresh_interval == 0:
                        cached_geo = self._geometry_loss(model, snapshot, device, embeddings.dtype, lam_kd, kd_temp)
                    geo = cached_geo if bidx % self.geometry_refresh_interval == 0 else cached_geo.detach()
                    ewc = self._ewc_loss(ewc_params, ewc_param_names, reference, fisher_splits, snapshot) * lam_ewc

                reg_loss = lam_g * geo + ewc
                total_loss = ce + reg_loss

                optimizer.zero_grad(set_to_none=True)

                if enable_grad_projection and reg_loss.requires_grad:
                    ce_grads = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
                    reg_grads = torch.autograd.grad(reg_loss, params, retain_graph=False, allow_unused=True)
                    projected_this_step = False

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
                            projected_this_step = True
                            gce_norm2 = (gce * gce).sum() + 1e-12
                            grec_proj = grec - (dot / gce_norm2) * gce
                            p.grad = gce + grec_proj
                        else:
                            p.grad = gce + grec
                    if projected_this_step:
                        projected_steps += 1
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
                        vloss = criterion(logits, targets).mean()
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

    def _fit_importance_weighter(
        self,
        model: nn.Module,
        new_data_loader: DataLoader,
        snapshot: TaskSnapshot,
        device: torch.device,
    ) -> None:
        model.eval()
        embeddings_all: list[Tensor] = []
        targets_all: list[Tensor] = []
        with torch.no_grad():
            for inputs, targets in new_data_loader:
                inputs = inputs.to(device)
                embeddings_all.append(model.embed(inputs))
                targets_all.append(targets.to(device))
        if not embeddings_all:
            snapshot.importance_weights = {}
            return
        embeddings = torch.cat(embeddings_all, dim=0)
        targets = torch.cat(targets_all, dim=0)
        snapshot.importance_weights = self.weighter.fit(embeddings, snapshot, targets)

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
        if not snapshot.class_ids:
            return torch.tensor(0.0, device=device)

        old_class_ids = torch.tensor(snapshot.class_ids, device=device, dtype=torch.long)
        teacher_weight = torch.stack(
            [
                torch.from_numpy(snapshot.classifier_weights[cid]).to(device=device, dtype=dtype)
                for cid in snapshot.class_ids
                if cid in snapshot.classifier_weights
            ],
            dim=0,
        ) if snapshot.classifier_weights else torch.empty(0, device=device, dtype=dtype)
        teacher_bias = torch.tensor(
            [snapshot.classifier_biases[cid] for cid in snapshot.class_ids if cid in snapshot.classifier_biases],
            device=device,
            dtype=dtype,
        ) if snapshot.classifier_biases else torch.empty(0, device=device, dtype=dtype)

        for cid in snapshot.class_ids:
            anch_np = snapshot.class_anchors.get(cid)
            logit_np = snapshot.class_anchor_logits.get(cid)
            if anch_np is None or logit_np is None or len(anch_np) == 0 or teacher_weight.numel() == 0:
                continue

            anchors = torch.from_numpy(anch_np).to(device=device, dtype=dtype)
            teacher_logits = torch.from_numpy(logit_np).to(device=device, dtype=dtype)
            current_logits = model.classifier(anchors).index_select(1, old_class_ids)
            anchor_kd = _kd_loss(current_logits, teacher_logits, kd_temp)

            mean = torch.from_numpy(snapshot.class_means[cid]).to(device=device, dtype=dtype)
            std = torch.sqrt(
                torch.from_numpy(snapshot.class_covs[cid]).to(device=device, dtype=dtype).clamp_min(1e-6)
            )
            sample_count = max(1, min(self.manifold_samples_per_class, anchors.size(0)))
            gaussian_samples = mean.unsqueeze(0) + torch.randn(
                sample_count,
                mean.numel(),
                device=device,
                dtype=dtype,
            ) * std.unsqueeze(0)
            teacher_gaussian_logits = F.linear(gaussian_samples, teacher_weight, teacher_bias)
            current_gaussian_logits = model.classifier(gaussian_samples).index_select(1, old_class_ids)
            gaussian_kd = _kd_loss(current_gaussian_logits, teacher_gaussian_logits, kd_temp)

            losses.append((anchor_kd + gaussian_kd) * lambda_kd)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

    def _ewc_loss(
        self,
        params: list[Tensor],
        param_names: list[str],
        reference: list[Tensor],
        fisher_splits: list[Tensor],
        snapshot: TaskSnapshot,
    ) -> Tensor:
        if not reference or not params:
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


def _freeze_batch_norm_stats(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


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

        criterion = nn.CrossEntropyLoss(label_smoothing=float(getattr(config, "label_smoothing", 0.0)))

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


class FrozenBackboneAnalyticUpdater(ManifoldUpdater):
    """Replay-free analytic updater over frozen features.

    Task 0 should still be trained with the standard SGD updater. For
    incremental tasks, this updater keeps the backbone fixed and fits the
    newest classifier head with ridge regression on current-task features.
    """

    def update(
        self,
        model: nn.Module,
        new_data_loader: DataLoader,
        snapshot: TaskSnapshot | None,
        config: object,
    ) -> tuple[nn.Module, TrainArtifacts]:
        if snapshot is None:
            return FullRetrainUpdater().update(model, new_data_loader, snapshot, config)

        device = next(model.parameters()).device
        start = time.time()

        for param in model.backbone.parameters():
            param.requires_grad_(False)

        model.eval()
        features, local_targets = self._collect_task_features(model, new_data_loader, device)
        if features.numel() == 0 or local_targets.numel() == 0:
            return model, TrainArtifacts(
                epochs_run=0,
                lambda_schedule=[],
                geometry_loss_per_epoch=[],
                ewc_loss_per_epoch=[],
                ce_loss_per_epoch=[],
                train_accuracy_per_epoch=[],
                projected_step_fraction=None,
                wall_time_seconds=time.time() - start,
                loss_history=[],
                skipped=True,
            )

        head = model.classifier.heads[-1]
        weight, bias = self._solve_ridge_head(
            features,
            local_targets,
            num_classes=head.out_features,
            ridge=float(getattr(config, "analytic_ridge", 1e-3)),
        )

        with torch.no_grad():
            head.weight.copy_(weight)
            head.bias.copy_(bias)

        logits = head(features)
        ce = F.cross_entropy(logits, local_targets)
        train_accuracy = float((logits.argmax(dim=1) == local_targets).float().mean().item())

        for param in model.backbone.parameters():
            param.requires_grad_(True)

        return model, TrainArtifacts(
            epochs_run=1,
            lambda_schedule=[],
            geometry_loss_per_epoch=[0.0],
            ewc_loss_per_epoch=[0.0],
            ce_loss_per_epoch=[float(ce.item())],
            train_accuracy_per_epoch=[train_accuracy],
            projected_step_fraction=None,
            wall_time_seconds=time.time() - start,
            loss_history=[float(ce.item())],
        )

    def _collect_task_features(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        features: list[Tensor] = []
        local_targets: list[Tensor] = []
        class_to_local: dict[int, int] = {}

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                embeddings = model.embed(inputs)
                features.append(embeddings)
                for class_id in targets.detach().cpu().unique().tolist():
                    if int(class_id) not in class_to_local:
                        _, offset = model.classifier.class_to_head[int(class_id)]
                        class_to_local[int(class_id)] = int(offset)
                mapped = torch.tensor(
                    [class_to_local[int(class_id)] for class_id in targets.detach().cpu().tolist()],
                    device=device,
                    dtype=torch.long,
                )
                local_targets.append(mapped)

        if not features:
            return torch.empty(0, model.out_dim, device=device), torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(features, dim=0), torch.cat(local_targets, dim=0)

    @staticmethod
    def _solve_ridge_head(
        features: Tensor,
        targets: Tensor,
        num_classes: int,
        ridge: float,
    ) -> tuple[Tensor, Tensor]:
        dtype = features.dtype
        device = features.device
        ones = torch.ones(features.size(0), 1, device=device, dtype=dtype)
        design = torch.cat((features, ones), dim=1)
        target_matrix = F.one_hot(targets, num_classes=num_classes).to(dtype=dtype)
        gram = design.t() @ design
        gram = gram + ridge * torch.eye(gram.size(0), device=device, dtype=dtype)
        rhs = design.t() @ target_matrix
        solution = torch.linalg.solve(gram, rhs)
        weight = solution[:-1].t().contiguous()
        bias = solution[-1].contiguous()
        return weight, bias
