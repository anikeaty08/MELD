"""
DeltaTrainer — The main framework class.

Wraps any nn.Module and provides:
  fit()        — standard first training
  fit_delta()  — delta update on new data only
  save_state() — persist state (no raw data)
  load_state() — restore state
  certify()    — get equivalence certificate
"""

from __future__ import annotations
import time
import warnings
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from .core.state import DeltaState
from .core.fisher import KFACComputer
from .core.shift import ShiftDetector
from .core.certificate import EquivalenceCertificate, CertificateComputer
from .core.calibration import CalibrationTracker


def _to_device(inputs, device):
    if isinstance(inputs, dict):
        return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
    return inputs.to(device)


class DeltaTrainer:

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.model = model.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state: DeltaState | None = None
        self._last_full_retrain_time: float = 0.0
        self._fisher = KFACComputer()
        self._shift = ShiftDetector()
        self._calibration = CalibrationTracker()
        self._certificate = CertificateComputer()

    def fit(self, dataloader: DataLoader, epochs: int = 10, lr: float = 0.01, weight_decay: float = 5e-4) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        t_start = time.time()
        self.model.train()
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                inputs = _to_device(inputs, self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        self._last_full_retrain_time = time.time() - t_start
        self.state = self._fisher.compute(self.model, dataloader, self.device)
        self.state.n_old = len(dataloader.dataset)
        self._shift.update_state(dataloader, self.model, self.state, self.device)

    def fit_delta(self, new_dataloader: DataLoader, epochs: int = 10, lr: float = 0.01,
                  weight_decay: float = 5e-4, val_loader: DataLoader | None = None) -> EquivalenceCertificate:
        if self.state is None:
            raise RuntimeError("Call fit() before fit_delta().")

        n_new = len(new_dataloader.dataset)
        n_old = self.state.n_old
        n_total = n_old + n_new
        ce_scale = 1.0
        ewc_scale = min(float(n_old) / float(max(n_new, 1)), 1.0) if n_old > 0 else 0.0

        shift_type = self._shift.detect(new_dataloader, self.state, self.model, self.device)
        if shift_type == "concept":
            warnings.warn(
                "Concept drift detected. Setting ewc_scale=0.0 and ce_scale=1.0. "
                "Consider full retraining for best results.",
                UserWarning, stacklevel=2,
            )
            ce_scale = 1.0
            ewc_scale = 0.0

        ref_params = {n: torch.from_numpy(v).to(self.device) for n, v in self.state.theta_ref.items()}
        kfac_A = {n: torch.from_numpy(v).to(self.device) for n, v in self.state.kfac_A.items()}
        kfac_G = {n: torch.from_numpy(v).to(self.device) for n, v in self.state.kfac_G.items()}
        kfac_names = set(self.state.kfac_param_names)

        fisher_diag_splits: dict[str, Tensor] = {}
        fisher_trace = 1.0
        if self.state.fisher_diag is not None:
            fd = torch.from_numpy(self.state.fisher_diag).to(self.device)
            fisher_trace = max(float(fd.sum().item()), 1e-8)
            cursor = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    n = param.numel()
                    if cursor + n <= fd.numel():
                        fisher_diag_splits[name] = fd[cursor:cursor + n].view_as(param)
                    cursor += n

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(reduction="none")

        ece_before = self._calibration.compute_ece(
            self.model, val_loader if val_loader else new_dataloader, self.device)

        t_start = time.time()
        self.model.train()
        for epoch in range(epochs):
            for inputs, targets in new_dataloader:
                inputs = _to_device(inputs, self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                per_sample = criterion(logits, targets)
                ce = ce_scale * per_sample.mean()

                kfac_pen = torch.tensor(0.0, device=self.device)
                if ewc_scale > 0.0:
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad or name not in ref_params:
                            continue
                        dW = param - ref_params[name]
                        layer = name.replace(".weight", "")
                        if name in kfac_names and layer in kfac_A and layer in kfac_G:
                            A = kfac_A[layer]
                            G = kfac_G[layer]
                            if dW.dim() == 2:
                                kfac_pen = kfac_pen + (G @ dW @ A * dW).sum()
                            else:
                                kfac_pen = kfac_pen + (fisher_diag_splits.get(name, torch.zeros_like(dW)) * dW.pow(2)).sum()
                        elif name in fisher_diag_splits:
                            kfac_pen = kfac_pen + (fisher_diag_splits[name] * dW.pow(2)).sum()
                kfac_pen = kfac_pen / max(fisher_trace, 1e-8)

                loss = ce + ewc_scale * kfac_pen
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        delta_time = time.time() - t_start

        ece_after = self._calibration.compute_ece(
            self.model, val_loader if val_loader else new_dataloader, self.device)

        cert = self._certificate.compute(
            model=self.model, state=self.state, new_loader=new_dataloader,
            shift_type=shift_type, fisher_computer=self._fisher,
            ece_before=ece_before, ece_after=ece_after,
            delta_time=delta_time, full_retrain_time=self._last_full_retrain_time,
            device=self.device, n_old=n_old, n_new=n_new,
            ce_scale=ce_scale, ewc_scale=ewc_scale,
        )

        new_state = self._fisher.compute(self.model, new_dataloader, self.device)
        alpha = float(n_new) / float(max(n_total, 1))
        for name in new_state.kfac_A:
            if name in self.state.kfac_A:
                self.state.kfac_A[name] = (
                    (1 - alpha) * self.state.kfac_A[name] + alpha * new_state.kfac_A[name]
                )
            else:
                self.state.kfac_A[name] = new_state.kfac_A[name]
        for name in new_state.kfac_G:
            if name in self.state.kfac_G:
                self.state.kfac_G[name] = (
                    (1 - alpha) * self.state.kfac_G[name] + alpha * new_state.kfac_G[name]
                )
            else:
                self.state.kfac_G[name] = new_state.kfac_G[name]
        self.state.kfac_param_names = sorted(
            set(self.state.kfac_param_names) | set(new_state.kfac_param_names)
        )
        self.state.theta_ref = {
            n: p.detach().cpu().numpy().copy()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        if (
            new_state.fisher_diag is not None
            and self.state.fisher_diag is not None
            and new_state.fisher_diag.shape == self.state.fisher_diag.shape
        ):
            self.state.fisher_diag = (
                (1 - alpha) * self.state.fisher_diag + alpha * new_state.fisher_diag
            )
        elif new_state.fisher_diag is not None:
            self.state.fisher_diag = new_state.fisher_diag
        self._shift.update_state(new_dataloader, self.model, self.state, self.device)
        self.state.n_old = n_total
        return cert

    def save_state(self, path: str) -> None:
        if self.state is None:
            raise RuntimeError("No state to save.")
        self.state.save(path)

    def load_state(self, path: str) -> None:
        self.state = DeltaState.load(path)

    def certify(self, val_loader: DataLoader, full_retrain_time: float = 0.0) -> EquivalenceCertificate:
        if self.state is None:
            raise RuntimeError("No state.")
        ece = self._calibration.compute_ece(self.model, val_loader, self.device)
        return self._certificate.compute(
            model=self.model, state=self.state, new_loader=val_loader,
            shift_type="none", fisher_computer=self._fisher,
            ece_before=ece, ece_after=ece, delta_time=0.0,
            full_retrain_time=full_retrain_time, device=self.device,
            n_old=self.state.n_old, n_new=0, ce_scale=1.0, ewc_scale=0.0,
        )
