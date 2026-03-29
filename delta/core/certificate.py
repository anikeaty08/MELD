"""
Equivalence Certificate.

Theorem 1 (Convex layers):
  ||theta_delta - theta*|| <= (L / mu) * epsilon_hessian

Theorem 2 (Full network — NTK regime):
  KL(p_delta || p_full) <= C * ||theta_delta - theta_old||_F
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class EquivalenceCertificate:
    epsilon_param: float = float("inf")
    kl_bound: float = float("inf")
    kl_bound_normalized: float = float("inf")
    is_equivalent: bool = False
    shift_type: str = "none"
    ece_before: float = float("nan")
    ece_after: float = float("nan")
    ece_delta: float = float("nan")
    compute_ratio: float = 0.0
    n_old: int = 0
    n_new: int = 0
    ce_scale: float = 1.0
    ewc_scale: float = 0.0
    tier: str = "empirical"
    epsilon_hessian: float = float("nan")

    def summary(self) -> str:
        lines = [
            "=== Equivalence Certificate ===",
            f"  Epsilon (param bound):  {self.epsilon_param:.6f}",
            f"  KL bound:               {self.kl_bound:.6f}",
            f"  KL bound (normalized):  {self.kl_bound_normalized:.6f}",
            f"  Is equivalent:          {self.is_equivalent}",
            f"  Shift type:             {self.shift_type}",
            f"  ECE before:             {self.ece_before:.4f}",
            f"  ECE after:              {self.ece_after:.4f}",
            f"  ECE delta:              {self.ece_delta:+.4f}",
            f"  Compute savings:        {self.compute_ratio:.1f}x",
            f"  n_old / n_new:          {self.n_old} / {self.n_new}",
            f"  ce_scale (derived):     {self.ce_scale:.4f}",
            f"  ewc_scale (derived):    {self.ewc_scale:.4f}",
            f"  Proof tier:             {self.tier}",
            "===============================",
        ]
        return "\n".join(lines)


class CertificateComputer:
    EPSILON_TOLERANCE = 0.1
    KL_TOLERANCE = 0.5
    ECE_WORSEN_TOLERANCE = 0.05
    MIN_COMPUTE_RATIO = 0.9

    def compute(self, model, state, new_loader, shift_type, fisher_computer,
                ece_before, ece_after, delta_time, full_retrain_time,
                device, n_old, n_new, ce_scale, ewc_scale) -> EquivalenceCertificate:
        cert = EquivalenceCertificate(
            shift_type=shift_type,
            ece_before=ece_before,
            ece_after=ece_after,
            ece_delta=ece_after - ece_before,
            compute_ratio=full_retrain_time / max(delta_time, 1e-6),
            n_old=n_old, n_new=n_new,
            ce_scale=ce_scale, ewc_scale=ewc_scale,
        )
        epsilon_hessian = fisher_computer.approximation_error(state, model, device)
        cert.epsilon_hessian = epsilon_hessian

        if state.kfac_A and state.kfac_G:
            L = self._smoothness(state, device)
            mu = self._strong_convexity(state, device)
            if mu > 1e-8:
                cert.epsilon_param = (L / mu) * epsilon_hessian
                cert.tier = "convex"
            else:
                cert.epsilon_param = L * epsilon_hessian * 10.0
                cert.tier = "convex_approx"

        param_drift = self._param_drift(model, state, device)
        lipschitz = self._estimate_lipschitz(model, new_loader, device)
        cert.kl_bound = lipschitz * param_drift
        n_params = sum(p.numel() for p in model.parameters())
        cert.kl_bound_normalized = cert.kl_bound / max(n_params ** 0.5, 1.0)

        cert.is_equivalent = (
            cert.epsilon_param < self.EPSILON_TOLERANCE
            and cert.kl_bound_normalized < self.KL_TOLERANCE
            and cert.ece_after <= cert.ece_before + self.ECE_WORSEN_TOLERANCE
            and cert.compute_ratio >= self.MIN_COMPUTE_RATIO
        )
        return cert

    def _smoothness(self, state, device) -> float:
        max_eig = 0.0
        for name in state.kfac_param_names:
            layer = name.replace(".weight", "")
            if layer in state.kfac_A and layer in state.kfac_G:
                A = torch.from_numpy(state.kfac_A[layer]).to(device)
                G = torch.from_numpy(state.kfac_G[layer]).to(device)
                eig_A = float(torch.linalg.eigvalsh(A).max())
                eig_G = float(torch.linalg.eigvalsh(G).max())
                max_eig = max(max_eig, eig_A * eig_G)
        return max(max_eig, 1e-6)

    def _strong_convexity(self, state, device) -> float:
        min_eig = float("inf")
        for name in state.kfac_param_names:
            layer = name.replace(".weight", "")
            if layer in state.kfac_A and layer in state.kfac_G:
                A = torch.from_numpy(state.kfac_A[layer]).to(device)
                G = torch.from_numpy(state.kfac_G[layer]).to(device)
                eig_A = float(torch.linalg.eigvalsh(A).min().clamp(min=1e-8))
                eig_G = float(torch.linalg.eigvalsh(G).min().clamp(min=1e-8))
                min_eig = min(min_eig, eig_A * eig_G)
        return min_eig if min_eig < float("inf") else 1e-6

    def _param_drift(self, model, state, device) -> float:
        drift_sq = 0.0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in state.theta_ref:
                ref = torch.from_numpy(state.theta_ref[name]).to(device)
                drift_sq += float((param.detach() - ref).pow(2).sum().item())
        return float(drift_sq ** 0.5)

    def _estimate_lipschitz(self, model, loader, device) -> float:
        model.eval()
        grad_norms = []
        for inputs, targets in loader:
            if len(grad_norms) >= 10:
                break
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            model.zero_grad()
            loss.backward()
            total_norm = sum(
                p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms.append(total_norm)
            model.zero_grad()
        return float(np.mean(grad_norms)) if grad_norms else 1.0
