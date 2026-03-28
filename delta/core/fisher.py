"""
KFAC Fisher Information Computer.

Implements correct Fisher computation per van de Ven 2025:
  - Uses model output distribution p(y|x,theta) not empirical labels
  - Forward hooks capture input activations A_l
  - Backward hooks capture pre-activation gradient covariance G_l
  - Supports nn.Linear AND nn.Conv2d layers for KFAC
  - All other parameter types get diagonal Fisher as fallback
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Union
from .state import DeltaState


# Layer types that get full KFAC treatment
_KFAC_LAYER_TYPES = (nn.Linear, nn.Conv2d)


class KFACComputer:

    def compute(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        n_samples: int = 512,
    ) -> DeltaState:
        state = DeltaState()
        state.theta_ref = {
            n: p.detach().cpu().numpy().copy()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        handles = []
        activations: dict[str, list[Tensor]] = {}
        grad_outputs: dict[str, list[Tensor]] = {}
        layer_types: dict[str, str] = {}  # "linear" or "conv2d"

        for name, module in model.named_modules():
            if isinstance(module, _KFAC_LAYER_TYPES):
                activations[name] = []
                grad_outputs[name] = []
                layer_types[name] = (
                    "linear" if isinstance(module, nn.Linear) else "conv2d"
                )

                def make_forward_hook(n, ltype, mod_ref):
                    def hook(mod, inp, out):
                        a = inp[0].detach()
                        if ltype == "conv2d":
                            # Unfold input patches to match kernel shape
                            # Input: (B, C_in, H, W)
                            # Unfold: (B, C_in*kH*kW, L) where L = output spatial elements
                            k = mod_ref.kernel_size
                            p = mod_ref.padding
                            s = mod_ref.stride
                            kh = k[0] if isinstance(k, tuple) else k
                            kw = k[1] if isinstance(k, tuple) else k
                            ph = p[0] if isinstance(p, tuple) else p
                            pw = p[1] if isinstance(p, tuple) else p
                            sh = s[0] if isinstance(s, tuple) else s
                            sw = s[1] if isinstance(s, tuple) else s
                            unfolded = torch.nn.functional.unfold(
                                a, (kh, kw), padding=(ph, pw), stride=(sh, sw)
                            )  # (B, C_in*kH*kW, L)
                            # Reshape to (B*L, C_in*kH*kW)
                            a = unfolded.permute(0, 2, 1).reshape(-1, unfolded.shape[1])
                        else:
                            a = a.reshape(-1, a.shape[-1])
                        activations[n].append(a)
                    return hook

                def make_backward_hook(n, ltype):
                    def hook(mod, grad_in, grad_out):
                        g = grad_out[0].detach()
                        if ltype == "conv2d":
                            # (B, C_out, H, W) -> (B*H*W, C_out)
                            g = g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
                        else:
                            g = g.reshape(-1, g.shape[-1])
                        grad_outputs[n].append(g)
                    return hook

                handles.append(
                    module.register_forward_hook(
                        make_forward_hook(name, layer_types[name], module)
                    )
                )
                handles.append(
                    module.register_full_backward_hook(
                        make_backward_hook(name, layer_types[name])
                    )
                )

        fisher_diag_accum: dict[str, Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        model.train()
        n_seen = 0
        n_batches = 0

        for inputs, targets in dataloader:
            if n_seen >= n_samples:
                break
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if isinstance(v, Tensor) else v
                          for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)

            # Fisher uses model's own distribution, not empirical labels
            with torch.no_grad():
                logits = model(inputs)
                probs = torch.softmax(logits, dim=-1)
                sampled_labels = torch.multinomial(probs, 1).squeeze(1)

            model.zero_grad()
            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.nll_loss(log_probs, sampled_labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diag_accum[name] += param.grad.detach().pow(2)

            batch_size = (
                inputs.shape[0]
                if not isinstance(inputs, dict)
                else list(inputs.values())[0].shape[0]
            )
            n_seen += batch_size
            n_batches += 1

        n_batches = max(1, n_batches)
        flat_fisher = torch.cat([
            fisher_diag_accum[n].flatten() / n_batches
            for n in sorted(fisher_diag_accum.keys())
        ])
        state.fisher_diag = flat_fisher.cpu().numpy()
        state.fisher_eigenvalue_max = float(flat_fisher.max().item())

        # Build KFAC factors for Linear and Conv2d layers
        for name in list(activations.keys()):
            acts = activations[name]
            grads = grad_outputs[name]
            if not acts or not grads:
                continue
            A_sum = torch.zeros(
                acts[0].shape[1], acts[0].shape[1], device=device
            )
            G_sum = torch.zeros(
                grads[0].shape[1], grads[0].shape[1], device=device
            )
            for a in acts:
                A_sum += a.T @ a
            for g in grads:
                G_sum += g.T @ g
            n_a = max(1, sum(a.shape[0] for a in acts))
            n_g = max(1, sum(g.shape[0] for g in grads))
            state.kfac_A[name] = (A_sum / n_a).cpu().numpy()
            state.kfac_G[name] = (G_sum / n_g).cpu().numpy()
            state.kfac_param_names.append(name + ".weight")

        for h in handles:
            h.remove()
        return state

    def approximation_error(
        self,
        state: DeltaState,
        model: nn.Module,
        device: torch.device,
        sample_loader=None,
    ) -> float:
        if not state.kfac_A or not state.kfac_G:
            return float("inf")
        errors = []
        for name in state.kfac_param_names:
            layer_name = name.replace(".weight", "")
            if layer_name not in state.kfac_A:
                continue
            A = torch.from_numpy(state.kfac_A[layer_name]).to(device)
            G = torch.from_numpy(state.kfac_G[layer_name]).to(device)
            eig_A = float(torch.linalg.eigvalsh(A).max().item())
            eig_G = float(torch.linalg.eigvalsh(G).max().item())
            kfac_max = eig_A * eig_G
            if state.fisher_eigenvalue_max > 0:
                rel_error = abs(kfac_max - state.fisher_eigenvalue_max) / (
                    state.fisher_eigenvalue_max + 1e-8
                )
                errors.append(rel_error)
        return float(np.mean(errors)) if errors else 0.01

# Update 15 - 2026-03-29 01:48:22
# Update 16 - 2026-03-29 03:02:47
# Update 29 - 2026-03-29 02:14:10
# Update 15 @ 2026-03-28 18:43:27
# Update 21 @ 2026-03-29 00:45:50
# Update 33 @ 2026-03-29 08:55:48
# Update 15 @ 2026-03-28 13:16:37
# Update 20 @ 2026-03-28 11:05:20
# Update 28 @ 2026-03-28 13:21:22