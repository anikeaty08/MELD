"""Snapshot capture logic for MELD."""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..interfaces.base import SnapshotStrategy, TaskSnapshot


def _move_inputs_to_device(inputs: Tensor | dict[str, Tensor] | tuple[Tensor, ...], device: torch.device) -> Tensor | dict[str, Tensor] | tuple[Tensor, ...]:
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in inputs.items()
        }
    if isinstance(inputs, tuple):
        return tuple(value.to(device) if isinstance(value, Tensor) else value for value in inputs)
    return inputs.to(device)


class FisherManifoldSnapshot(SnapshotStrategy):
    def __init__(
        self,
        fisher_samples: int = 512,
        covariance_eps: float = 1e-6,
        anchors_per_class: int = 20,
        input_feature_samples: int = 128,
    ) -> None:
        self.fisher_samples = fisher_samples
        self.covariance_eps = covariance_eps
        self.anchors_per_class = anchors_per_class
        self.input_feature_samples = input_feature_samples
        self._ema_fisher: np.ndarray | None = None
        self._ema_decay: float = 0.9
        # EMA factors for K-FAC style curvature approximation on a small set of
        # parameters (last 2 linear layers).
        self._ema_kfac_A: dict[str, np.ndarray] = {}
        self._ema_kfac_G: dict[str, np.ndarray] = {}

    def capture(self, model: nn.Module, dataloader: DataLoader, class_ids: list[int], task_id: int) -> TaskSnapshot:
        device = next(model.parameters()).device
        model.eval()
        valid_class_ids = [int(class_id) for class_id in class_ids if int(class_id) in model.classifier.class_to_head]
        protected_head_indices = sorted(
            {
                model.classifier.class_to_head[class_id][0]
                for class_id in valid_class_ids
            }
        )
        protected_parameter_names = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and self._is_protected_parameter(name, protected_head_indices)
        ]
        if not protected_parameter_names:
            protected_parameter_names = [
                name for name, param in model.named_parameters() if param.requires_grad
            ]
        class_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
        class_inputs: dict[int, list[np.ndarray]] = defaultdict(list)
        class_logits: dict[int, list[np.ndarray]] = defaultdict(list)
        input_feature_batches: list[np.ndarray] = []
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = _move_inputs_to_device(inputs, device)
                input_feature_batches.append(self._input_features(inputs).detach().cpu().numpy())
                embeddings_tensor = model.embed(inputs)
                logits_tensor = model.classifier(embeddings_tensor)
                if valid_class_ids:
                    logits_tensor = logits_tensor.index_select(
                        1,
                        torch.tensor(valid_class_ids, device=device, dtype=torch.long),
                    )
                else:
                    logits_tensor = logits_tensor[:, :0]
                embeddings = embeddings_tensor.detach().cpu().numpy()
                logits = logits_tensor.detach().cpu().numpy()
                targets_list = targets.tolist()
                for i, target in enumerate(targets_list):
                    tid = int(target)
                    if tid in valid_class_ids:
                        sample_input = self._sample_input_array(inputs, i)
                        if sample_input is not None:
                            class_inputs[tid].append(sample_input)
                        class_embeddings[tid].append(embeddings[i])
                        class_logits[tid].append(logits[i])
                        total_samples += 1

        class_means: dict[int, np.ndarray] = {}
        class_covs: dict[int, np.ndarray] = {}
        class_anchors: dict[int, np.ndarray] = {}
        class_anchor_inputs: dict[int, np.ndarray] = {}
        class_anchor_logits: dict[int, np.ndarray] = {}
        available_class_ids: list[int] = []
        for class_id in valid_class_ids:
            inputs_list = class_inputs.get(class_id, [])
            vectors_list = class_embeddings.get(class_id, [])
            logits_list = class_logits.get(class_id, [])
            if not vectors_list or not logits_list:
                continue
            if inputs_list and len(inputs_list) == len(vectors_list):
                raw_inputs = np.nan_to_num(
                    np.stack(inputs_list, axis=0),
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                ).astype(np.float32, copy=False)
            else:
                raw_inputs = np.empty((0, 0), dtype=np.float32)
            vectors = np.nan_to_num(np.stack(vectors_list, axis=0), nan=0.0, posinf=1e6, neginf=-1e6)
            logits = np.nan_to_num(np.stack(logits_list, axis=0), nan=0.0, posinf=1e6, neginf=-1e6)
            class_means[class_id] = np.nan_to_num(vectors.mean(axis=0), nan=0.0, posinf=1e6, neginf=-1e6).astype(
                np.float32,
                copy=False,
            )
            class_covs[class_id] = np.clip(
                np.nan_to_num(vectors.var(axis=0), nan=0.0, posinf=1e6, neginf=0.0) + self.covariance_eps,
                self.covariance_eps,
                1e6,
            ).astype(np.float32, copy=False)
            anchor_count = min(self.anchors_per_class, len(vectors))
            if anchor_count > 0:
                indices = np.linspace(0, len(vectors) - 1, num=anchor_count, dtype=int)
                if raw_inputs.shape[0] >= len(vectors):
                    class_anchor_inputs[class_id] = raw_inputs[indices]
                else:
                    class_anchor_inputs[class_id] = np.empty((0, 0), dtype=np.float32)
                class_anchors[class_id] = vectors[indices]
                class_anchor_logits[class_id] = logits[indices]
            else:
                class_anchor_inputs[class_id] = raw_inputs[:0]
                class_anchors[class_id] = vectors[:0]
                class_anchor_logits[class_id] = logits[:0]
            available_class_ids.append(class_id)

        classifier_weights = {
            class_id: model.classifier.weight_vector(class_id).detach().cpu().numpy().copy()
            for class_id in available_class_ids
        }
        classifier_biases = {
            class_id: float(model.classifier.bias_value(class_id).detach().cpu().item())
            for class_id in available_class_ids
        }
        if input_feature_batches:
            input_features = np.concatenate(input_feature_batches, axis=0).astype(np.float32, copy=False)
            input_feature_mean = np.nan_to_num(input_features.mean(axis=0), nan=0.0, posinf=1e6, neginf=-1e6)
            input_feature_var = np.clip(
                np.nan_to_num(input_features.var(axis=0), nan=0.0, posinf=1e6, neginf=0.0) + self.covariance_eps,
                self.covariance_eps,
                1e6,
            )
            sample_count = min(self.input_feature_samples, len(input_features))
            if sample_count > 0:
                indices = np.linspace(0, len(input_features) - 1, num=sample_count, dtype=int)
                input_feature_samples = np.nan_to_num(
                    input_features[indices],
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                ).astype(np.float32, copy=False)
            else:
                input_feature_samples = np.empty((0, 0), dtype=np.float32)
        else:
            input_feature_mean = np.array([], dtype=np.float32)
            input_feature_var = np.array([], dtype=np.float32)
            input_feature_samples = np.empty((0, 0), dtype=np.float32)

        fisher_samples = self.fisher_samples
        try:
            fisher_samples = min(fisher_samples, int(len(dataloader.dataset)))
        except Exception:
            pass

        (
            fisher_diagonal,
            mean_gradient_norm,
            kfac_weight_param_names,
            kfac_factors_A,
            kfac_factors_G,
            kfac_eig_max,
        ) = self._compute_fisher(model, dataloader, fisher_samples, protected_parameter_names)
        fisher_diagonal = np.clip(
            np.nan_to_num(fisher_diagonal, nan=0.0, posinf=1e6, neginf=0.0),
            0.0,
            1e6,
        ).astype(np.float32, copy=False)
        if self._ema_fisher is not None and self._ema_fisher.shape == fisher_diagonal.shape:
            fisher_diagonal = self._ema_decay * self._ema_fisher + (1.0 - self._ema_decay) * fisher_diagonal
        fisher_diagonal = np.clip(
            np.nan_to_num(fisher_diagonal, nan=0.0, posinf=1e6, neginf=0.0),
            0.0,
            1e6,
        ).astype(np.float32, copy=False)
        mean_gradient_norm = float(np.nan_to_num(mean_gradient_norm, nan=0.0, posinf=1e6, neginf=0.0))
        kfac_eig_max = float(np.nan_to_num(kfac_eig_max, nan=0.0, posinf=1e6, neginf=0.0))
        if np.isfinite(fisher_diagonal).all():
            self._ema_fisher = fisher_diagonal.copy()
        parameter_reference = [
            param.detach().cpu().numpy().copy()
            for name, param in model.named_parameters()
            if param.requires_grad and name in protected_parameter_names
        ]
        steps_per_epoch = max(1, math.ceil(max(1, len(dataloader.dataset)) / max(1, dataloader.batch_size or 1)))
        diag_eig_max = float(np.max(fisher_diagonal)) if fisher_diagonal.size else 0.0
        raw_max = max(diag_eig_max, float(kfac_eig_max))
        fisher_eigenvalue_max = float(np.clip(raw_max, 0.0, 1000.0))

        return TaskSnapshot(
            task_id=task_id,
            class_ids=available_class_ids,
            class_means=class_means,
            class_covs=class_covs,
            class_anchors=class_anchors,
            class_anchor_inputs=class_anchor_inputs,
            class_anchor_logits=class_anchor_logits,
            classifier_norms=model.classifier.all_norms(),
            fisher_diagonal=fisher_diagonal,
            fisher_eigenvalue_max=fisher_eigenvalue_max,
            mean_gradient_norm=mean_gradient_norm,
            timestamp=time.time(),
            embedding_dim=int(next(iter(class_means.values())).shape[0]) if class_means else int(model.out_dim),
            dataset_size=len(dataloader.dataset),
            steps_per_epoch=steps_per_epoch,
            parameter_reference=parameter_reference,
            protected_parameter_names=protected_parameter_names,
            classifier_weights=classifier_weights,
            classifier_biases=classifier_biases,
            importance_weights={},
            input_feature_mean=input_feature_mean,
            input_feature_var=input_feature_var,
            input_feature_samples=input_feature_samples,
            kfac_weight_param_names=kfac_weight_param_names,
            kfac_factors_A=kfac_factors_A,
            kfac_factors_G=kfac_factors_G,
        )

    def _compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        fisher_samples: int,
        protected_parameter_names: list[str] | None = None,
    ) -> tuple[
        np.ndarray,
        float,
        list[str],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        float,
    ]:
        device = next(model.parameters()).device
        protected = set(protected_parameter_names or [])
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and (not protected or name in protected)
        ]
        params = [param for _, param in named_params]
        fisher = [torch.zeros_like(param, device=device) for param in params]
        grad_norms = []
        total = 0
        criterion = nn.CrossEntropyLoss()

        # K-FAC factors for all sufficiently large Linear layers.
        linear_modules = [
            (name, module)
            for name, module in model.named_modules()
            if (
                isinstance(module, nn.Linear)
                and module.weight.numel() > 1000
                and (not protected or f"{name}.weight" in protected)
            )
        ]
        if linear_modules:
            kfac_modules = linear_modules
        else:
            kfac_modules = [
                (name, module)
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and (not protected or f"{name}.weight" in protected)
            ]
        name_by_param = {p: n for n, p in model.named_parameters()}

        kfac_weight_param_names: list[str] = []
        for _, m in kfac_modules:
            pname = name_by_param.get(m.weight)
            if pname is not None:
                kfac_weight_param_names.append(pname)

        activations: dict[str, Tensor] = {}
        grad_outputs: dict[str, Tensor] = {}
        A_acc: dict[str, Tensor] = {}
        G_acc: dict[str, Tensor] = {}
        handles = []

        for _, m in kfac_modules:
            pname = name_by_param.get(m.weight)
            if pname is None:
                continue
            in_dim = int(m.in_features)
            out_dim = int(m.out_features)
            A_acc[pname] = torch.zeros((in_dim, in_dim), device=device)
            G_acc[pname] = torch.zeros((out_dim, out_dim), device=device)

            def _make_fwd_hook(pn: str):
                def _hook(mod: nn.Module, inp: tuple[Tensor, ...], out: Tensor) -> None:
                    activations[pn] = inp[0].detach()
                return _hook

            def _make_bwd_hook(pn: str):
                def _hook(mod: nn.Module, grad_in: tuple[Tensor, ...], grad_out: tuple[Tensor, ...]) -> None:
                    grad_outputs[pn] = grad_out[0].detach()
                return _hook

            handles.append(m.register_forward_hook(_make_fwd_hook(pname)))
            handles.append(m.register_full_backward_hook(_make_bwd_hook(pname)))

        try:
            for inputs, targets in dataloader:
                if total >= fisher_samples:
                    break
                inputs = _move_inputs_to_device(inputs, device)
                targets = targets.to(device)
                remaining = fisher_samples - total
                batch_size = self._batch_size(inputs)
                if batch_size > remaining:
                    inputs = self._slice_batch(inputs, remaining)
                    targets = targets[:remaining]

                model.zero_grad(set_to_none=True)
                logits = torch.nan_to_num(model(inputs), nan=0.0, posinf=1e4, neginf=-1e4)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    continue
                loss.backward()

                batch_size = targets.size(0)
                for index, param in enumerate(params):
                    if param.grad is None:
                        continue
                    grad = torch.nan_to_num(param.grad.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
                    fisher[index] += grad.pow(2) * batch_size

                # Mean grad norm proxy (used for oracle calibration).
                per_param_norms = [
                    torch.nan_to_num(param.grad.detach(), nan=0.0, posinf=1e3, neginf=-1e3).pow(2).sum()
                    for param in params
                    if param.grad is not None
                ]
                if per_param_norms:
                    grad_norm = torch.sqrt(torch.stack(per_param_norms).sum())
                    grad_norms.append(float(grad_norm.item()))

                # K-FAC accumulation from hooked activations/gradients.
                if kfac_weight_param_names:
                    for pname in kfac_weight_param_names:
                        a = activations.get(pname)
                        g = grad_outputs.get(pname)
                        if a is None or g is None:
                            continue
                        a = torch.nan_to_num(a, nan=0.0, posinf=1e3, neginf=-1e3)
                        g = torch.nan_to_num(g, nan=0.0, posinf=1e3, neginf=-1e3)
                        A_acc[pname] += a.t() @ a
                        G_acc[pname] += g.t() @ g

                total += batch_size
        finally:
            for h in handles:
                h.remove()

        if total == 0:
            return np.array([], dtype=np.float32), 0.0, [], {}, {}, 0.0

        flat = torch.cat([(entry / total).reshape(-1) for entry in fisher])
        fisher_np = np.clip(
            np.nan_to_num(flat.detach().cpu().numpy(), nan=0.0, posinf=1e6, neginf=0.0),
            0.0,
            1e6,
        ).astype(np.float32, copy=False)
        mean_grad_norm = float(np.nan_to_num(np.mean(grad_norms), nan=0.0, posinf=1e6, neginf=0.0)) if grad_norms else 0.0

        kfac_A_np: dict[str, np.ndarray] = {}
        kfac_G_np: dict[str, np.ndarray] = {}
        kfac_eig_max = 0.0

        for pname in kfac_weight_param_names:
            A = (A_acc[pname] / total).detach()
            G = (G_acc[pname] / total).detach()

            # Optional EMA smoothing for K-FAC factors.
            old_A = self._ema_kfac_A.get(pname)
            old_G = self._ema_kfac_G.get(pname)
            if old_A is not None and old_A.shape == tuple(A.shape):
                A = self._ema_decay * torch.from_numpy(old_A).to(device) + (1.0 - self._ema_decay) * A
            if old_G is not None and old_G.shape == tuple(G.shape):
                G = self._ema_decay * torch.from_numpy(old_G).to(device) + (1.0 - self._ema_decay) * G

            A_np = A.detach().cpu().numpy().astype(np.float32, copy=False)
            G_np = G.detach().cpu().numpy().astype(np.float32, copy=False)
            A_np = np.nan_to_num(0.5 * (A_np + A_np.T), nan=0.0, posinf=1e6, neginf=-1e6)
            G_np = np.nan_to_num(0.5 * (G_np + G_np.T), nan=0.0, posinf=1e6, neginf=-1e6)
            A_np = A_np + np.eye(A_np.shape[0], dtype=np.float32) * self.covariance_eps
            G_np = G_np + np.eye(G_np.shape[0], dtype=np.float32) * self.covariance_eps
            self._ema_kfac_A[pname] = A_np.copy()
            self._ema_kfac_G[pname] = G_np.copy()

            # Upper-bound-ish spectral proxy: maxeig(A) * maxeig(G).
            eigA = np.linalg.eigvalsh(A_np)
            eigG = np.linalg.eigvalsh(G_np)
            eigA = np.clip(eigA, 0.0, None)
            eigG = np.clip(eigG, 0.0, None)
            est = float(np.max(eigA) * np.max(eigG))
            kfac_eig_max = max(kfac_eig_max, est)

            kfac_A_np[pname] = A_np
            kfac_G_np[pname] = G_np

        return fisher_np, mean_grad_norm, kfac_weight_param_names, kfac_A_np, kfac_G_np, kfac_eig_max

    @staticmethod
    def _is_protected_parameter(name: str, protected_head_indices: list[int]) -> bool:
        if name.startswith("backbone."):
            return True
        return any(name.startswith(f"classifier.heads.{head_index}.") for head_index in protected_head_indices)

    @staticmethod
    def _sample_input_array(inputs: Tensor | dict[str, Tensor] | tuple[Tensor, ...], index: int) -> np.ndarray | None:
        if isinstance(inputs, Tensor):
            return inputs[index].detach().cpu().numpy()
        return None

    @staticmethod
    def _batch_size(inputs: Tensor | dict[str, Tensor] | tuple[Tensor, ...]) -> int:
        if isinstance(inputs, Tensor):
            return int(inputs.size(0))
        if isinstance(inputs, dict):
            for value in inputs.values():
                if isinstance(value, Tensor):
                    return int(value.size(0))
            return 0
        if isinstance(inputs, tuple):
            for value in inputs:
                if isinstance(value, Tensor):
                    return int(value.size(0))
            return 0
        return 0

    @staticmethod
    def _slice_batch(inputs: Tensor | dict[str, Tensor] | tuple[Tensor, ...], size: int) -> Tensor | dict[str, Tensor] | tuple[Tensor, ...]:
        if isinstance(inputs, Tensor):
            return inputs[:size]
        if isinstance(inputs, dict):
            return {
                key: value[:size] if isinstance(value, Tensor) else value
                for key, value in inputs.items()
            }
        if isinstance(inputs, tuple):
            return tuple(value[:size] if isinstance(value, Tensor) else value for value in inputs)
        return inputs

    @staticmethod
    def _input_features(inputs: Tensor | dict[str, Tensor] | tuple[Tensor, ...]) -> Tensor:
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            if isinstance(input_ids, Tensor) and isinstance(attention_mask, Tensor):
                ids = input_ids.float()
                mask = attention_mask.float()
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                masked_ids = ids * mask
                token_mean = masked_ids.sum(dim=1, keepdim=True) / lengths
                centered = (ids - token_mean) * mask
                token_std = torch.sqrt(centered.pow(2).sum(dim=1, keepdim=True) / lengths)
                density = lengths / max(ids.size(1), 1)
                return torch.cat((token_mean, token_std, lengths, density), dim=1)
            batch = FisherManifoldSnapshot._batch_size(inputs)
            return torch.zeros((batch, 4), dtype=torch.float32)
        if isinstance(inputs, tuple):
            tensor_inputs = next((value for value in inputs if isinstance(value, Tensor)), None)
            if tensor_inputs is None:
                return torch.zeros((0, 4), dtype=torch.float32)
            inputs = tensor_inputs
        channel_mean = inputs.mean(dim=(2, 3))
        channel_std = inputs.std(dim=(2, 3), unbiased=False)
        flat = inputs.flatten(start_dim=1)
        global_mean = flat.mean(dim=1, keepdim=True)
        global_std = flat.std(dim=1, unbiased=False, keepdim=True)
        return torch.cat((channel_mean, channel_std, global_mean, global_std), dim=1)
