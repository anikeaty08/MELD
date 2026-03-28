"""
3-Way Distribution Shift Detector.
Returns exactly one of: "none" | "covariate" | "concept"

Class-incremental aware:
  - New classes appearing is NOT concept drift (it's task expansion)
  - Concept drift = P(Y|X) changed for EXISTING classes
  - Only tests label shift on classes shared between old and new data
"""

import numpy as np
import torch
from torch import Tensor
from .state import DeltaState

try:
    from scipy.stats import chisquare
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ShiftDetector:
    MMD_THRESHOLD = 0.05
    CHISQ_ALPHA = 0.05

    def detect(self, new_loader, state: DeltaState, model, device: torch.device) -> str:
        if state.input_mean is None:
            return "none"
        new_embeddings, new_labels = self._collect(model, new_loader, device)
        if len(new_embeddings) == 0:
            return "none"
        mmd = self._compute_mmd(new_embeddings, state)
        label_shifted = self._test_label_shift(new_labels, state)
        if label_shifted:
            return "concept"
        elif mmd > self.MMD_THRESHOLD:
            return "covariate"
        else:
            return "none"

    def update_state(self, new_loader, model, state: DeltaState, device: torch.device) -> None:
        embeddings, labels = self._collect(model, new_loader, device)
        if len(embeddings) == 0:
            return
        emb_np = embeddings.cpu().numpy()
        state.input_mean = emb_np.mean(axis=0)
        state.input_var = emb_np.var(axis=0) + 1e-6
        # label_counts ownership lives here so drift detection sees a
        # single, consistent history across strategies.
        for label in labels.cpu().numpy().tolist():
            label = int(label)
            state.label_counts[label] = state.label_counts.get(label, 0) + 1

    def _collect(self, model, loader, device):
        model.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    continue
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                if hasattr(model, "embed"):
                    emb = model.embed(inputs)
                else:
                    emb = model(inputs)
                embeddings.append(emb.cpu())
                labels.append(targets.cpu())
        if not embeddings:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

    def _compute_mmd(self, new_emb: Tensor, state: DeltaState) -> float:
        if state.input_mean is None or state.input_var is None:
            return 0.0
        new_np = new_emb.numpy()[:200]
        n = min(200, len(new_np))
        old_samples = (
            state.input_mean[None, :] +
            np.random.randn(n, len(state.input_mean)) * np.sqrt(state.input_var)[None, :]
        )
        dists = np.sqrt(((new_np[:50, None] - new_np[None, :50]) ** 2).sum(axis=-1))
        bandwidth = float(np.median(dists[dists > 0])) + 1e-6
        def rbf(x, y):
            d2 = ((x[:, None] - y[None, :]) ** 2).sum(axis=-1)
            return np.exp(-d2 / (2 * bandwidth ** 2))
        Knn = rbf(new_np, new_np)
        Koo = rbf(old_samples, old_samples)
        Kno = rbf(new_np, old_samples)
        mmd2 = Knn.mean() - 2 * Kno.mean() + Koo.mean()
        return float(max(mmd2, 0.0))

    def _test_label_shift(self, new_labels: Tensor, state: DeltaState) -> bool:
        """Test for concept drift on SHARED classes only.

        New classes appearing is class expansion (normal in CIL),
        not concept drift. We only flag concept drift when the
        label distribution of previously-seen classes changes
        significantly.
        """
        if not state.label_counts or not SCIPY_AVAILABLE:
            return False
        new_counts: dict[int, int] = {}
        for label in new_labels.numpy().tolist():
            k = int(label)
            new_counts[k] = new_counts.get(k, 0) + 1

        # Only compare classes that exist in BOTH old and new data
        shared_classes = sorted(set(state.label_counts) & set(new_counts))
        if len(shared_classes) < 2:
            # No shared classes or only 1 — this is task expansion, not drift
            return False

        old_freq = np.array([state.label_counts.get(c, 0) for c in shared_classes], dtype=float)
        new_freq = np.array([new_counts.get(c, 0) for c in shared_classes], dtype=float)
        if old_freq.sum() == 0 or new_freq.sum() == 0:
            return False
        old_freq = old_freq / old_freq.sum() * new_freq.sum()
        old_freq = np.maximum(old_freq, 1e-6)
        try:
            _, p_value = chisquare(new_freq, f_exp=old_freq)
            return bool(p_value < self.CHISQ_ALPHA)
        except Exception:
            return False

# Update 11 - 2026-03-28 14:41:33
# Update 12 - 2026-03-29 04:43:12
# Update 17 - 2026-03-29 03:14:40
# Update 21 - 2026-03-28 18:05:58
# Update 33 - 2026-03-29 04:49:36
# Update 12 @ 2026-03-28 16:24:55
# Update 3 @ 2026-03-29 06:38:19
# Update 5 @ 2026-03-28 23:50:36
# Update 7 @ 2026-03-28 11:33:11
# Update 23 @ 2026-03-29 08:35:08
# Update 29 @ 2026-03-28 18:25:24
# Update 6 @ 2026-03-28 21:41:10
# Update 8 @ 2026-03-29 03:41:22
# Update 9 @ 2026-03-28 18:37:42
# Update 26 @ 2026-03-29 03:47:04
# Update 29 @ 2026-03-29 01:10:58