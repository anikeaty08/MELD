"""Deployment decision policy for MELD."""

from __future__ import annotations

from ..interfaces.base import Decision, DeployPolicy, DriftResult, VALID_DECISION_STATES


class FourStateDeployPolicy(DeployPolicy):
    def decide(self, pre_bound: float, post_bound: float, drift_result: DriftResult, config: object) -> Decision:
        bound_held = post_bound <= pre_bound
        threshold = float(getattr(config, "shift_threshold", 0.3))
        delta_time = float(getattr(config, "delta_wall_time_seconds", 0.0))
        full_time = float(getattr(config, "full_retrain_wall_time_seconds", 0.0))
        savings = ((full_time - delta_time) / full_time * 100.0) if full_time > 0 else 0.0

        if post_bound > pre_bound:
            decision = Decision(
                state="BOUND_VIOLATED",
                pre_bound=pre_bound,
                post_bound=post_bound,
                bound_held=False,
                shift_score=drift_result.shift_score,
                shift_detected=drift_result.shift_detected,
                reason="post-training drift exceeded the pre-training bound",
                compute_savings_percent=savings,
                confidence=1.0,
                recommended_action="full_retrain",
            )
            self._validate(decision)
            return decision
        if drift_result.severity == "critical":
            decision = Decision(
                state="SHIFT_CRITICAL",
                pre_bound=pre_bound,
                post_bound=post_bound,
                bound_held=bound_held,
                shift_score=drift_result.shift_score,
                shift_detected=True,
                reason="critical manifold shift detected",
                compute_savings_percent=savings,
                confidence=1.0,
                recommended_action="full_retrain",
            )
            self._validate(decision)
            return decision
        if drift_result.severity == "minor":
            confidence = max(0.4, 0.6 - (drift_result.shift_score / max(2.0 * threshold, 1e-6)) * 0.2)
            decision = Decision(
                state="CAUTIOUS_DELTA",
                pre_bound=pre_bound,
                post_bound=post_bound,
                bound_held=bound_held,
                shift_score=drift_result.shift_score,
                shift_detected=drift_result.shift_detected,
                reason="bound held with minor shift detected",
                compute_savings_percent=savings,
                confidence=confidence,
                recommended_action="delta_update",
            )
            self._validate(decision)
            return decision
        confidence = max(0.0, 1.0 - ((post_bound / pre_bound) * 0.3 if pre_bound > 0 else 0.0))
        decision = Decision(
            state="SAFE_DELTA",
            pre_bound=pre_bound,
            post_bound=post_bound,
            bound_held=bound_held,
            shift_score=drift_result.shift_score,
            shift_detected=drift_result.shift_detected,
            reason="bound held, no shift detected",
            compute_savings_percent=savings,
            confidence=confidence,
            recommended_action="delta_update",
        )
        self._validate(decision)
        return decision

    def _validate(self, decision: Decision) -> None:
        if decision.state not in VALID_DECISION_STATES:
            raise ValueError(f"Unknown decision state: {decision.state}")
