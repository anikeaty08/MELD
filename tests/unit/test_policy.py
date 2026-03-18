from meld.core.policy import FourStateDeployPolicy
from meld.interfaces.base import DriftResult


class _Config:
    shift_threshold = 0.3
    delta_wall_time_seconds = 40.0
    full_retrain_wall_time_seconds = 100.0


def test_policy_returns_safe_delta_for_clean_bound():
    policy = FourStateDeployPolicy()
    decision = policy.decide(
        pre_bound=0.1,
        post_bound=0.05,
        drift_result=DriftResult(shift_score=0.1, shift_detected=False, per_class_drift={}, severity="none"),
        config=_Config(),
    )
    assert decision.state == "SAFE_DELTA"
    assert decision.recommended_action == "delta_update"


def test_policy_prioritizes_bound_violation():
    policy = FourStateDeployPolicy()
    decision = policy.decide(
        pre_bound=0.05,
        post_bound=0.1,
        drift_result=DriftResult(shift_score=1.0, shift_detected=True, per_class_drift={}, severity="critical"),
        config=_Config(),
    )
    assert decision.state == "BOUND_VIOLATED"
