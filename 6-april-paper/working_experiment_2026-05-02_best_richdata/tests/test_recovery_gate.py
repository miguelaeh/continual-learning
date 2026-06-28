from smf_retrofit.eval import recovery_gate_status


def test_recovery_gate_passes_for_small_degradation():
    status = recovery_gate_status(
        base_loss=1.0,
        recovery_loss=1.15,
        max_loss_delta_vs_base=0.25,
        max_loss_ratio_vs_base=1.25,
    )
    assert status["passed"] is True


def test_recovery_gate_fails_for_large_degradation():
    status = recovery_gate_status(
        base_loss=1.0,
        recovery_loss=2.0,
        max_loss_delta_vs_base=0.25,
        max_loss_ratio_vs_base=1.25,
    )
    assert status["passed"] is False
