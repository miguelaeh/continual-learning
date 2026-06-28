from smf_retrofit.eval import load_prompt_specs, recovery_gate_status


def test_load_prompt_specs_accepts_string_entries(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text('["Hello", {"prompt":"What is 2 plus 2?","expected_any":["4"]}]')
    specs = load_prompt_specs(str(path))
    assert specs[0] == {"prompt": "Hello"}
    assert specs[1]["expected_any"] == ["4"]


def test_recovery_gate_fails_when_prompt_checks_fail():
    status = recovery_gate_status(
        base_loss=1.0,
        recovery_loss=0.9,
        max_loss_delta_vs_base=0.25,
        max_loss_ratio_vs_base=1.25,
        prompt_checks_passed=False,
    )
    assert status["passed"] is False
    assert status["loss_checks_passed"] is True
