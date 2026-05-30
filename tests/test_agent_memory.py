from sensorforge.agent.memory import best_prior, load_learnings, record_run
from sensorforge.agent.state import AgentState, Attempt, TunableParams


def _state(best_de, awb_r, target="uniform"):
    attempt = Attempt(
        iteration=1, params=TunableParams(awb_gain_r=awb_r), metrics={"deltaE2000": best_de}
    )
    return AgentState(
        target=target, best=attempt, history=[attempt], stop_reason="within_tolerance"
    )


def test_record_run_writes_state_and_learning(tmp_path):
    learnings = tmp_path / "learnings.jsonl"
    out = record_run(_state(2.5, 1.1), tmp_path / "run1", learnings)
    assert out.exists()  # state.json
    rows = load_learnings(learnings)
    assert len(rows) == 1
    assert rows[0]["best_deltaE"] == 2.5 and rows[0]["target"] == "uniform"


def test_best_prior_picks_lowest_delta_e(tmp_path):
    learnings = tmp_path / "learnings.jsonl"
    record_run(_state(8.0, 1.5), tmp_path / "a", learnings)
    record_run(_state(2.1, 1.05), tmp_path / "b", learnings)  # the better run
    record_run(_state(5.0, 1.3), tmp_path / "c", learnings)
    prior = best_prior("uniform", learnings)
    assert prior is not None
    assert prior.awb_gain_r == 1.05  # from the lowest-deltaE run


def test_best_prior_filters_by_target_and_handles_empty(tmp_path):
    learnings = tmp_path / "learnings.jsonl"
    assert best_prior("uniform", learnings) is None  # nothing logged yet
    record_run(_state(1.0, 1.2, target="checkerboard"), tmp_path / "a", learnings)
    assert best_prior("uniform", learnings) is None  # different target
    assert best_prior("checkerboard", learnings).awb_gain_r == 1.2
