from io import BytesIO

from sensorforge.agent.memory import record_run
from sensorforge.agent.state import AgentState, Attempt, TunableParams
from sensorforge.dashboard.app import list_runs, load_run, save_uploaded_image


def _run_with_history(run_dir):
    history = [
        Attempt(iteration=0, params=TunableParams(), metrics={"deltaE2000": 12.0, "SSIM": 0.7}),
        Attempt(
            iteration=1,
            params=TunableParams(awb_gain_r=1.1),
            metrics={"deltaE2000": 2.5, "SSIM": 0.95},
        ),
    ]
    state = AgentState(history=history, best=history[1], stop_reason="within_tolerance")
    record_run(state, run_dir, run_dir.parent / "learnings.jsonl")
    return state


def test_load_run_extracts_series_and_params(tmp_path):
    rd = tmp_path / "runs" / "r1"
    _run_with_history(rd)
    run = load_run(rd)
    assert run["deltaE"] == [12.0, 2.5]
    assert run["ssim"] == [0.7, 0.95]
    assert run["best_deltaE"] == 2.5
    assert run["best_params"]["awb_gain_r"] == 1.1


def test_list_runs_finds_only_run_dirs_newest_first(tmp_path):
    runs_dir = tmp_path / "runs"
    _run_with_history(runs_dir / "20260101_000000")
    _run_with_history(runs_dir / "20260102_000000")
    (runs_dir / "not_a_run").mkdir()  # no state.json -> ignored
    found = list_runs(runs_dir)
    assert [p.name for p in found] == ["20260102_000000", "20260101_000000"]


def test_list_runs_empty_when_missing(tmp_path):
    assert list_runs(tmp_path / "nope") == []


def test_save_uploaded_image_persists_bytes(tmp_path):
    uploaded = BytesIO(b"fake image")
    uploaded.name = "../target.png"

    path = save_uploaded_image(uploaded, tmp_path)

    assert path.parent == tmp_path
    assert path.name.endswith("_target.png")
    assert path.read_bytes() == b"fake image"
