from adfd import pipeline
from adfd.eval.sweep import severity_of


def test_severity_uses_config_sign():
    assert severity_of({"kind": "noise", "severity_sign": -1.0}, 20) == -20.0
    assert severity_of({"kind": "reverb", "severity_sign": 1.0}, 0.5) == 0.5
    assert severity_of({"kind": "gain", "severity_sign": 1.0}, -10) == 10.0


def test_run_stages_optional_skips_and_required_records(tmp_path):
    calls = []

    def ok(args):
        calls.append(args.name)

    def boom(args):
        raise RuntimeError("boom")

    stages = [
        {"name": "a", "type": "run", "params": {"name": "a"}},
        {"name": "b", "type": "run", "optional": True, "params": {"name": "b"}},
        {"name": "c", "type": "run", "params": {"name": "c"}},
    ]
    dispatch = {"run": ok}
    dispatch_boom = {"run": boom}

    status = pipeline.run_stages(stages, dispatch, {"run": {"name": None}}, base=str(tmp_path))
    assert [entry["state"] for entry in status] == ["done", "done", "done"]

    stages_fail = [
        {"name": "x", "type": "run", "optional": True, "params": {"name": "x"}},
        {"name": "y", "type": "run", "params": {"name": "y"}},
    ]
    caught = False
    try:
        pipeline.run_stages(stages_fail, dispatch_boom, {"run": {"name": None}}, base=str(tmp_path / "fail"))
    except RuntimeError:
        caught = True
    assert caught


def test_run_stages_resume_marks_cached(tmp_path):
    seen = []

    def once(args):
        seen.append(args.name)

    stages = [{"name": "only", "type": "run", "params": {"name": "only"}}]
    dispatch = {"run": once}
    defaults = {"run": {"name": None}}
    first = pipeline.run_stages(stages, dispatch, defaults, base=str(tmp_path))
    second = pipeline.run_stages(stages, dispatch, defaults, base=str(tmp_path))
    assert first[0]["state"] == "done"
    assert second[0]["state"] == "cached"
    assert seen == ["only"]
