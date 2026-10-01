import argparse
import json
import time
from pathlib import Path


def stage_defaults():
    return {
        "manifest": {"dataset": None, "root": None, "split": "eval", "out": None, "train_ratio": 0.0},
        "subset": {"manifest": None, "out": None, "n": 800, "seed": 1337, "split": "eval"},
        "merge": {"train_from": None, "eval_from": None, "out": None},
        "verify-dataset": {"manifest": None, "min_duration": 0.3, "expect_total": None, "expect_bona": None, "expect_spoof": None},
        "run": {"config": "configs/default.yaml", "models": "configs/models.yaml", "manifest": None, "detector": None, "run_dir": None},
        "sweep": {"config": "configs/default.yaml", "models": "configs/models.yaml", "distortions": "configs/distortions.yaml", "manifest": None, "detector": None, "only": None, "clusters": 6, "run_dir": None},
        "attack": {"config": "configs/default.yaml", "models": "configs/models.yaml", "attacks": "configs/attacks.yaml", "manifest": None, "detector": "torch_reference", "only": None, "budget": "snr", "transfer": None, "defences": None, "run_dir": None},
        "combine": {"error_table": [], "out": "runs/combined", "clusters": 6, "seed": 1337},
        "inverse": {"error_table": [], "out": "runs/inverse", "seed": 1337},
        "eda": {"error_table": [], "out": "runs/eda", "seed": 1337},
        "align": {"config": "configs/default.yaml", "models": "configs/models.yaml", "manifest": None, "detector": None, "n": 300, "seed": 1337, "run_dir": None},
        "gradcheck": {"config": "configs/default.yaml", "models": "configs/models.yaml", "manifest": None, "detector": None, "n": 40, "seed": 1337, "fine_snr": 80.0, "attack_snr": 50.0, "directions": 8, "trajectory_files": 10, "trajectory_snr": 45.0, "trajectory_steps": 100, "run_dir": None},
        "perceptual": {"config": "configs/default.yaml", "models": "configs/models.yaml", "manifest": None, "detector": None, "attacks": None, "n": 40, "seed": 1337, "save_n": 4, "run_dir": None},
        "report": {"pipeline": "configs/pipeline.yaml", "out": "runs/pipeline/report"},
    }


def _pipeline_dir(base):
    path = Path(base) / "pipeline"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_status(pipeline_dir, status):
    (pipeline_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))


def run_stages(stages, dispatch, defaults, base="runs", force=False):
    pipeline_dir = _pipeline_dir(base)
    status = []
    for stage in stages:
        name = stage["name"]
        stage_type = stage["type"]
        optional = bool(stage.get("optional", False))
        marker = pipeline_dir / f"{name}.done"
        if not stage.get("enabled", True):
            status.append({"name": name, "type": stage_type, "state": "disabled"})
            _write_status(pipeline_dir, status)
            continue
        if marker.exists() and not force and not stage.get("always"):
            status.append({"name": name, "type": stage_type, "state": "cached"})
            _write_status(pipeline_dir, status)
            print(f"[pipeline] cached: {name}", flush=True)
            continue
        function = dispatch.get(stage_type)
        if function is None:
            raise ValueError(f"unknown stage type '{stage_type}' in stage '{name}'")
        args = argparse.Namespace(**{**defaults.get(stage_type, {}), **(stage.get("params") or {})})
        print(f"[pipeline] >>> {name} ({stage_type})", flush=True)
        started = time.time()
        try:
            function(args)
        except (Exception, SystemExit) as error:
            code = error.code if isinstance(error, SystemExit) else None
            if isinstance(error, SystemExit) and code in (0, None):
                pass
            else:
                reason = f"exit {code}" if isinstance(error, SystemExit) else f"{type(error).__name__}: {error}"
                state = "skipped" if optional else "failed"
                status.append({"name": name, "type": stage_type, "state": state, "reason": reason})
                _write_status(pipeline_dir, status)
                print(f"[pipeline] {state}: {name} ({reason})", flush=True)
                if optional:
                    continue
                raise
        seconds = round(time.time() - started, 1)
        marker.write_text(json.dumps({"seconds": seconds}))
        status.append({"name": name, "type": stage_type, "state": "done", "seconds": seconds})
        _write_status(pipeline_dir, status)
        print(f"[pipeline] done: {name} ({seconds}s)", flush=True)
    _write_status(pipeline_dir, status)
    return status
