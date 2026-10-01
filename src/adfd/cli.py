import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import pipeline
from .attacks.base import available_attacks
from .datasets import protocols
from .datasets.bootstrap import make_bootstrap
from .datasets.manifest import label_to_int, read_manifest, split_view
from .datasets.verify import dataset_report
from .distortions import available_distortions
from .eval import alignment as alignment_eval
from .eval import combine as combine_eval
from .eval import gradcheck as gradcheck_eval
from .eval import perceptual as perceptual_eval
from .eval import compare, eda, inverse, metrics, report, runner
from .eval.attack import attack_aggregate, defence_aggregate, run_attack, transfer_aggregate
from .eval.clustering import cluster_errors, cluster_profile
from .eval.error_table import build_error_table, file_features
from .eval.response import per_distortion_response, per_file_response
from .eval.sweep import aggregate_sweep, clean_scores_map, run_sweep
from .models.base import available_detectors, build_detector
from .repro.artifacts import RunDir
from .repro.checkpoint import load_partial
from .repro.environment import capture_environment, git_commit, render_environment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_models_config(path):
    return cfg.load_yaml(path) if Path(path).exists() else {}


def require_manifest(path):
    if path is None or not Path(path).exists():
        print(f"manifest not found: {path} (run scripts/prepare_login.sh first)")
        sys.exit(2)
    return read_manifest(path)


def make_detector(configuration, name, models_config):
    chosen = name or configuration["detector"]
    return build_detector(chosen, sample_rate=configuration["sample_rate"], **models_config.get(chosen, {}))


def cmd_bootstrap(args):
    path = make_bootstrap(args.out, n_per_class=args.n, sr=args.sample_rate, seed=args.seed)
    print(f"bootstrap manifest: {path}")


def cmd_list(args):
    print("detectors:", ", ".join(available_detectors()))
    print("distortions:", ", ".join(available_distortions()))
    print("attacks:", ", ".join(available_attacks()))


def cmd_manifest(args):
    builder = protocols.BUILDERS.get(args.dataset)
    if builder is None:
        print(f"unknown dataset '{args.dataset}'")
        sys.exit(2)
    manifest = builder(args.root, args.out, split=args.split, train_ratio=args.train_ratio)
    print(f"manifest: {args.out} ({len(manifest)} files)")


def cmd_subset(args):
    frame = read_manifest(args.manifest)
    rng = np.random.default_rng(args.seed)
    target = frame[frame["split"] == args.split]
    kept = []
    for _, group in target.groupby("label"):
        index = group.index.to_numpy().copy()
        rng.shuffle(index)
        kept.append(frame.loc[index[: args.n]])
    subset = pd.concat([frame[frame["split"] != args.split]] + kept, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.out, index=False)
    counts = subset[subset["split"] == args.split]["label"].value_counts().to_dict()
    print(f"subset: {args.out} ({args.split}={counts})")


def cmd_merge(args):
    train = read_manifest(args.train_from)
    evaluation = read_manifest(args.eval_from)
    train_rows = train[train["split"] == "train"].copy()
    if train_rows.empty:
        train_rows = train.copy()
        train_rows["split"] = "train"
    eval_rows = evaluation[evaluation["split"] == "eval"].copy()
    if eval_rows.empty:
        eval_rows = evaluation.copy()
        eval_rows["split"] = "eval"
    merged = pd.concat([train_rows, eval_rows], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"merged: {args.out} (train={len(train_rows)} from {train['dataset'].iloc[0]}, eval={len(eval_rows)} from {evaluation['dataset'].iloc[0]})")


def cmd_verify_dataset(args):
    report = dataset_report(read_manifest(args.manifest), min_duration=args.min_duration)
    for key, value in report.items():
        print(f"{key}: {value}")
    ok = report["missing_files"] == 0 and report["unreadable_files"] == 0
    for field, expected in (("total", args.expect_total), ("bona_fide", args.expect_bona), ("spoof", args.expect_spoof)):
        if expected is None:
            continue
        actual = report["total"] if field == "total" else report["by_label"].get(field, 0)
        status = "PASS" if actual == expected else "FAIL"
        ok = ok and actual == expected
        print(f"expect {field}={expected} actual={actual}: {status}")
    print("status:", "OK" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def cmd_verify_model(args):
    configuration = cfg.load_config(args.config)
    models_config = load_models_config(args.models)
    manifest = read_manifest(args.manifest or cfg.get(configuration, "dataset.manifest"))
    detector = make_detector(configuration, args.detector, models_config)
    print(f"detector: {detector.name} available: {detector.available()}")
    if not detector.available():
        sys.exit(2)
    detector.load()
    runner.fit_detector(detector, manifest)
    scores = runner.score_manifest(detector, manifest)
    evaluation = split_view(scores, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    report = metrics.summary(evaluation["score"].to_numpy(), targets, dcf=configuration.get("dcf"))
    print(f"metadata: {detector.metadata()}")
    for key in ("eer", "auc", "f1", "min_dcf", "cllr", "min_cllr", "n"):
        print(f"  {key}: {report[key]}")
    if report["eer"] > 0.5:
        print(f"WARN: eer>0.5 — scores likely inverted; check spoof_index (1-eer={round(1.0 - report['eer'], 4)})")
    if args.expected_eer is not None:
        ok = abs(report["eer"] - args.expected_eer) <= args.tolerance
        print(f"sanity expected_eer={args.expected_eer} tolerance={args.tolerance}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


def cmd_run(args):
    configuration = cfg.load_config(args.config)
    models_config = load_models_config(args.models)
    manifest_path = args.manifest or cfg.get(configuration, "dataset.manifest")
    manifest = require_manifest(manifest_path)
    detector = make_detector(configuration, args.detector, models_config)
    if not detector.available():
        print(f"detector '{detector.name}' is not available in this environment")
        sys.exit(2)
    detector.load()
    runner.fit_detector(detector, manifest)
    eval_dataset = split_view(manifest, "eval")["dataset"].iloc[0]
    run_dir = RunDir(configuration["paths"]["runs"], detector.name, eval_dataset, path=args.run_dir)
    scores = runner.score_manifest(detector, manifest, partial_path=str(run_dir.file("scores.partial.csv")))
    evaluation = split_view(scores, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    report = metrics.summary(evaluation["score"].to_numpy(), targets, dcf=configuration.get("dcf"))
    predictions = runner.predictions_from_scores(scores, report["threshold"])
    environment = capture_environment()

    run_dir.save_table("scores.csv", scores)
    run_dir.save_table("predictions.csv", predictions)
    run_dir.save_json("metrics.json", report)
    run_dir.save_yaml("config.yaml", {key: value for key, value in configuration.items() if not key.startswith("_")})
    run_dir.save_text("environment.txt", render_environment(environment))
    run_dir.save_json("meta.json", {
        "detector": detector.metadata(),
        "dataset": eval_dataset,
        "manifest": str(manifest_path),
        "commit": git_commit(PROJECT_ROOT),
        "run_date": environment["timestamp"],
        "n_eval": report["n"],
    })

    print(f"run: {run_dir.path}")
    for key in ("eer", "auc", "f1", "min_dcf"):
        print(f"  {key}: {report[key]:.4f}")
    expected = cfg.get(configuration, "sanity.expected_eer")
    if expected is not None:
        tolerance = cfg.get(configuration, "sanity.tolerance", 0.05)
        status = "PASS" if abs(report["eer"] - expected) <= tolerance else "FAIL"
        print(f"  sanity expected_eer={expected} tolerance={tolerance}: {status}")


def cmd_sweep(args):
    configuration = cfg.load_config(args.config)
    distortions = cfg.load_distortions(args.distortions)
    if args.only:
        wanted = set(args.only)
        distortions = {name: spec for name, spec in distortions.items() if name in wanted}
    models_config = load_models_config(args.models)
    sample_rate = configuration["sample_rate"]
    manifest_path = args.manifest or cfg.get(configuration, "dataset.manifest")
    manifest = require_manifest(manifest_path)
    evaluation = split_view(manifest, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    features = file_features(evaluation, sample_rate)

    eval_dataset = evaluation["dataset"].iloc[0]
    names = args.detector or [configuration["detector"]]
    tag = "multi" if len(names) > 1 else names[0]
    run_dir = RunDir(configuration["paths"]["runs"], tag, f"{eval_dataset}_sweep", path=args.run_dir)
    sweeps, aggregates, error_tables, used = [], [], [], []
    clean_reports = {}
    for name in names:
        detector = make_detector(configuration, name, models_config)
        if not detector.available():
            print(f"skip '{name}': not available in this environment")
            continue
        detector.load()
        runner.fit_detector(detector, manifest)
        clean = clean_scores_map(detector, evaluation, partial_path=str(run_dir.file(f"clean.partial.{detector.name}.csv")))
        clean_report = metrics.summary(evaluation["file_id"].map(clean).to_numpy(), targets, dcf=configuration.get("dcf"))
        clean_reports[detector.name] = clean_report
        sweep = run_sweep(detector, evaluation, distortions, clean, clean_report["threshold"], sample_rate, partial_path=str(run_dir.file(f"sweep.partial.{detector.name}.csv")))
        sweep["model_name"] = detector.name
        aggregate = aggregate_sweep(sweep, clean_report["eer"], dcf=configuration.get("dcf"), n_boot=cfg.get(configuration, "bootstrap.n_boot", 1000), seed=configuration.get("seed", 1337))
        aggregate["model_name"] = detector.name
        sweeps.append(sweep)
        aggregates.append(aggregate)
        error_tables.append(build_error_table(sweep, features, detector.metadata()))
        used.append(detector.metadata())

    if not sweeps:
        print("no available detectors")
        sys.exit(2)

    sweep_all = pd.concat(sweeps, ignore_index=True)
    aggregate_all = pd.concat(aggregates, ignore_index=True)
    error_all = pd.concat(error_tables, ignore_index=True)
    per_file = per_file_response(sweep_all)
    per_distortion = per_distortion_response(per_file)
    clustered = cluster_errors(error_all, n_clusters=args.clusters, seed=configuration.get("seed", 1337))
    profile = cluster_profile(clustered)
    distortion_summary = compare.model_distortion_summary(aggregate_all)
    distortion_pivot = compare.model_distortion_pivot(aggregate_all)
    cluster_table, cluster_pivot = compare.model_cluster_error(clustered)
    environment = capture_environment()

    run_dir.save_json("clean_metrics.json", clean_reports)
    run_dir.save_table("sweep.csv", sweep_all)
    run_dir.save_table("aggregate.csv", aggregate_all)
    run_dir.save_table("response_per_file.csv", per_file)
    run_dir.save_table("response_per_distortion.csv", per_distortion)
    run_dir.save_table("error_table.csv", error_all)
    run_dir.save_table("clusters.csv", clustered)
    run_dir.save_table("cluster_profile.csv", profile)
    run_dir.save_table("model_distortion_summary.csv", distortion_summary)
    run_dir.save_table("model_distortion_pivot.csv", distortion_pivot)
    run_dir.save_table("model_cluster_error.csv", cluster_table)
    run_dir.save_table("model_cluster_pivot.csv", cluster_pivot)
    run_dir.save_yaml("config.yaml", {key: value for key, value in configuration.items() if not key.startswith("_")})
    run_dir.save_text("environment.txt", render_environment(environment))
    run_dir.save_json("meta.json", {
        "detectors": used,
        "dataset": eval_dataset,
        "manifest": str(manifest_path),
        "distortions": list(distortions),
        "commit": git_commit(PROJECT_ROOT),
        "run_date": environment["timestamp"],
        "clean_eer": {name: report["eer"] for name, report in clean_reports.items()},
    })

    print(f"run: {run_dir.path}")
    print("  models:", ", ".join(f"{name}(clean eer={report['eer']:.3f})" for name, report in clean_reports.items()))
    print("  worst degradation per model (model distortion level dEER):")
    for model, group in aggregate_all.groupby("model_name"):
        worst = group.sort_values("delta_eer", ascending=False).iloc[0]
        print(f"    {model:24} {worst.distortion_type:14} L={worst.level:<6g} dEER={worst.delta_eer:+.3f}")
    print("  model x cluster error rate:")
    print(cluster_pivot.to_string(index=False))


def cmd_attack(args):
    configuration = cfg.load_config(args.config)
    attacks_config = cfg.load_yaml(args.attacks)
    if args.only:
        wanted = set(args.only)
        attacks_config = {name: spec for name, spec in attacks_config.items() if name in wanted}
    models_config = load_models_config(args.models)
    sample_rate = configuration["sample_rate"]
    manifest_path = args.manifest or cfg.get(configuration, "dataset.manifest")
    manifest = require_manifest(manifest_path)
    detector = make_detector(configuration, args.detector, models_config)
    if not detector.available():
        print(f"detector '{detector.name}' is not available in this environment")
        sys.exit(2)
    if not getattr(detector, "differentiable", False):
        print(f"detector '{detector.name}' is not differentiable; gradient attacks need logits_tensor")
        sys.exit(2)
    detector.load()
    runner.fit_detector(detector, manifest)
    evaluation = split_view(manifest, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    eval_dataset = evaluation["dataset"].iloc[0]
    run_dir = RunDir(configuration["paths"]["runs"], detector.name, f"{eval_dataset}_attack", path=args.run_dir)
    clean = clean_scores_map(detector, evaluation, partial_path=str(run_dir.file("clean.partial.csv")))
    clean_report = metrics.summary(evaluation["file_id"].map(clean).to_numpy(), targets, dcf=configuration.get("dcf"))
    threshold = clean_report["threshold"]

    transfer_detectors, transfer_clean_eer = [], {}
    for name in (args.transfer or []):
        other = make_detector(configuration, name, models_config)
        if not other.available():
            print(f"skip transfer '{name}': not available")
            continue
        other.load()
        runner.fit_detector(other, manifest)
        other_clean = clean_scores_map(other, evaluation, partial_path=str(run_dir.file(f"clean.transfer.{other.name}.csv")))
        transfer_clean_eer[other.name] = metrics.summary(evaluation["file_id"].map(other_clean).to_numpy(), targets)["eer"]
        transfer_detectors.append(other)

    defences_config = cfg.load_yaml(args.defences) if args.defences else {}
    if defences_config:
        print(f"defences: {', '.join(defences_config)}")
    n_boot = cfg.get(configuration, "bootstrap.n_boot", 1000)
    tables, surrogates = [], {}
    for name, spec in attacks_config.items():
        options = dict(spec.get("options") or {})
        if isinstance(options.get("surrogate"), str):
            key = options["surrogate"]
            if key not in surrogates:
                other = make_detector(configuration, key, models_config)
                other.load()
                runner.fit_detector(other, manifest)
                surrogates[key] = other
                print(f"surrogate: {other.name}")
            options["surrogate"] = surrogates[key]
        table = run_attack(detector, evaluation, spec["attack"], spec["levels"], clean, threshold, options, sample_rate, partial_path=str(run_dir.file(f"attack.partial.{name}.csv")), budget=args.budget, transfer=transfer_detectors, defences=defences_config)
        table["model_name"] = detector.name
        tables.append(table)
    attack_all = pd.concat(tables, ignore_index=True)
    aggregate = attack_aggregate(attack_all, clean_report["eer"], dcf=configuration.get("dcf"), n_boot=n_boot, seed=configuration.get("seed", 1337))
    aggregate["model_name"] = detector.name
    features = file_features(evaluation, sample_rate)
    error_table = build_error_table(attack_all, features, detector.metadata())
    environment = capture_environment()

    run_dir.save_json("clean_metrics.json", clean_report)
    run_dir.save_table("attack_table.csv", attack_all)
    run_dir.save_table("attack_aggregate.csv", aggregate)
    if transfer_detectors:
        run_dir.save_table("transfer_aggregate.csv", transfer_aggregate(attack_all, [other.name for other in transfer_detectors], transfer_clean_eer))
    if defences_config:
        defence_table = defence_aggregate(attack_all, list(defences_config), clean_report["eer"])
        defence_table["model_name"] = detector.name
        run_dir.save_table("defence_aggregate.csv", defence_table)
    run_dir.save_table("error_table.csv", error_table)
    run_dir.save_yaml("config.yaml", {key: value for key, value in configuration.items() if not key.startswith("_")})
    run_dir.save_text("environment.txt", render_environment(environment))
    run_dir.save_json("meta.json", {
        "detector": detector.metadata(),
        "dataset": eval_dataset,
        "manifest": str(manifest_path),
        "attacks": list(attacks_config),
        "budget": args.budget,
        "transfer": [other.name for other in transfer_detectors],
        "surrogates": sorted(surrogates),
        "commit": git_commit(PROJECT_ROOT),
        "run_date": environment["timestamp"],
        "clean_eer": clean_report["eer"],
    })

    print(f"run: {run_dir.path}")
    print(f"  clean eer={clean_report['eer']:.3f} cllr={clean_report['cllr']:.3f} min_cllr={clean_report['min_cllr']:.3f}")
    unit = "snr" if args.budget == "snr" else "eps"
    print(f"  attack (attack {unit} eer dEER ASR):")
    for row in aggregate.sort_values(["distortion_type", "severity"]).itertuples():
        print(f"    {row.distortion_type:6} {unit}={row.level:<7g} eer={row.eer:.3f} dEER={row.delta_eer:+.3f} ASR={row.attack_success_rate:.2f}")


def existing_tables(paths):
    return [path for path in (paths or []) if Path(path).exists()]


def cmd_inverse(args):
    present = existing_tables(args.error_table)
    if not present:
        print("inverse: no error tables present, skipping")
        return
    tables = [read_manifest(path) for path in present]
    error_table = pd.concat(tables, ignore_index=True)
    matrix, target, names = inverse.build_dataset(error_table)
    model, auc = inverse.fit_inverse(matrix, target, seed=args.seed)
    importance = inverse.feature_importance(model, names)
    by_distortion = inverse.failure_by_distortion(error_table)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out / "inverse_importance.csv", index=False)
    by_distortion.to_csv(out / "failure_by_distortion.csv", index=False)
    (out / "inverse_report.json").write_text(json.dumps({
        "predict_error_auc": auc,
        "n": int(len(target)),
        "error_rate": float(target.mean()),
        "top_features": importance.head(8).to_dict("records"),
    }, ensure_ascii=False, indent=2))

    print(f"inverse: predict-error AUC = {auc:.3f} over n={len(target)} (error_rate={target.mean():.3f})")
    print("top predictive features (sample/noise params -> detector error):")
    print(importance.head(8).to_string(index=False))
    print(f"saved: {out}")


def cmd_combine(args):
    present = existing_tables(args.error_table)
    if not present:
        print("combine: no error tables present, skipping")
        return
    results = combine_eval.combine(present, n_clusters=args.clusters, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    print(f"combined {len(present)} tables -> {out}")
    print("clean EER by model:")
    print(results["clean_eer_by_model"].to_string(index=False))
    print("worst ΔEER (distortion × model):")
    print(results["model_distortion_pivot"].to_string(index=False))
    print("predict-error AUC by model (inverse):")
    print(results["inverse_by_model"].to_string(index=False))
    print("error rate by cluster × model:")
    print(results["model_cluster_pivot"].to_string(index=False))


def cmd_eda(args):
    present = existing_tables(args.error_table)
    if not present:
        print("eda: no error tables present, skipping")
        return
    table = pd.concat([read_manifest(path) for path in present], ignore_index=True)
    summary = eda.feature_summary(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "feature_summary.csv", index=False)
    print(f"eda -> {out}")
    for model, group in summary.groupby("model_name"):
        print(f"\n{model} — чем 'ошибочные' семплы отличаются от верных (по убыванию различия):")
        print(group.head(5).to_string(index=False))


def _pipeline_run_dirs(configuration):
    sweep_dirs, attack_dirs = [], []
    for stage in configuration["stages"]:
        run_dir = (stage.get("params") or {}).get("run_dir")
        if not run_dir:
            continue
        if stage["type"] == "sweep":
            sweep_dirs.append(run_dir)
        elif stage["type"] == "attack":
            attack_dirs.append(run_dir)
    return sweep_dirs, attack_dirs


def cmd_report(args):
    configuration = cfg.load_yaml(args.pipeline)
    sweep_dirs, attack_dirs = _pipeline_run_dirs(configuration)
    results = report.build_report(sweep_dirs, attack_dirs, args.out)
    print(f"report -> {args.out}")
    if not results["clean"].empty:
        print("\nclean quality by model/dataset:")
        print(results["clean"].round(4).to_string(index=False))
    if not results["dataset_shift"].empty:
        print("\ndataset shift (EER by domain):")
        print(results["dataset_shift"].round(4).to_string(index=False))
    if not results["worst"].empty:
        print("\nworst degradation (ΔEER) per model/distortion:")
        print(results["worst"].round(4).to_string(index=False))
    if results["plots"]:
        print("\nplots:", ", ".join(results["plots"]))


def _stage_rows_done(run_dir):
    total = 0
    directory = Path(run_dir)
    if not directory.exists():
        return 0
    for partial in directory.glob("*.partial.*.csv"):
        frame = load_partial(str(partial))
        if frame is not None and "file_id" in frame.columns:
            total = max(total, frame["file_id"].nunique())
    return total


def cmd_progress(args):
    configuration = cfg.load_yaml(args.pipeline)
    base = configuration.get("runs", "runs")
    status = report.read_json(Path(base) / "pipeline" / "status.json") or []
    done = {entry["name"]: entry for entry in status}
    print(f"pipeline: {args.pipeline}")
    running_marked = False
    for stage in configuration["stages"]:
        name = stage["name"]
        entry = done.get(name)
        if entry is not None:
            detail = f" ({entry['reason']})" if entry.get("reason") else ""
            seconds = f" {entry['seconds']}s" if entry.get("seconds") else ""
            print(f"  [{entry['state']:8}] {name}{seconds}{detail}")
            continue
        run_dir = (stage.get("params") or {}).get("run_dir")
        rows = _stage_rows_done(run_dir) if run_dir else 0
        if not running_marked:
            marker = f"  files scored: {rows}" if rows else ""
            print(f"  [running ] {name}{marker}")
            running_marked = True
        else:
            print(f"  [pending ] {name}")


def cmd_align(args):
    import gc

    configuration = cfg.load_config(args.config)
    models_config = load_models_config(args.models)
    sample_rate = configuration["sample_rate"]
    manifest = require_manifest(args.manifest)
    evaluation = alignment_eval.balanced_sample(split_view(manifest, "eval"), args.n, args.seed)
    eval_dataset = evaluation["dataset"].iloc[0]
    run_dir = RunDir(configuration["paths"]["runs"], "multi", f"{eval_dataset}_align", path=args.run_dir)
    print(f"align: {len(evaluation)} files from {eval_dataset}, detectors: {', '.join(args.detector)}")

    gradients, support, used = {}, {}, []
    for name in args.detector:
        detector = make_detector(configuration, name, models_config)
        if not detector.available() or not getattr(detector, "differentiable", False):
            print(f"skip '{name}': not available or not differentiable")
            continue
        detector.load()
        runner.fit_detector(detector, manifest)
        grads, sup = alignment_eval.input_gradients(detector, evaluation, sample_rate)
        gradients[detector.name] = grads
        support.update(sup)
        used.append(detector.name)
        print(f"  {detector.name}: {len(grads)} gradients")
        torch = detector.torch
        del detector
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary, per_file = alignment_eval.pair_alignment(gradients, support)
    run_dir.save_table("alignment_pairs.csv", summary)
    run_dir.save_table("alignment_per_file.csv", per_file)
    run_dir.save_json("meta.json", {
        "dataset": eval_dataset,
        "manifest": str(args.manifest),
        "detectors": used,
        "n": int(len(evaluation)),
        "seed": args.seed,
        "commit": git_commit(PROJECT_ROOT),
    })
    print(f"run: {run_dir.path}")
    if summary.empty:
        print("  fewer than two differentiable detectors, no pairs to compare")
        return
    print(summary.sort_values("sign_alignment_mean", ascending=False).head(12).to_string(index=False))


def cmd_gradcheck(args):
    import gc

    configuration = cfg.load_config(args.config)
    models_config = load_models_config(args.models)
    sample_rate = configuration["sample_rate"]
    manifest = require_manifest(args.manifest)
    evaluation = alignment_eval.balanced_sample(split_view(manifest, "eval"), args.n, args.seed)
    eval_dataset = evaluation["dataset"].iloc[0]
    run_dir = RunDir(configuration["paths"]["runs"], "multi", f"{eval_dataset}_gradcheck", path=args.run_dir)
    print(f"gradcheck: {len(evaluation)} files from {eval_dataset}, detectors: {', '.join(args.detector)}")

    files, paths, models = [], [], {}
    for name in args.detector:
        detector = make_detector(configuration, name, models_config)
        if not detector.available() or not getattr(detector, "differentiable", False):
            print(f"skip '{name}': not available or not differentiable")
            continue
        detector.load()
        runner.fit_detector(detector, manifest)
        mismatch = gradcheck_eval.checkpoint_mismatch(detector)
        per_file, trace = gradcheck_eval.gradient_check(
            detector, evaluation, sample_rate, fine_snr=args.fine_snr, attack_snr=args.attack_snr,
            directions=args.directions, trajectory_files=args.trajectory_files, trajectory_snr=args.trajectory_snr,
            trajectory_steps=args.trajectory_steps, seed=args.seed,
        )
        models[detector.name] = dict(mismatch, training_modules=gradcheck_eval.training_modules(detector))
        files.append(per_file)
        paths.append(trace)
        print(f"  {detector.name}: {len(per_file)} files, missing keys {mismatch['missing_keys']}, modules in train mode {models[detector.name]['training_modules']}")
        torch = detector.torch
        del detector
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_file = pd.concat(files, ignore_index=True) if files else pd.DataFrame()
    trace = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    summary = gradcheck_eval.gradient_summary(per_file, trace) if not per_file.empty else pd.DataFrame()
    run_dir.save_table("gradcheck_files.csv", per_file)
    run_dir.save_table("gradcheck_trajectories.csv", trace)
    run_dir.save_table("gradcheck_summary.csv", summary)
    run_dir.save_json("meta.json", {
        "dataset": eval_dataset,
        "manifest": str(args.manifest),
        "models": models,
        "n": int(len(evaluation)),
        "seed": args.seed,
        "fine_snr": args.fine_snr,
        "attack_snr": args.attack_snr,
        "trajectory_snr": args.trajectory_snr,
        "commit": git_commit(PROJECT_ROOT),
    })
    print(f"run: {run_dir.path}")
    if not summary.empty:
        print(summary.to_string(index=False))


def cmd_perceptual(args):
    import gc

    configuration = cfg.load_config(args.config)
    models_config = load_models_config(args.models)
    sample_rate = configuration["sample_rate"]
    manifest = require_manifest(args.manifest)
    attacks_config = cfg.load_yaml(args.attacks)
    evaluation = alignment_eval.balanced_sample(split_view(manifest, "eval"), args.n, args.seed)
    eval_dataset = evaluation["dataset"].iloc[0]
    run_dir = RunDir(configuration["paths"]["runs"], "multi", f"{eval_dataset}_perceptual", path=args.run_dir)
    print(f"perceptual: {len(evaluation)} files from {eval_dataset}, detectors: {', '.join(args.detector)}, attacks: {', '.join(attacks_config)}")

    tables, used = [], []
    for name in args.detector:
        detector = make_detector(configuration, name, models_config)
        if not detector.available() or not getattr(detector, "differentiable", False):
            print(f"skip '{name}': not available or not differentiable")
            continue
        detector.load()
        runner.fit_detector(detector, manifest)
        table = perceptual_eval.perceptual_rows(detector, evaluation, attacks_config, sample_rate, seed=args.seed,
                                                save_dir=run_dir.file("audio"), save_n=args.save_n)
        tables.append(table)
        used.append(detector.name)
        print(f"  {detector.name}: {len(table)} rows")
        torch = detector.torch
        del detector
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    summary = perceptual_eval.perceptual_summary(rows) if not rows.empty else pd.DataFrame()
    run_dir.save_table("perceptual_files.csv", rows)
    run_dir.save_table("perceptual_summary.csv", summary)
    run_dir.save_json("meta.json", {
        "dataset": eval_dataset,
        "manifest": str(args.manifest),
        "detectors": used,
        "attacks": attacks_config,
        "n": int(len(evaluation)),
        "seed": args.seed,
        "commit": git_commit(PROJECT_ROOT),
    })
    print(f"run: {run_dir.path}")
    if not summary.empty:
        print(summary.to_string(index=False))


def cmd_pipeline(args):
    configuration = cfg.load_yaml(args.pipeline)
    dispatch = {
        "manifest": cmd_manifest,
        "subset": cmd_subset,
        "merge": cmd_merge,
        "verify-dataset": cmd_verify_dataset,
        "run": cmd_run,
        "sweep": cmd_sweep,
        "attack": cmd_attack,
        "align": cmd_align,
        "gradcheck": cmd_gradcheck,
        "perceptual": cmd_perceptual,
        "combine": cmd_combine,
        "inverse": cmd_inverse,
        "eda": cmd_eda,
        "report": cmd_report,
    }
    base = configuration.get("runs", "runs")
    stages = configuration["stages"]
    if args.only:
        wanted = set(args.only)
        stages = [stage for stage in stages if stage["name"] in wanted]
    status = pipeline.run_stages(stages, dispatch, pipeline.stage_defaults(), base=base, force=args.force)
    print("\npipeline summary:")
    for entry in status:
        detail = f" ({entry['reason']})" if entry.get("reason") else ""
        print(f"  {entry['name']:30} {entry['type']:14} {entry['state']}{detail}")
    if any(entry["state"] == "failed" for entry in status):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(prog="adfd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--out", default="data/bootstrap")
    bootstrap.add_argument("--n", type=int, default=80)
    bootstrap.add_argument("--sample-rate", type=int, default=16000)
    bootstrap.add_argument("--seed", type=int, default=1337)
    bootstrap.set_defaults(func=cmd_bootstrap)

    listing = subparsers.add_parser("list")
    listing.set_defaults(func=cmd_list)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--dataset", required=True, choices=sorted(protocols.BUILDERS))
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--split", default="eval")
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--train-ratio", type=float, default=0.0)
    manifest.set_defaults(func=cmd_manifest)

    subset = subparsers.add_parser("subset")
    subset.add_argument("--manifest", required=True)
    subset.add_argument("--out", required=True)
    subset.add_argument("--n", type=int, default=800)
    subset.add_argument("--split", default="eval")
    subset.add_argument("--seed", type=int, default=1337)
    subset.set_defaults(func=cmd_subset)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--train-from", required=True)
    merge.add_argument("--eval-from", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge)

    verify_dataset = subparsers.add_parser("verify-dataset")
    verify_dataset.add_argument("--manifest", required=True)
    verify_dataset.add_argument("--min-duration", type=float, default=0.3)
    verify_dataset.add_argument("--expect-total", type=int, default=None)
    verify_dataset.add_argument("--expect-bona", type=int, default=None)
    verify_dataset.add_argument("--expect-spoof", type=int, default=None)
    verify_dataset.set_defaults(func=cmd_verify_dataset)

    verify_model = subparsers.add_parser("verify-model")
    verify_model.add_argument("--config", default="configs/default.yaml")
    verify_model.add_argument("--models", default="configs/models.yaml")
    verify_model.add_argument("--manifest", default=None)
    verify_model.add_argument("--detector", default=None)
    verify_model.add_argument("--expected-eer", type=float, default=None)
    verify_model.add_argument("--tolerance", type=float, default=0.05)
    verify_model.set_defaults(func=cmd_verify_model)

    run = subparsers.add_parser("run")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--models", default="configs/models.yaml")
    run.add_argument("--manifest", default=None)
    run.add_argument("--detector", default=None)
    run.add_argument("--run-dir", default=None)
    run.set_defaults(func=cmd_run)

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--config", default="configs/default.yaml")
    sweep.add_argument("--models", default="configs/models.yaml")
    sweep.add_argument("--distortions", default="configs/distortions.yaml")
    sweep.add_argument("--manifest", default=None)
    sweep.add_argument("--detector", nargs="+", default=None)
    sweep.add_argument("--only", nargs="*", default=None)
    sweep.add_argument("--clusters", type=int, default=6)
    sweep.add_argument("--run-dir", default=None)
    sweep.set_defaults(func=cmd_sweep)

    attack = subparsers.add_parser("attack")
    attack.add_argument("--config", default="configs/default.yaml")
    attack.add_argument("--models", default="configs/models.yaml")
    attack.add_argument("--attacks", default="configs/attacks.yaml")
    attack.add_argument("--manifest", default=None)
    attack.add_argument("--detector", default="torch_reference")
    attack.add_argument("--only", nargs="*", default=None)
    attack.add_argument("--budget", default="snr", choices=["snr", "epsilon"])
    attack.add_argument("--transfer", nargs="*", default=None)
    attack.add_argument("--defences", default=None)
    attack.add_argument("--run-dir", default=None)
    attack.set_defaults(func=cmd_attack)

    inverse_p = subparsers.add_parser("inverse")
    inverse_p.add_argument("--error-table", nargs="+", required=True)
    inverse_p.add_argument("--out", default="runs/inverse")
    inverse_p.add_argument("--seed", type=int, default=1337)
    inverse_p.set_defaults(func=cmd_inverse)

    combine_p = subparsers.add_parser("combine")
    combine_p.add_argument("--error-table", nargs="+", required=True)
    combine_p.add_argument("--out", default="runs/combined")
    combine_p.add_argument("--clusters", type=int, default=6)
    combine_p.add_argument("--seed", type=int, default=1337)
    combine_p.set_defaults(func=cmd_combine)

    eda_p = subparsers.add_parser("eda")
    eda_p.add_argument("--error-table", nargs="+", required=True)
    eda_p.add_argument("--out", default="runs/eda")
    eda_p.set_defaults(func=cmd_eda)

    align_p = subparsers.add_parser("align")
    align_p.add_argument("--config", default="configs/default.yaml")
    align_p.add_argument("--models", default="configs/models.yaml")
    align_p.add_argument("--manifest", required=True)
    align_p.add_argument("--detector", nargs="+", required=True)
    align_p.add_argument("--n", type=int, default=300)
    align_p.add_argument("--seed", type=int, default=1337)
    align_p.add_argument("--run-dir", default=None)
    align_p.set_defaults(func=cmd_align)

    gradcheck_p = subparsers.add_parser("gradcheck")
    gradcheck_p.add_argument("--config", default="configs/default.yaml")
    gradcheck_p.add_argument("--models", default="configs/models.yaml")
    gradcheck_p.add_argument("--manifest", required=True)
    gradcheck_p.add_argument("--detector", nargs="+", required=True)
    gradcheck_p.add_argument("--n", type=int, default=40)
    gradcheck_p.add_argument("--seed", type=int, default=1337)
    gradcheck_p.add_argument("--fine-snr", type=float, default=80.0)
    gradcheck_p.add_argument("--attack-snr", type=float, default=50.0)
    gradcheck_p.add_argument("--directions", type=int, default=8)
    gradcheck_p.add_argument("--trajectory-files", type=int, default=10)
    gradcheck_p.add_argument("--trajectory-snr", type=float, default=45.0)
    gradcheck_p.add_argument("--trajectory-steps", type=int, default=100)
    gradcheck_p.add_argument("--run-dir", default=None)
    gradcheck_p.set_defaults(func=cmd_gradcheck)

    perceptual_p = subparsers.add_parser("perceptual")
    perceptual_p.add_argument("--config", default="configs/default.yaml")
    perceptual_p.add_argument("--models", default="configs/models.yaml")
    perceptual_p.add_argument("--manifest", required=True)
    perceptual_p.add_argument("--detector", nargs="+", required=True)
    perceptual_p.add_argument("--attacks", required=True)
    perceptual_p.add_argument("--n", type=int, default=40)
    perceptual_p.add_argument("--seed", type=int, default=1337)
    perceptual_p.add_argument("--save-n", type=int, default=4)
    perceptual_p.add_argument("--run-dir", default=None)
    perceptual_p.set_defaults(func=cmd_perceptual)

    pipeline_p = subparsers.add_parser("pipeline")
    pipeline_p.add_argument("--pipeline", default="configs/pipeline.yaml")
    pipeline_p.add_argument("--only", nargs="*", default=None)
    pipeline_p.add_argument("--force", action="store_true")
    pipeline_p.set_defaults(func=cmd_pipeline)

    report_p = subparsers.add_parser("report")
    report_p.add_argument("--pipeline", default="configs/pipeline.yaml")
    report_p.add_argument("--out", default="runs/pipeline/report")
    report_p.set_defaults(func=cmd_report)

    progress_p = subparsers.add_parser("progress")
    progress_p.add_argument("--pipeline", default="configs/pipeline.yaml")
    progress_p.set_defaults(func=cmd_progress)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
