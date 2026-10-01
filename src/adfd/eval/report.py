import json
from pathlib import Path

import pandas as pd


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def collect_sweeps(run_dirs):
    clean_rows, degradation = [], []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        meta = read_json(run_dir / "meta.json")
        clean = read_json(run_dir / "clean_metrics.json")
        aggregate = read_csv(run_dir / "aggregate.csv")
        if meta is None:
            continue
        dataset = meta.get("dataset", run_dir.name)
        if isinstance(clean, dict):
            for model, report in clean.items():
                clean_rows.append({
                    "model": model,
                    "dataset": dataset,
                    "eer": report.get("eer"),
                    "auc": report.get("auc"),
                    "cllr": report.get("cllr"),
                    "min_dcf": report.get("min_dcf"),
                    "n": report.get("n"),
                })
        if aggregate is not None and not aggregate.empty:
            frame = aggregate.copy()
            frame["dataset"] = dataset
            frame["run_dir"] = str(run_dir)
            degradation.append(frame)
    clean_frame = pd.DataFrame(clean_rows).drop_duplicates(["model", "dataset"]).reset_index(drop=True)
    degradation_frame = pd.concat(degradation, ignore_index=True) if degradation else pd.DataFrame()
    return clean_frame, degradation_frame


def collect_attacks(run_dirs):
    attacks, transfers = [], []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        meta = read_json(run_dir / "meta.json")
        aggregate = read_csv(run_dir / "attack_aggregate.csv")
        transfer = read_csv(run_dir / "transfer_aggregate.csv")
        if meta is None:
            continue
        dataset = meta.get("dataset", run_dir.name)
        attacked = meta.get("detector", {}).get("name", run_dir.name)
        if aggregate is not None and not aggregate.empty:
            frame = aggregate.copy()
            frame["dataset"] = dataset
            if "model_name" not in frame.columns:
                frame["model_name"] = attacked
            attacks.append(frame)
        if transfer is not None and not transfer.empty:
            frame = transfer.copy()
            frame["dataset"] = dataset
            frame["attacked_model"] = attacked
            transfers.append(frame)
    attack_frame = pd.concat(attacks, ignore_index=True) if attacks else pd.DataFrame()
    transfer_frame = pd.concat(transfers, ignore_index=True) if transfers else pd.DataFrame()
    return attack_frame, transfer_frame


def dataset_shift_table(clean_frame):
    if clean_frame.empty:
        return pd.DataFrame()
    return clean_frame.pivot_table(index="model", columns="dataset", values="eer").reset_index()


def worst_degradation(degradation_frame):
    if degradation_frame.empty:
        return pd.DataFrame()
    index = degradation_frame.groupby(["dataset", "model_name", "distortion_type"])["delta_eer"].idxmax()
    columns = ["dataset", "model_name", "distortion_type", "level", "severity", "eer", "delta_eer", "delta_eer_ci_low", "delta_eer_ci_high"]
    present = [column for column in columns if column in degradation_frame.columns]
    return degradation_frame.loc[index, present].sort_values(["dataset", "model_name", "delta_eer"], ascending=[True, True, False]).reset_index(drop=True)


def _plot_calibration(plt, out, clean_frame):
    figure, axis = plt.subplots(figsize=(7, 5))
    for dataset, group in clean_frame.groupby("dataset"):
        axis.scatter(group["eer"], group["cllr"], label=str(dataset), s=60)
        for _, row in group.iterrows():
            axis.annotate(str(row["model"]), (row["eer"], row["cllr"]), fontsize=7)
    axis.set_xlabel("EER (clean)")
    axis.set_ylabel("Cllr (calibration)")
    axis.set_title("Clean quality vs calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out / "clean_calibration.png", dpi=130)
    plt.close(figure)
    return ["clean_calibration.png"]


def _plot_degradation(plt, out, degradation_frame):
    made = []
    for dataset, group in degradation_frame.groupby("dataset"):
        worst = group.groupby(["model_name", "distortion_type"])["delta_eer"].max().reset_index()
        pivot = worst.pivot(index="distortion_type", columns="model_name", values="delta_eer")
        figure, axis = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel("worst ΔEER")
        axis.set_title(f"Degradation by distortion — {dataset}")
        figure.tight_layout()
        name = f"degradation_{dataset}.png"
        figure.savefig(out / name, dpi=130)
        plt.close(figure)
        made.append(name)
    return made


def _plot_attacks(plt, out, attack_frame):
    group_key = ["model_name", "distortion_type"] if "model_name" in attack_frame.columns else ["distortion_type"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for key, group in attack_frame.groupby(group_key):
        ordered = group.sort_values("level")
        axis.plot(ordered["level"], ordered["attack_success_rate"], marker="o", label=str(key))
    axis.set_xlabel("target SNR of perturbation (dB)")
    axis.set_ylabel("attack success rate")
    axis.set_title("Adversarial success vs perturbation SNR")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "attack_asr.png", dpi=130)
    plt.close(figure)
    return ["attack_asr.png"]


def _try_plots(out, clean_frame, degradation_frame, attack_frame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    made = []
    jobs = []
    if not clean_frame.empty:
        jobs.append(lambda: _plot_calibration(plt, out, clean_frame))
    if not degradation_frame.empty:
        jobs.append(lambda: _plot_degradation(plt, out, degradation_frame))
    if not attack_frame.empty and "attack_success_rate" in attack_frame.columns:
        jobs.append(lambda: _plot_attacks(plt, out, attack_frame))
    for job in jobs:
        try:
            made.extend(job())
        except Exception as error:
            print(f"report: plot skipped ({type(error).__name__}: {error})")
    return made


def _block(title, frame, columns=None):
    if frame is None or frame.empty:
        return []
    view = frame[[c for c in columns if c in frame.columns]] if columns else frame
    return [f"## {title}", "", "```", view.round(4).to_string(index=False), "```", ""]


def _markdown(clean_frame, shift, worst, attack_frame, transfer_frame, plots):
    lines = ["# Отчёт по устойчивости детекторов", ""]
    lines += _block("Чистое качество и калибровка", clean_frame)
    lines += _block("Сдвиг распределения (EER по доменам)", shift)
    lines += _block("Худшая деградация по искажениям (ΔEER)", worst)
    lines += _block("Adversarial-атаки (ASR по SNR возмущения)", attack_frame,
                    ["model_name", "distortion_type", "level", "eer", "delta_eer", "attack_success_rate", "cllr"])
    lines += _block("Перенос атак между моделями", transfer_frame)
    if plots:
        lines += ["## Графики", ""] + [f"- `{name}`" for name in plots] + [""]
    return "\n".join(lines)


def build_report(sweep_dirs, attack_dirs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clean_frame, degradation_frame = collect_sweeps(sweep_dirs)
    attack_frame, transfer_frame = collect_attacks(attack_dirs)
    shift = dataset_shift_table(clean_frame)
    worst = worst_degradation(degradation_frame)

    clean_frame.to_csv(out / "clean_summary.csv", index=False)
    degradation_frame.to_csv(out / "degradation_summary.csv", index=False)
    shift.to_csv(out / "dataset_shift.csv", index=False)
    worst.to_csv(out / "worst_degradation.csv", index=False)
    attack_frame.to_csv(out / "attack_summary.csv", index=False)
    transfer_frame.to_csv(out / "transfer_summary.csv", index=False)

    plots = _try_plots(out, clean_frame, degradation_frame, attack_frame)
    (out / "report.md").write_text(_markdown(clean_frame, shift, worst, attack_frame, transfer_frame, plots))
    return {
        "clean": clean_frame,
        "degradation": degradation_frame,
        "dataset_shift": shift,
        "worst": worst,
        "attacks": attack_frame,
        "transfers": transfer_frame,
        "plots": plots,
    }
