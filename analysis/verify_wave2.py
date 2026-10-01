import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/wave2")
parser.add_argument("--permutations", type=int, default=5000)
args = parser.parse_args()

XLSR = {"ssl_aasist", "xlsr_mamba", "xlsr_linear", "nes2net_v2"}
WAVLM = {"nes2net", "wavlm_linear"}
STRONG = ["pgd_restarts", "pgd_margin", "pgd_warm"]

transfer = pd.read_csv(f"{args.results}/transfer.csv")
attack = pd.read_csv(f"{args.results}/attack.csv")
alignment = pd.read_csv(f"{args.results}/alignment.csv")
alignment = alignment[alignment.run_group == "runs_align"]
perceptual = pd.read_csv(f"{args.results}/perceptual.csv")
asv5 = pd.read_csv(f"{args.results}/asv5_per_attack.csv")


def shared(source, victim):
    return int((source in XLSR and victim in XLSR) or (source in WAVLM and victim in WAVLM))


def matrix_rows():
    asv19 = transfer[(transfer.run_group == "runs") & transfer.run.str.endswith("_asv19_attack_matrix") & (transfer.run != "spectra_aasist3_asv19_attack_matrix")]
    spectra = transfer[(transfer.run_group == "runs_spectrafix") & (transfer.run == "spectra_asv19_attack_matrix")]
    fine = transfer[(transfer.run_group == "runs_fine") & transfer.run.str.endswith("_attack")]
    frames = []
    for domain, frame in (("asv19", pd.concat([asv19, spectra])), ("itw", fine[fine.run.str.endswith("_itw_attack")]), ("asv21", fine[fine.run.str.endswith("_asv21_attack")])):
        frame = frame[(frame.distortion_type == "pgd") & (frame.level == 20.0)].copy()
        frame["domain"] = domain
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.rename(columns={"source_model": "source", "transfer_model": "victim"})
    rows["shared"] = [shared(s, v) for s, v in zip(rows.source, rows.victim)]
    rows["victim_clean"] = (rows.eer - rows.delta_eer).groupby([rows.domain, rows.victim]).transform("median")
    return rows[["domain", "source", "victim", "shared", "delta_eer", "victim_clean"]]


def ols(frame, columns, effects):
    design = [np.ones(len(frame))] + [frame[c].to_numpy(float) for c in columns]
    for effect in effects:
        for level in sorted(frame[effect].unique())[1:]:
            design.append((frame[effect] == level).to_numpy(float))
    design = np.column_stack(design)
    target = frame.delta_eer.to_numpy(float)
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ beta
    r2 = 1.0 - (residual ** 2).sum() / ((target - target.mean()) ** 2).sum()
    return beta[1:1 + len(columns)], r2


def permutation(frame, groups, columns, effects, n, seed=0):
    rng = np.random.default_rng(seed)
    frame = frame.reset_index(drop=True)
    observed = ols(frame, columns, effects)[0][0]
    blocks = [group.index.to_numpy() for _, group in frame.groupby(groups)]
    exceed = 0
    for _ in range(n):
        labels = frame.shared.to_numpy().copy()
        for block in blocks:
            labels[block] = rng.permutation(labels[block])
        if ols(frame.assign(shared=labels), columns, effects)[0][0] >= observed:
            exceed += 1
    return observed, (exceed + 1) / (n + 1)


rows = matrix_rows()
print("H4: transfer of PGD at 20 dB, shared vs different pretrained encoder")
for domain, frame in rows.groupby("domain", sort=False):
    wins, total = 0, 0
    for source, group in frame.groupby("source"):
        if group.shared.sum() == 0:
            continue
        total += 1
        wins += int(group[group.shared == 1].delta_eer.median() > group[group.shared == 0].delta_eer.median())
    effect, p = permutation(frame, ["source"], ["shared", "victim_clean"], ["source"], args.permutations)
    print(f"  {domain}: pairs {len(frame)} (shared {int(frame.shared.sum())}), median {frame[frame.shared == 1].delta_eer.median():.3f} vs {frame[frame.shared == 0].delta_eer.median():.3f}, "
          f"sources {wins}/{total} (sign p = {binomtest(wins, total, 0.5, alternative='greater').pvalue:.4f}), adjusted effect {effect:+.3f} (permutation p = {p:.1e})")
effect, p = permutation(rows, ["domain", "source"], ["shared", "victim_clean"], ["domain", "source"], args.permutations)
print(f"  pooled: pairs {len(rows)}, adjusted effect {effect:+.3f} (permutation p = {p:.1e})")

merged = rows.merge(alignment.assign(
    domain=lambda frame: frame.dataset.map({"asvspoof2019_la": "asv19", "in_the_wild": "itw", "asvspoof2021_la": "asv21"}))[["domain", "source", "victim", "sign_alignment_mean", "cosine_mean"]],
    on=["domain", "source", "victim"], how="left")
print("\nmechanism: Spearman rho of sign alignment with transfer")
for domain, frame in merged.groupby("domain", sort=False):
    print(f"  {domain}: n {len(frame)}, sign {spearmanr(frame.sign_alignment_mean, frame.delta_eer)[0]:+.2f}, cosine {spearmanr(frame.cosine_mean, frame.delta_eer)[0]:+.2f}")
print(f"  pooled: n {len(merged)}, sign {spearmanr(merged.sign_alignment_mean, merged.delta_eer)[0]:+.2f}")
base = ols(merged, ["shared", "victim_clean"], ["domain", "source"])[0][0]
mediated = ols(merged, ["shared", "victim_clean", "sign_alignment_mean"], ["domain", "source"])[0][0]
print(f"  shared-encoder effect {base:+.3f} -> {mediated:+.3f} with alignment ({100 * (1 - mediated / base):.0f}% explained)")

strength = attack[attack.run_group.isin(["runs_strength", "runs_diag"]) & attack.run.str.contains("_asv19_")]
strength = strength.assign(model=strength.run.str.replace(r"_asv19_.*", "", regex=True))
print("\nH7: attack success rate, PGD-10 vs strongest attack (max of 100x3 CE, 100x3 margin, surrogate warm start)")
for model, frame in strength.groupby("model"):
    weak = frame[frame.distortion_type == "pgd"].groupby("level").attack_success_rate.max()
    strong = frame[frame.distortion_type.isin(STRONG)].groupby("level").attack_success_rate.max()
    levels = sorted(set(weak.index) | set(strong.index))
    cells = ", ".join(f"{level:g} dB {weak.get(level, np.nan):.3f}/{strong.get(level, np.nan):.3f}" for level in levels)
    print(f"  {model}: {cells}")

print("\nmasking checks at 50/45 dB: best direct / surrogate transfer / surrogate warm start")
for model, frame in strength.groupby("model"):
    if "pgd_warm" not in set(frame.distortion_type):
        continue
    direct = frame[frame.distortion_type.isin(["pgd", "pgd_restarts", "pgd_margin"])].groupby("level").attack_success_rate.max()
    moved = frame[frame.distortion_type == "surrogate_transfer"].groupby("level").attack_success_rate.max()
    warm = frame[frame.distortion_type == "pgd_warm"].groupby("level").attack_success_rate.max()
    cells = ", ".join(f"{level:g} dB {direct.get(level, np.nan):.3f}/{moved.get(level, np.nan):.3f}/{warm.get(level, np.nan):.3f}" for level in (50.0, 45.0))
    print(f"  {model}: {cells}")

print("\nperceptual quality (median PESQ-WB attack / matched-energy noise)")
table = perceptual.pivot_table(index="level", columns="kind", values="pesq_wb", aggfunc=["min", "max"]).sort_index(ascending=False)
for level, row in table.iterrows():
    print(f"  {level:g} dB: attack {row[('min', 'attack')]:.2f}-{row[('max', 'attack')]:.2f}, noise {row[('min', 'noise')]:.2f}-{row[('max', 'noise')]:.2f}")

seeds = attack[attack.run_group == "runs_seeds"]
if not seeds.empty:
    print("\nprobe seeds: success rate at the threshold boundary (PGD-100x3)")
    boundary = seeds[seeds.distortion_type == "pgd_restarts"]
    for (model, level), frame in boundary.groupby(["model_name", "level"]):
        print(f"  {model} {level:g} dB: {frame.attack_success_rate.iloc[0]:.3f}")

print("\nASVspoof 5: EER on A17 (TTS) and A18 (A17 + Malafide built on an AASIST surrogate)")
pairs = asv5[asv5.attack.isin(["A17", "A18"])].pivot_table(index="model_name", columns="attack", values="eer")
for model, row in pairs.iterrows():
    print(f"  {model}: {row['A17']:.3f} -> {row['A18']:.3f}")
