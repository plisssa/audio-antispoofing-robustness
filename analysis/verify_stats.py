import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--report", default="results/report")
args = parser.parse_args()

cs = pd.read_csv(f"{args.report}/clean_summary.csv")
dg = pd.read_csv(f"{args.report}/degradation_summary.csv")
at = pd.read_csv(f"{args.report}/attack_summary.csv")
tr = pd.read_csv(f"{args.report}/transfer_summary.csv")

piv = cs.pivot(index="model", columns="dataset", values="eer")
print("clean EER by model and domain")
print(piv.round(4).to_string())

rho, p = spearmanr(piv["asvspoof2019_la"], piv["in_the_wild"])
print(f"\nspearman rho (2019 LA vs ITW): {rho:.3f}, p = {p:.3f}, n = {len(piv)}")

dg19 = dg[dg["dataset"] == "asvspoof2019_la"]
at19 = at[at["dataset"] == "asvspoof2019_la"]
noise = dg19[dg19["distortion_type"] == "noise"][["model_name", "level", "delta_eer"]]
pgd = at19[at19["distortion_type"] == "pgd"][["model_name", "level", "delta_eer"]]
wins = 0
models = sorted(set(pgd["model_name"]))
for m in models:
    nm = noise[noise["model_name"] == m].set_index("level")["delta_eer"]
    pm = pgd[pgd["model_name"] == m].set_index("level")["delta_eer"]
    common = sorted(set(nm.index) & set(pm.index))
    greater = all(pm[lv] > nm[lv] for lv in common)
    wins += int(greater)
    print(f"{m}: matched SNR levels {common}, pgd > noise at all: {greater}")
res = binomtest(wins, len(models), 0.5, alternative="greater")
print(f"sign test: {wins}/{len(models)} models, one-sided p = {res.pvalue:.4f}")

print("\nPGD transfer from xlsr_mamba (delta EER of victim):")
tp = tr[(tr["distortion_type"] == "pgd") & (tr["attacked_model"] == "xlsr_mamba")]
for v in sorted(set(tp["transfer_model"])):
    s = tp[tp["transfer_model"] == v].sort_values("level", ascending=False)
    vals = ", ".join(f"{r.level:g} dB: {r.delta_eer:.3f}" for r in s.itertuples())
    print(f"  -> {v}: {vals}")

itw = cs[cs["dataset"] == "in_the_wild"]
print("\ncalibration on ITW (EER / Cllr):")
for r in itw.sort_values("eer").itertuples():
    print(f"  {r.model}: {r.eer:.3f} / {r.cllr:.3f}")
