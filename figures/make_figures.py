import argparse
import os
from math import pi

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/report")
parser.add_argument("--out", default="figures/out")
args = parser.parse_args()

R = args.results
OUT = args.out
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 200, "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})

MODELS = ["AASIST", "AASIST3 (KAN)", "Nes2Net", "RawNet2", "reference",
          "Spectra-AASIST3", "SSL-AASIST", "XLSR-Mamba"]
COL = {"AASIST": "#0072B2", "AASIST3 (KAN)": "#D55E00", "Nes2Net": "#009E73",
       "RawNet2": "#CC79A7", "reference": "#666666", "Spectra-AASIST3": "#E69F00",
       "SSL-AASIST": "#56B4E9", "XLSR-Mamba": "#111111"}
KEY = {"AASIST": "aasist", "AASIST3 (KAN)": "aasist3_kan", "Nes2Net": "nes2net",
       "RawNet2": "rawnet2", "reference": "reference_lfcc_logreg",
       "Spectra-AASIST3": "spectra_aasist3", "SSL-AASIST": "ssl_aasist",
       "XLSR-Mamba": "xlsr_mamba"}
INV = {v: k for k, v in KEY.items()}
DOMS = ["asvspoof2019_la", "asvspoof2021_la", "in_the_wild"]

dg = pd.read_csv(f"{R}/degradation_summary.csv")
wd = pd.read_csv(f"{R}/worst_degradation.csv")
tr = pd.read_csv(f"{R}/transfer_summary.csv")
cs = pd.read_csv(f"{R}/clean_summary.csv")
at = pd.read_csv(f"{R}/attack_summary.csv")
dg19 = dg[dg["dataset"] == "asvspoof2019_la"]
wd19 = wd[wd["dataset"] == "asvspoof2019_la"]

def clean_eer(m, d):
    s = cs[(cs["model"] == KEY[m]) & (cs["dataset"] == d)]
    return float(s["eer"].iloc[0]) if len(s) else np.nan

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

L = {
 "ru": dict(
    doms=["ASVspoof\n2019 LA", "ASVspoof\n2021 LA", "In-the-Wild"], eer="EER",
    dsh="Сдвиг домена: EER детекторов по трём корпусам",
    snr="Бюджет возмущения, дБ (SNR)", asr="Доля успеха атаки (ASR)",
    dose="Доза–эффект атак: ASR по бюджету SNR (ASVspoof 2019 LA)",
    deer="ΔEER", vdeer="Приращение ошибки жертвы, ΔEER",
    trf="Перенос PGD: общий фронтенд против иного (ASVspoof 2019 LA)",
    shared="XLSR-Mamba → SSL-AASIST (общий XLS-R)", diff="XLSR-Mamba → AASIST (иной вход)",
    cal="Разъединение различения и калибровки (In-the-Wild)",
    cllr="Cllr (калибровка, лог. шкала)", broken="Cllr = 1 (калибровка непригодна выше)",
    hm="Худшее ΔEER по искажениям (ASVspoof 2019 LA)",
    br="Битрейт EnCodec, кбит/с", enc="Немонотонный отклик на EnCodec (ASVspoof 2019 LA)",
    sev="сила воздействия →",
    sm="Малые кратные: доза–эффект сигнальных искажений (ASVspoof 2019 LA)",
    rs="Реальные против синтетических искажений (ΔEER, In-the-Wild)",
    real="реальное", syn="синтетика",
    tm="Матрица переноса PGD при 20 дБ: ΔEER жертвы (ASVspoof 2019 LA)",
    src="источник (атакован)", vic="жертва",
    bump="Ранги моделей по EER: перестановка при смене домена", rank="ранг (1 — лучший)",
    fr="Форест-график: худшее ΔEER по модели с 95 % ДИ (ASVspoof 2019 LA)",
    rad="Профили устойчивости моделей по семействам искажений (ASVspoof 2019 LA)",
    db="Чистое качество против худшего искажения (EER, ASVspoof 2019 LA)",
    dbc="чистый EER", dbw="EER при худшем искажении",
    cov="Карта покрытия измерений: модель × воздействие × домен",
    covy="есть измерение", covn="нет измерения",
    pipe="Дизайн эксперимента иллюстративной валидации",
    arch="Атакуемая пара с общим SSL-фронтендом",
    fams={"noise": "шум", "reverb": "реверб.", "chirp": "чирп", "bandpass": "полоса",
          "codec": "MP3", "gain": "усил.", "transcode_identity": "перекод.",
          "transmission": "потери", "neural_encodec": "EnCodec",
          "telephony_g711a": "G.711a", "telephony_g711u": "G.711u",
          "telephony_g722": "G.722", "real_bird": "птицы (реал.)",
          "real_noise": "шум (реал.)", "real_rir": "RIR (реал.)"},
    rsl={"birds": "птицы", "noise": "шум", "reverb": "реверб."}),
 "en": dict(
    doms=["ASVspoof\n2019 LA", "ASVspoof\n2021 LA", "In-the-Wild"], eer="EER",
    dsh="Domain shift: detector EER across three corpora",
    snr="Perturbation budget, dB (SNR)", asr="Attack success rate (ASR)",
    dose="Attack dose–response: ASR by SNR budget (ASVspoof 2019 LA)",
    deer="ΔEER", vdeer="Victim error increment, ΔEER",
    trf="PGD transfer: shared front-end vs different (ASVspoof 2019 LA)",
    shared="XLSR-Mamba → SSL-AASIST (shared XLS-R)", diff="XLSR-Mamba → AASIST (different input)",
    cal="Discrimination–calibration decoupling (In-the-Wild)",
    cllr="Cllr (calibration, log scale)", broken="Cllr = 1 (calibration unusable above)",
    hm="Worst ΔEER by distortion (ASVspoof 2019 LA)",
    br="EnCodec bitrate, kbps", enc="Non-monotone response to EnCodec (ASVspoof 2019 LA)",
    sev="severity →",
    sm="Small multiples: dose–response of signal distortions (ASVspoof 2019 LA)",
    rs="Real versus synthetic distortions (ΔEER, In-the-Wild)",
    real="real", syn="synthetic",
    tm="PGD transfer matrix at 20 dB: victim ΔEER (ASVspoof 2019 LA)",
    src="source (attacked)", vic="victim",
    bump="Model ranks by EER: reordering under domain shift", rank="rank (1 = best)",
    fr="Forest plot: worst ΔEER per model with 95% CI (ASVspoof 2019 LA)",
    rad="Model robustness profiles by distortion family (ASVspoof 2019 LA)",
    db="Clean quality versus worst distortion (EER, ASVspoof 2019 LA)",
    dbc="clean EER", dbw="EER under worst distortion",
    cov="Measurement coverage map: model × perturbation × domain",
    covy="measured", covn="not measured",
    pipe="Experimental design of the illustrative validation",
    arch="Attacked pair with a shared SSL front-end",
    fams={"noise": "noise", "reverb": "reverb", "chirp": "chirp", "bandpass": "bandpass",
          "codec": "MP3", "gain": "gain", "transcode_identity": "transcode",
          "transmission": "pkt loss", "neural_encodec": "EnCodec",
          "telephony_g711a": "G.711a", "telephony_g711u": "G.711u",
          "telephony_g722": "G.722", "real_bird": "birds (real)",
          "real_noise": "noise (real)", "real_rir": "RIR (real)"},
    rsl={"birds": "birds", "noise": "noise", "reverb": "reverb"}),
}

RS = {"Spectra-AASIST3": [("birds", 0.084, 0.015), ("noise", 0.104, 0.187), ("reverb", 0.230, 0.309)],
      "AASIST3 (KAN)": [("birds", 0.129, 0.063), ("noise", 0.143, 0.178), ("reverb", 0.115, 0.158)],
      "AASIST": [("birds", 0.154, 0.133), ("noise", 0.124, 0.145), ("reverb", 0.014, 0.131)]}

SNR = [40, 30, 20, 10]

PIPE = {
 "ru": dict(c1="Корпуса", c1b=["ASVspoof 2019 LA", "ASVspoof 2021 LA", "In-the-Wild"],
    c2="Оценочные подвыборки", c2b=["сбалансированные\nbona fide / spoof,\nфиксированные списки"],
    c3="Воздействия", c3b=["сигнальные: шум, реверб.,\nчирп, полоса, усиление",
        "канал: MP3, перекод.,\nпотери, G.711a/u, G.722,\nEnCodec",
        "реальные корпуса:\nптицы, шум, RIR",
        "атаки: FGSM, PGD\n(бюджет SNR 40–10 дБ)"],
    c4="Детекторы (8)", c4b=["классический: reference\n(LFCC + логрег)",
        "end-to-end: RawNet2,\nAASIST",
        "SSL: AASIST3, Spectra-\nAASIST3, SSL-AASIST,\nXLSR-Mamba, Nes2Net"],
    c5="Метрики", c5b=["EER, ΔEER (95 % ДИ,\nпарный бутстрэп)",
        "Cllr, minDCF", "ASR, перенос атак"]),
 "en": dict(c1="Corpora", c1b=["ASVspoof 2019 LA", "ASVspoof 2021 LA", "In-the-Wild"],
    c2="Evaluation subsets", c2b=["balanced\nbona fide / spoof,\nfixed lists"],
    c3="Perturbations", c3b=["signal: noise, reverb,\nchirp, bandpass, gain",
        "channel: MP3, transcode,\npacket loss, G.711a/u,\nG.722, EnCodec",
        "real corpora:\nbirds, noise, RIR",
        "attacks: FGSM, PGD\n(SNR budget 40–10 dB)"],
    c4="Detectors (8)", c4b=["classical: reference\n(LFCC + logreg)",
        "end-to-end: RawNet2,\nAASIST",
        "SSL: AASIST3, Spectra-\nAASIST3, SSL-AASIST,\nXLSR-Mamba, Nes2Net"],
    c5="Metrics", c5b=["EER, ΔEER (95% CI,\npaired bootstrap)",
        "Cllr, minDCF", "ASR, attack transfer"]),
}

ARCH = {
 "ru": dict(wav="входной сигнал x", pert="возмущение δ\n(PGD, бюджет SNR)",
    fe="Общий SSL-фронтенд\nXLS-R 300M", fe2="CNN-энкодер → 24 слоя Transformer",
    b1="Бэкенд AASIST\n(графовое внимание)", b2="Бэкенд Mamba\n(SSM)",
    s1="оценка\nSSL-AASIST", s2="оценка\nXLSR-Mamba",
    note="атака, посчитанная по одной ветви,\nпереносится через общий фронтенд\n(ΔEER жертвы до 0,94)"),
 "en": dict(wav="input signal x", pert="perturbation δ\n(PGD, SNR budget)",
    fe="Shared SSL front-end\nXLS-R 300M", fe2="CNN encoder → 24 Transformer layers",
    b1="AASIST back-end\n(graph attention)", b2="Mamba back-end\n(SSM)",
    s1="SSL-AASIST\nscore", s2="XLSR-Mamba\nscore",
    note="an attack computed through one branch\ntransfers via the shared front-end\n(victim ΔEER up to 0.94)"),
}

def fig_domainshift(t, lang):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = [0, 1, 2]
    for m in MODELS:
        ys = [clean_eer(m, d) for d in DOMS]
        ax.plot(x, ys, "-o", color=COL[m], lw=1.8, ms=5, mec="white", mew=0.6)
        ax.annotate(m, (2, ys[2]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=COL[m])
    ax.set_xticks(x)
    ax.set_xticklabels(t["doms"])
    ax.set_ylabel(t["eer"])
    ax.set_title(t["dsh"], fontsize=11.5)
    ax.set_xlim(-0.1, 2.85)
    ax.set_ylim(-0.02, 0.72)
    save(fig, f"fig_domainshift_{lang}")

def fig_attackdose(t, lang):
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    for ax, kind, ti in [(axs[0], "pgd", "PGD"), (axs[1], "fgsm", "FGSM")]:
        sub = at[(at["distortion_type"] == kind) & (at["dataset"] == "asvspoof2019_la")]
        for m in MODELS:
            s = sub[sub["model_name"] == KEY[m]].sort_values("level", ascending=False)
            if not len(s):
                continue
            ax.plot(s["level"], s["attack_success_rate"], "-o", color=COL[m],
                    lw=1.6, ms=4.5, mec="white", mew=0.5, label=m)
        ax.set_xticks(SNR)
        ax.invert_xaxis()
        ax.set_xlabel(t["snr"])
        ax.set_title(ti, fontsize=11)
        ax.set_ylim(-0.03, 1.05)
    axs[0].set_ylabel(t["asr"])
    axs[0].legend(fontsize=7.5, frameon=False, loc="lower left", ncol=1)
    fig.suptitle(t["dose"], fontsize=11.5, y=1.02)
    save(fig, f"fig_attackdose_{lang}")

def fig_transfer(t, lang):
    p = tr[tr["distortion_type"] == "pgd"]
    pairs = [("xlsr_mamba", "ssl_aasist", "#D55E00", t["shared"]),
             ("xlsr_mamba", "aasist", "#0072B2", t["diff"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for src, vic, c, lab in pairs:
        s = p[(p["attacked_model"] == src) & (p["transfer_model"] == vic)].sort_values("level", ascending=False)
        ax.plot(s["level"], s["delta_eer"], "-o", color=c, lw=2.2, ms=6,
                mec="white", mew=0.7, label=lab)
    ax.set_xticks(SNR)
    ax.invert_xaxis()
    ax.set_xlabel(t["snr"])
    ax.set_ylabel(t["vdeer"])
    ax.set_title(t["trf"], fontsize=11.5)
    ax.set_ylim(-0.03, 1.02)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    save(fig, f"fig_transfer_{lang}")

def fig_calibration(t, lang):
    off = {"SSL-AASIST": (0, 11, "center"), "Nes2Net": (8, -3, "left"),
           "XLSR-Mamba": (-2, -16, "center"), "Spectra-AASIST3": (8, -3, "left"),
           "reference": (0, 9, "center"), "RawNet2": (8, 2, "left"),
           "AASIST": (8, 2, "left"), "AASIST3 (KAN)": (8, 2, "left")}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for m in MODELS:
        e = clean_eer(m, "in_the_wild")
        s = cs[(cs["model"] == KEY[m]) & (cs["dataset"] == "in_the_wild")]
        c = float(s["cllr"].iloc[0])
        ax.scatter(e, c, color=COL[m], s=55, zorder=3, ec="white", lw=0.6)
        dx, dy, ha = off[m]
        ax.annotate(m, (e, c), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=COL[m], ha=ha)
    ax.axhline(1.0, color="#999999", ls="--", lw=1)
    ax.text(0.755, 1.06, t["broken"], fontsize=8.5, color="#666666", va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xlabel(t["eer"])
    ax.set_ylabel(t["cllr"])
    ax.set_title(t["cal"], fontsize=11.5)
    ax.set_xlim(-0.04, 0.78)
    save(fig, f"fig_calibration_{lang}")

def fig_heatmap(t, lang):
    cols = ["noise", "reverb", "chirp", "bandpass", "gain", "codec",
            "transcode_identity", "transmission", "neural_encodec", "telephony_g722"]
    M = np.full((8, len(cols)), np.nan)
    for i, m in enumerate(MODELS):
        sub = dg19[dg19["model_name"] == KEY[m]]
        for j, c in enumerate(cols):
            s = sub[sub["distortion_type"] == c]
            if len(s):
                M[i, j] = s["delta_eer"].max()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([t["fams"][c] for c in cols], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(8))
    ax.set_yticklabels(MODELS, fontsize=9)
    for i in range(8):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.4 * np.nanmax(M) else "#333333")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("ΔEER", fontsize=9)
    ax.set_title(t["hm"], fontsize=11.5)
    save(fig, f"fig_heatmap_{lang}")

def fig_encodec(t, lang):
    sub = dg19[dg19["distortion_type"] == "neural_encodec"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    br = sorted(sub["level"].unique(), reverse=True)
    for m in ["AASIST", "AASIST3 (KAN)", "Spectra-AASIST3"]:
        s = sub[sub["model_name"] == KEY[m]]
        ys = [float(s[s["level"] == b]["eer"].iloc[0]) for b in br]
        ax.plot(range(len(br)), ys, "-o", color=COL[m], lw=1.9, ms=5.5,
                mec="white", mew=0.6)
        ax.annotate(m, (len(br) - 1, ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=COL[m])
    ax.set_xticks(range(len(br)))
    ax.set_xticklabels([f"{b:g}" for b in br])
    ax.set_xlabel(t["br"])
    ax.set_ylabel(t["eer"])
    ax.set_title(t["enc"], fontsize=11.5)
    ax.set_xlim(-0.1, len(br) - 0.1 + 0.8)
    ax.set_ylim(0, 0.6)
    save(fig, f"fig_encodec_{lang}")

def fig_smallmult(t, lang):
    dists = ["noise", "reverb", "chirp", "bandpass"]
    mods = ["AASIST", "Spectra-AASIST3", "XLSR-Mamba", "reference"]
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.6))
    axs = axs.ravel()
    for k, d in enumerate(dists):
        ax = axs[k]
        sub = dg19[dg19["distortion_type"] == d]
        levels = sorted(sub["level"].unique(),
                        key=lambda lv: sub[sub["level"] == lv]["delta_eer"].mean())
        xs = range(len(levels))
        for m in mods:
            ys = [sub[(sub["model_name"] == KEY[m]) & (sub["level"] == lv)]["delta_eer"].mean()
                  for lv in levels]
            ax.plot(xs, ys, "-o", color=COL[m], lw=1.5, ms=4, mec="white", mew=0.5, label=m)
        ax.set_title(t["fams"][d], fontsize=10.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([])
        ax.set_ylabel(t["deer"], fontsize=9)
        ax.set_xlabel(t["sev"], fontsize=9)
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(t["sm"], fontsize=11.5, y=1.01)
    fig.tight_layout()
    save(fig, f"fig_smallmult_{lang}")

def fig_realsynth(t, lang):
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 4.2), sharey=True)
    for ax, (m, rows) in zip(axs, RS.items()):
        fams = [r[0] for r in rows]
        x = np.arange(len(fams))
        w = 0.38
        ax.bar(x - w / 2, [r[1] for r in rows], w, color=COL[m])
        ax.bar(x + w / 2, [r[2] for r in rows], w, color=COL[m], alpha=0.45, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([t["rsl"][f] for f in fams], fontsize=9)
        ax.set_title(m, fontsize=10)
        ax.set_ylim(0, 0.34)
    axs[0].set_ylabel(t["deer"])
    handles = [Patch(facecolor="#888888", label=t["real"]),
               Patch(facecolor="#cccccc", hatch="//", label=t["syn"])]
    axs[2].legend(handles=handles, fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(t["rs"], fontsize=11.5, y=1.02)
    fig.tight_layout()
    save(fig, f"fig_realsynth_{lang}")

def fig_transfermatrix(t, lang):
    p20 = tr[(tr["distortion_type"] == "pgd") & (tr["level"] == 20.0)]
    srcs = ["aasist", "aasist3_kan", "spectra_aasist3", "ssl_aasist", "rawnet2",
            "xlsr_mamba", "nes2net"]
    vics = ["aasist", "aasist3_kan", "ssl_aasist"]
    disp = {"aasist": "AASIST", "aasist3_kan": "AASIST3", "spectra_aasist3": "Spectra",
            "ssl_aasist": "SSL-AASIST", "rawnet2": "RawNet2", "xlsr_mamba": "XLSR-Mamba",
            "nes2net": "Nes2Net"}
    M = np.full((len(srcs), len(vics)), np.nan)
    for _, r in p20.iterrows():
        s, v = r["attacked_model"], r["transfer_model"]
        if s in srcs and v in vics:
            M[srcs.index(s), vics.index(v)] = r["delta_eer"]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    im = ax.imshow(M, cmap="PuBuGn", aspect="auto", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(vics)))
    ax.set_xticklabels([disp[v] for v in vics], fontsize=9)
    ax.set_yticks(range(len(srcs)))
    ax.set_yticklabels([disp[s] for s in srcs], fontsize=9)
    ax.set_xlabel(t["vic"])
    ax.set_ylabel(t["src"])
    for i in range(len(srcs)):
        for j in range(len(vics)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] > 0.45 * np.nanmax(M) else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ΔEER", fontsize=9)
    ax.set_title(t["tm"], fontsize=11)
    fig.tight_layout()
    save(fig, f"fig_transfermatrix_{lang}")

def fig_bump(t, lang):
    ranks = {}
    for d in DOMS:
        order = sorted(MODELS, key=lambda m: clean_eer(m, d))
        for r, m in enumerate(order, 1):
            ranks.setdefault(m, []).append(r)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for m in MODELS:
        ax.plot([0, 1, 2], ranks[m], "-o", color=COL[m], lw=1.9, ms=6, mec="white", mew=0.7)
        ax.annotate(m, (2, ranks[m][2]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=COL[m])
        ax.annotate(m, (0, ranks[m][0]), xytext=(-8, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=8.5, color=COL[m])
    ax.set_yticks(range(1, 9))
    ax.invert_yaxis()
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(t["doms"])
    ax.set_ylabel(t["rank"])
    ax.set_title(t["bump"], fontsize=11.5)
    ax.set_xlim(-0.7, 2.7)
    ax.grid(axis="x", alpha=0)
    save(fig, f"fig_bump_{lang}")

def fig_forest(t, lang):
    rows = []
    for m in MODELS:
        s = wd19[wd19["model_name"] == KEY[m]].sort_values("delta_eer", ascending=False).head(1)
        if len(s):
            rows.append((m, float(s["delta_eer"].iloc[0]), float(s["delta_eer_ci_low"].iloc[0]),
                         float(s["delta_eer_ci_high"].iloc[0]), s["distortion_type"].iloc[0]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for i, (m, d, lo, hi, dist) in enumerate(rows):
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=COL[m], ms=7,
                    mec="white", mew=0.7, capsize=3, lw=1.6)
        ax.annotate(dist, (hi, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#555")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(t["deer"])
    ax.set_title(t["fr"], fontsize=11.5)
    ax.set_xlim(0, 0.95)
    ax.grid(axis="y", alpha=0)
    save(fig, f"fig_forest_{lang}")

def fig_radar(t, lang):
    fams = ["noise", "reverb", "chirp", "bandpass", "codec", "gain"]
    mods = ["AASIST", "Spectra-AASIST3", "XLSR-Mamba", "reference"]
    ang = [n / len(fams) * 2 * pi for n in range(len(fams))]
    ang += ang[:1]
    fig = plt.figure(figsize=(6.8, 6.4))
    ax = plt.subplot(111, polar=True)
    for m in mods:
        vals = [wd19[(wd19["model_name"] == KEY[m]) & (wd19["distortion_type"] == f)]["delta_eer"].max()
                for f in fams]
        vals = [0 if pd.isna(v) else max(0, v) for v in vals]
        vals += vals[:1]
        ax.plot(ang, vals, "-o", color=COL[m], lw=1.7, ms=4, label=m)
        ax.fill(ang, vals, color=COL[m], alpha=0.07)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([t["fams"][f] for f in fams], fontsize=9.5)
    ax.set_ylim(0, 0.85)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], ["0,2" if lang == "ru" else "0.2",
                  "0,4" if lang == "ru" else "0.4", "0,6" if lang == "ru" else "0.6",
                  "0,8" if lang == "ru" else "0.8"], angle=262, fontsize=8, color="#888888")
    ax.set_title(t["rad"], fontsize=11.5, y=1.10)
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.13))
    save(fig, f"fig_radar_{lang}")

def fig_dumbbell(t, lang):
    rows = []
    for m in MODELS:
        s = wd19[wd19["model_name"] == KEY[m]].sort_values("eer", ascending=False).head(1)
        if len(s):
            rows.append((m, clean_eer(m, "asvspoof2019_la"), float(s["eer"].iloc[0]),
                         s["distortion_type"].iloc[0]))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for i, (m, c, w, dist) in enumerate(rows):
        ax.plot([c, w], [i, i], "-", color="#bbbbbb", lw=2, zorder=1)
        ax.scatter(c, i, s=52, facecolor="white", edgecolor=COL[m], lw=1.6, zorder=3)
        ax.scatter(w, i, s=58, color=COL[m], ec="white", lw=0.6, zorder=3)
        ax.annotate(t["fams"].get(dist, dist), (w, i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#555")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(t["eer"])
    ax.set_title(t["db"], fontsize=11.5)
    ax.set_xlim(-0.02, 1.0)
    ax.grid(axis="y", alpha=0)
    h = [plt.scatter([], [], s=52, facecolor="white", edgecolor="#555", lw=1.6, label=t["dbc"]),
         plt.scatter([], [], s=58, color="#555", label=t["dbw"])]
    ax.legend(handles=h, fontsize=8.5, frameon=False, loc="lower right")
    save(fig, f"fig_dumbbell_{lang}")

def fig_coverage(t, lang):
    fams = ["noise", "reverb", "chirp", "bandpass", "gain", "codec", "transcode_identity",
            "transmission", "telephony_g711a", "telephony_g711u", "telephony_g722",
            "neural_encodec", "real_bird", "real_noise", "real_rir"]
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 4.6), sharey=True)
    for ax, d, ti in zip(axs, DOMS, t["doms"]):
        sub = dg[dg["dataset"] == d]
        M = np.zeros((8, len(fams)))
        for i, m in enumerate(MODELS):
            for j, f in enumerate(fams):
                M[i, j] = 1.0 if len(sub[(sub["model_name"] == KEY[m]) &
                                         (sub["distortion_type"] == f)]) else 0.0
        ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#eeeeee", "#2a9d8f"]),
                  aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([t["fams"][f] for f in fams], rotation=90, fontsize=7.5)
        ax.set_title(ti.replace("\n", " "), fontsize=10.5)
        ax.set_xticks(np.arange(-0.5, len(fams)), minor=True)
        ax.set_yticks(np.arange(-0.5, 8), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.grid(which="major", alpha=0)
        ax.tick_params(which="minor", length=0)
    axs[0].set_yticks(range(8))
    axs[0].set_yticklabels(MODELS, fontsize=8.5)
    handles = [Patch(facecolor="#2a9d8f", label=t["covy"]),
               Patch(facecolor="#eeeeee", label=t["covn"])]
    fig.legend(handles=handles, fontsize=9, frameon=False, loc="upper right",
               bbox_to_anchor=(0.99, 1.04), ncol=2)
    fig.suptitle(t["cov"], fontsize=11.5, y=1.05, x=0.44)
    fig.tight_layout()
    save(fig, f"fig_coverage_{lang}")

def _box(ax, x, y, w, h, text, fc, fs=8.2, bold=False, ec="#555555"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#222222")

def _arrow(ax, x1, y1, x2, y2, color="#555555", ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, ls=ls, lw=lw))

def fig_pipeline(t, lang):
    p = PIPE[lang]
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    cx = [1.5, 22.5, 41.5, 62.5, 83.0]
    cw = [17, 15, 18, 17, 15.5]
    heads = [p["c1"], p["c2"], p["c3"], p["c4"], p["c5"]]
    for x, w, htxt in zip(cx, cw, heads):
        ax.text(x + w / 2, 95, htxt, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="#222222")
    fills = ["#e8f1f8", "#eef7ee", "#fdf2e6", "#f3eef8", "#f7f7e8"]
    blocks = [p["c1b"], p["c2b"], p["c3b"], p["c4b"], p["c5b"]]
    ys_all = []
    for ci, (x, w, items, fc) in enumerate(zip(cx, cw, blocks, fills)):
        n = len(items)
        gap = 3.5
        avail = 84
        bh = min(19, (avail - gap * (n - 1)) / n)
        total = bh * n + gap * (n - 1)
        y0 = 6 + (avail - total) / 2
        ys = []
        for k, it in enumerate(items):
            y = y0 + (n - 1 - k) * (bh + gap)
            _box(ax, x, y, w, bh, it, fc)
            ys.append(y + bh / 2)
        ys_all.append(ys)
    for ci in range(4):
        x1 = cx[ci] + cw[ci]
        x2 = cx[ci + 1]
        y1 = float(np.mean(ys_all[ci]))
        y2 = float(np.mean(ys_all[ci + 1]))
        _arrow(ax, x1 + 0.4, y1, x2 - 0.4, y2)
    ax.set_title(t["pipe"], fontsize=12, pad=14)
    save(fig, f"fig_pipeline_{lang}")

def fig_arch(t, lang):
    a = ARCH[lang]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    _box(ax, 2, 42, 13, 16, a["wav"], "#eef3f7", fs=9)
    _box(ax, 6, 8, 17, 15, a["pert"], "#fdecec", fs=8.6, ec="#b03030")
    _arrow(ax, 14.5, 23, 10.5, 41, color="#b03030", ls="--", lw=1.6)
    _box(ax, 24, 36, 24, 28, a["fe"], "#fdf2e6", fs=10, bold=True)
    ax.text(36, 32.5, a["fe2"], ha="center", va="center", fontsize=8, color="#666666")
    _arrow(ax, 15.2, 50, 23.6, 50)
    _box(ax, 58, 62, 20, 16, a["b1"], "#e8f1f8", fs=9)
    _box(ax, 58, 22, 20, 16, a["b2"], "#eef7ee", fs=9)
    _arrow(ax, 48.4, 55, 57.6, 70)
    _arrow(ax, 48.4, 45, 57.6, 30)
    _box(ax, 86, 63, 12, 14, a["s1"], "#ffffff", fs=8.6)
    _box(ax, 86, 23, 12, 14, a["s2"], "#ffffff", fs=8.6)
    _arrow(ax, 78.4, 70, 85.6, 70)
    _arrow(ax, 78.4, 30, 85.6, 30)
    ax.text(50, 5, a["note"], ha="center", va="bottom", fontsize=9,
            color="#b03030", style="italic")
    ax.set_title(t["arch"], fontsize=12, pad=10)
    save(fig, f"fig_arch_{lang}")

for lang in ["ru", "en"]:
    t = L[lang]
    fig_domainshift(t, lang)
    fig_attackdose(t, lang)
    fig_transfer(t, lang)
    fig_calibration(t, lang)
    fig_heatmap(t, lang)
    fig_encodec(t, lang)
    fig_smallmult(t, lang)
    fig_realsynth(t, lang)
    fig_transfermatrix(t, lang)
    fig_bump(t, lang)
    fig_forest(t, lang)
    fig_radar(t, lang)
    fig_dumbbell(t, lang)
    fig_coverage(t, lang)
    fig_pipeline(t, lang)
    fig_arch(t, lang)

print("ok", len([f for f in os.listdir(OUT) if f.endswith(".png")]))
