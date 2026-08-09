import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

parser = argparse.ArgumentParser()
parser.add_argument("--found-db", type=int, required=True)
parser.add_argument("--found-other", type=int, default=0)
parser.add_argument("--duplicates", type=int, required=True)
parser.add_argument("--screened", type=int, required=True)
parser.add_argument("--excluded-screening", type=int, required=True)
parser.add_argument("--fulltext", type=int, required=True)
parser.add_argument("--excluded-fulltext", type=int, required=True)
parser.add_argument("--included", type=int, required=True)
parser.add_argument("--lang", choices=["ru", "en"], default="ru")
parser.add_argument("--out", default="prisma")
args = parser.parse_args()

L = {
 "ru": dict(ident="Идентификация", screen="Скрининг", incl="Включение",
    b1="Найдено записей в базах:\nn = {}", b1b="Из других источников:\nn = {}",
    b2="После удаления дубликатов:\nn = {}", b3="Просмотрено заголовков\nи аннотаций: n = {}",
    b3x="Исключено на скрининге:\nn = {}", b4="Полных текстов оценено:\nn = {}",
    b4x="Исключено полных текстов:\nn = {}", b5="Включено в обзор:\nn = {}"),
 "en": dict(ident="Identification", screen="Screening", incl="Included",
    b1="Records identified from\ndatabases: n = {}", b1b="From other sources:\nn = {}",
    b2="After duplicates removed:\nn = {}", b3="Titles and abstracts\nscreened: n = {}",
    b3x="Excluded at screening:\nn = {}", b4="Full texts assessed:\nn = {}",
    b4x="Full texts excluded:\nn = {}", b5="Included in the review:\nn = {}"),
}[args.lang]

fig, ax = plt.subplots(figsize=(8.4, 9.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def box(x, y, w, h, text, fc="#eef3f8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                fc=fc, ec="#444444", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color="#444444", lw=1.2))

for y0, y1, lab in [(80, 99, L["ident"]), (33, 76, L["screen"]), (4, 29, L["incl"])]:
    ax.add_patch(FancyBboxPatch((1, y0), 8, y1 - y0, boxstyle="round,pad=0.3",
                                fc="#dde6ee", ec="none"))
    ax.text(5, (y0 + y1) / 2, lab, rotation=90, ha="center", va="center", fontsize=10)

box(14, 88, 34, 9, L["b1"].format(args.found_db))
if args.found_other:
    box(56, 88, 30, 9, L["b1b"].format(args.found_other))
box(14, 66, 34, 8, L["b2"].format(args.found_db + args.found_other - args.duplicates))
box(14, 50, 34, 8, L["b3"].format(args.screened))
box(60, 50, 30, 8, L["b3x"].format(args.excluded_screening), "#f6ecec")
box(14, 34, 34, 8, L["b4"].format(args.fulltext))
box(60, 34, 30, 8, L["b4x"].format(args.excluded_fulltext), "#f6ecec")
box(14, 12, 34, 9, L["b5"].format(args.included), "#e9f3e9")

arrow(31, 87.6, 31, 74.5)
if args.found_other:
    arrow(71, 87.6, 33, 74.4)
arrow(31, 65.6, 31, 58.5)
arrow(31, 49.6, 31, 42.5)
arrow(48.4, 54, 59.6, 54)
arrow(31, 33.6, 31, 21.6)
arrow(48.4, 38, 59.6, 38)

for ext in ["png", "pdf"]:
    fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", facecolor="white", dpi=200)
print(f"{args.out}.png / {args.out}.pdf")
