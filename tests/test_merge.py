import pandas as pd

from adfd.datasets.bootstrap import make_bootstrap
from adfd.datasets.manifest import read_manifest, split_view


def _relabel(path, dataset):
    frame = pd.read_csv(path)
    frame["dataset"] = dataset
    frame.to_csv(path, index=False)


def test_merge_builds_cross_domain(tmp_path):
    from adfd.cli import build_parser

    make_bootstrap(tmp_path / "a", n_per_class=12, seed=1)
    make_bootstrap(tmp_path / "b", n_per_class=12, seed=2)
    _relabel(tmp_path / "a" / "manifest.csv", "domain_a")
    _relabel(tmp_path / "b" / "manifest.csv", "domain_b")

    out = tmp_path / "cross.csv"
    parser = build_parser()
    args = parser.parse_args([
        "merge",
        "--train-from", str(tmp_path / "a" / "manifest.csv"),
        "--eval-from", str(tmp_path / "b" / "manifest.csv"),
        "--out", str(out),
    ])
    args.func(args)

    merged = read_manifest(out)
    train = merged[merged["split"] == "train"]
    evaluation = split_view(merged, "eval")
    assert set(train["dataset"]) == {"domain_a"}
    assert set(evaluation["dataset"]) == {"domain_b"}


def test_subset_split_train(tmp_path):
    from adfd.cli import build_parser

    make_bootstrap(tmp_path / "a", n_per_class=20, seed=1)
    out = tmp_path / "train_small.csv"
    parser = build_parser()
    args = parser.parse_args([
        "subset",
        "--manifest", str(tmp_path / "a" / "manifest.csv"),
        "--out", str(out),
        "--n", "5",
        "--split", "train",
    ])
    args.func(args)
    frame = read_manifest(out)
    counts = frame[frame["split"] == "train"]["label"].value_counts().to_dict()
    assert counts == {"bona_fide": 5, "spoof": 5}
