from adfd.datasets.bootstrap import make_bootstrap
from adfd.datasets.manifest import read_manifest
from adfd.datasets.verify import dataset_report


def test_dataset_report_counts(tmp_path):
    out = tmp_path / "bootstrap"
    make_bootstrap(out, n_per_class=10, seed=1)
    report = dataset_report(read_manifest(out / "manifest.csv"))
    assert report["total"] == 20
    assert report["by_label"]["bona_fide"] == 10
    assert report["by_label"]["spoof"] == 10
    assert report["missing_files"] == 0
    assert report["unreadable_files"] == 0
    assert report["total_hours"] > 0.0
