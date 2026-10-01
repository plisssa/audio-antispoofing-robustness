import numpy as np
import pandas as pd
import pytest

from adfd.eval.alignment import alignment, pair_alignment


def test_alignment_extremes():
    rng = np.random.default_rng(0)
    g = rng.standard_normal(1000)
    cosine, sign = alignment(g, g)
    assert cosine == pytest.approx(1.0)
    assert sign == pytest.approx(1.0)
    cosine, sign = alignment(g, -g)
    assert cosine == pytest.approx(-1.0)
    assert sign == pytest.approx(-1.0)


def test_alignment_unrelated_is_near_zero():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(200000)
    b = rng.standard_normal(200000)
    cosine, sign = alignment(a, b)
    assert abs(cosine) < 0.01
    assert abs(sign) < 0.01


def test_sign_alignment_is_scale_invariant_and_asymmetric():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(5000)
    b = 0.7 * a + 0.7 * rng.standard_normal(5000) * np.abs(a)
    forward = alignment(a, b)[1]
    assert alignment(10.0 * a, 0.01 * b)[1] == pytest.approx(forward)
    assert alignment(b, a)[1] != pytest.approx(forward)


def test_pair_alignment_truncates_to_shared_support():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(64000)
    gradients = {
        "m1": {"f": np.concatenate([base, rng.standard_normal(600)])},
        "m2": {"f": base.copy()},
    }
    summary, per_file = pair_alignment(gradients, {"f": 64000})
    assert set(zip(summary["source"], summary["victim"])) == {("m1", "m2"), ("m2", "m1")}
    assert (summary["cosine_mean"] > 0.999).all()
    assert (per_file["length"] == 64000).all()


def test_pair_alignment_respects_short_signal_support():
    rng = np.random.default_rng(4)
    head = rng.standard_normal(1000)
    gradients = {
        "tiled": {"f": np.concatenate([head, rng.standard_normal(63000)])},
        "padded": {"f": np.concatenate([head, np.zeros(63000)])},
    }
    summary, _ = pair_alignment(gradients, {"f": 1000})
    assert (summary["cosine_mean"] > 0.999).all()


def test_spectra_preemphasis_moved_into_forward_pass():
    torch = pytest.importorskip("torch")
    from adfd.models.spectra_aasist3 import SpectraAASIST3Detector

    detector = SpectraAASIST3Detector(crop=4000)
    rng = np.random.default_rng(5)
    signal = (0.1 * rng.standard_normal(5000)).astype(np.float32)
    prepared = detector.prepare(signal)
    assert np.array_equal(prepared, signal[:4000])
    assert np.array_equal(detector.to_raw(prepared), prepared)

    detector.torch = torch
    detector.model = torch.nn.Identity()
    passed = detector.logits_tensor(torch.tensor(prepared)[None, :])[0].numpy()
    legacy = np.append(prepared[0], prepared[1:] - detector.preemphasis * prepared[:-1]).astype(np.float32)
    assert np.allclose(passed, legacy, atol=1e-6)

    x = torch.tensor(prepared)[None, :].requires_grad_(True)
    detector.logits_tensor(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_base_detector_raw_mappings_are_identity():
    from adfd.models.base import Detector

    detector = Detector()
    x = np.arange(10.0)
    assert np.array_equal(detector.raw_gradient(x), x)
    assert np.array_equal(detector.to_raw(x), x)


def test_ssl_probes_register_with_distinct_names():
    import yaml
    from adfd.models.base import build_detector

    models = yaml.safe_load(open("configs/models.yaml"))
    expected = {
        "hubert_linear": ("facebook/hubert-large-ll60k", "linear"),
        "hubert_mlp": ("facebook/hubert-large-ll60k", "mlp"),
        "w2v2lv60_linear": ("facebook/wav2vec2-large-lv60", "linear"),
        "w2v2lv60_mlp": ("facebook/wav2vec2-large-lv60", "mlp"),
        "wavlmbase_linear": ("microsoft/wavlm-base-plus", "linear"),
        "wavlmbase_mlp": ("microsoft/wavlm-base-plus", "mlp"),
    }
    names = set()
    for key, (model_id, head) in expected.items():
        detector = build_detector(key, sample_rate=16000, **models[key])
        assert detector.name == key
        assert detector.model_id == model_id
        assert detector.head_kind == head
        names.add(detector.name)
    assert len(names) == len(expected)
    assert build_detector("xlsr_linear", sample_rate=16000, **models["xlsr_linear"]).name == "xlsr_linear"
    assert build_detector("wavlm_linear", sample_rate=16000, **models["wavlm_linear"]).name == "wavlm_linear"


def test_pgd_restarts_is_at_least_as_strong_as_single_pgd():
    torch = pytest.importorskip("torch")
    from adfd.attacks.gradient import pgd, pgd_restarts
    from adfd.models.torch_reference import TorchReferenceDetector

    torch.manual_seed(0)
    detector = TorchReferenceDetector(crop=4000).load()
    x = (0.1 * torch.randn(1, 4000)).clamp(-1, 1)
    label = torch.tensor([1])
    epsilon = 0.01

    def loss(adv):
        with torch.no_grad():
            return float(torch.nn.functional.cross_entropy(detector.logits_tensor(adv), label))

    torch.manual_seed(1)
    single = loss(pgd(detector, x, label, epsilon, steps=5))
    torch.manual_seed(1)
    strong = pgd_restarts(detector, x, label, epsilon, steps=5, restarts=4, alpha_ratio=0.25)
    assert (strong - x).abs().max() <= epsilon + 1e-6
    assert loss(strong) >= single - 1e-6


def test_spsa_is_gradient_free_and_raises_loss():
    torch = pytest.importorskip("torch")
    from adfd.attacks.gradient import spsa
    from adfd.models.torch_reference import TorchReferenceDetector

    torch.manual_seed(0)
    detector = TorchReferenceDetector(crop=4000).load()
    for parameter in detector.model.parameters():
        parameter.requires_grad_(False)
    x = (0.1 * torch.randn(1, 4000)).clamp(-1, 1)
    label = torch.tensor([1])
    epsilon = 0.02

    def loss(signal):
        with torch.no_grad():
            return float(torch.nn.functional.cross_entropy(detector.logits_tensor(signal), label))

    torch.manual_seed(2)
    adversarial = spsa(detector, x, label, epsilon, steps=15, samples=16, chunk=8)
    assert not adversarial.requires_grad
    assert (adversarial - x).abs().max() <= epsilon + 1e-6
    assert loss(adversarial) > loss(x)


def test_square_schedule_halves_window():
    from adfd.attacks.gradient import square_fraction

    fractions = [square_fraction(0.8, i, 1000) for i in range(1, 1000)]
    assert fractions[0] == pytest.approx(0.8)
    assert fractions[-1] == pytest.approx(0.8 / 512)
    assert all(a >= b for a, b in zip(fractions, fractions[1:]))


def test_square_uses_full_budget_and_beats_random_start():
    torch = pytest.importorskip("torch")
    from adfd.attacks.gradient import logit_margin, square
    from adfd.models.torch_reference import TorchReferenceDetector

    torch.manual_seed(0)
    detector = TorchReferenceDetector(crop=4000).load()
    x = (0.1 * torch.randn(1, 4000)).clamp(-1, 1)
    label = torch.tensor([int(detector.logits_tensor(x).argmax())])
    epsilon = 0.02

    torch.manual_seed(3)
    start = (x + epsilon * (torch.randint(0, 2, x.shape).float() * 2 - 1)).clamp(-1, 1)
    torch.manual_seed(3)
    adversarial = square(detector, x, label, epsilon, queries=200, candidates=4)
    delta = (adversarial - x).abs()
    assert not adversarial.requires_grad
    assert delta.max() <= epsilon + 1e-6
    assert torch.allclose(delta, torch.full_like(delta, epsilon), atol=1e-6)
    assert float(logit_margin(detector, adversarial, label)[0]) < float(logit_margin(detector, start, label)[0])
    assert float(logit_margin(detector, adversarial, label)[0]) < float(logit_margin(detector, x, label)[0])


def test_margin_gradient_matches_finite_difference():
    torch = pytest.importorskip("torch")
    from adfd.attacks.gradient import logit_margin, margin_gradient
    from adfd.models.torch_reference import TorchReferenceDetector

    torch.manual_seed(0)
    detector = TorchReferenceDetector(crop=4000).load()
    detector.model.double()
    x = (0.1 * torch.randn(1, 4000, dtype=torch.float64)).clamp(-1, 1)
    label = torch.tensor([0])
    gradient = margin_gradient(detector, x, label)
    v = torch.randn_like(x)
    h = 1e-6
    numeric = (-float(logit_margin(detector, x + h * v, label)[0]) + float(logit_margin(detector, x - h * v, label)[0])) / (2 * h)
    assert float((gradient * v).sum()) == pytest.approx(numeric, rel=1e-4)


def test_pgd_margin_stays_in_budget_and_lowers_margin():
    torch = pytest.importorskip("torch")
    from adfd.attacks.gradient import logit_margin, pgd_margin
    from adfd.models.torch_reference import TorchReferenceDetector

    torch.manual_seed(0)
    detector = TorchReferenceDetector(crop=4000).load()
    x = (0.1 * torch.randn(1, 4000)).clamp(-1, 1)
    label = torch.tensor([int(detector.logits_tensor(x).argmax())])
    epsilon = 0.005
    adversarial = pgd_margin(detector, x, label, epsilon, steps=10, restarts=2, alpha_ratio=0.25)
    assert (adversarial - x).abs().max() <= epsilon + 1e-6
    assert float(logit_margin(detector, adversarial, label)[0]) < float(logit_margin(detector, x, label)[0])


def test_gradcheck_file_checks_and_trajectory():
    torch = pytest.importorskip("torch")
    from adfd.eval.gradcheck import checkpoint_mismatch, file_checks, trajectory, training_modules
    from adfd.models.torch_reference import TorchReferenceDetector

    torch.manual_seed(0)
    detector = TorchReferenceDetector(crop=4000).load()
    detector.model.eval()
    x = (0.1 * torch.randn(1, 4000)).clamp(-1, 1)
    label = torch.tensor([0])
    generator = torch.Generator().manual_seed(1)
    record = file_checks(detector, x, label, 1e-5, 1e-3, 4, generator)
    assert record["repeat_abs_diff"] == 0.0
    assert record["grad_finite"]
    assert record["fine_sign_match"] == 1.0
    assert record["fine_ratio"] == pytest.approx(1.0, abs=0.05)
    values = trajectory(detector, x, label, 0.005, 5, 0.25, "margin", generator)
    assert len(values) == 6
    assert training_modules(detector) == 0
    assert checkpoint_mismatch(detector)["missing_keys"] is None


def test_pgd_warm_starts_from_cached_surrogate_attack():
    torch = pytest.importorskip("torch")
    from adfd.attacks import gradient
    from adfd.models.torch_reference import TorchReferenceDetector

    torch.manual_seed(0)
    victim = TorchReferenceDetector(crop=4000).load()
    torch.manual_seed(1)
    surrogate = TorchReferenceDetector(crop=4000).load()
    surrogate.name = "reference_surrogate"
    x = (0.1 * torch.randn(1, 4000)).clamp(-1, 1)
    label = torch.tensor([1])
    epsilon = 0.005
    gradient.SURROGATE_CACHE.clear()

    transferred = gradient.surrogate_transfer(victim, x, label, epsilon, surrogate=surrogate, surrogate_steps=5, surrogate_restarts=2, alpha_ratio=0.25)
    assert len(gradient.SURROGATE_CACHE) == 1
    assert (transferred - x).abs().max() <= epsilon + 1e-6
    cached = next(iter(gradient.SURROGATE_CACHE.values()))
    assert torch.equal(transferred, cached)

    warm = gradient.pgd_warm(victim, x, label, epsilon, surrogate=surrogate, surrogate_steps=5, surrogate_restarts=2, steps=5, alpha_ratio=0.25)
    assert len(gradient.SURROGATE_CACHE) == 1
    assert (warm - x).abs().max() <= epsilon + 1e-6

    def loss(signal):
        with torch.no_grad():
            return float(torch.nn.functional.cross_entropy(victim.logits_tensor(signal), label))

    assert loss(warm) >= loss(transferred) - 1e-6

    surrogate.spoof_index = 0
    victim.spoof_index = 1
    gradient.surrogate_transfer(victim, x, label, epsilon, surrogate=surrogate, surrogate_steps=2, surrogate_restarts=1, alpha_ratio=0.25)
    assert {key[2] for key in gradient.SURROGATE_CACHE} == {0, 1}
    gradient.SURROGATE_CACHE.clear()


def test_perceptual_rows_measure_attack_and_matched_noise(tmp_path):
    torch = pytest.importorskip("torch")
    from adfd.datasets.bootstrap import make_bootstrap
    from adfd.datasets.manifest import read_manifest
    from adfd.eval.perceptual import pcm16, perceptual_rows, perceptual_summary, snr_db
    from adfd.models.torch_reference import TorchReferenceDetector

    manifest = read_manifest(make_bootstrap(tmp_path / "boot", n_per_class=2, sr=16000, seed=0))
    torch.manual_seed(0)
    detector = TorchReferenceDetector(crop=4000).load()
    attacks = {"pgd": {"attack": "pgd", "levels": [40, 20], "options": {"budget": "snr", "steps": 2}}}

    def fake_measure(reference, degraded, sample_rate):
        return {"pesq_wb": snr_db(reference, degraded - reference), "stoi": 1.0}

    rows = perceptual_rows(detector, manifest.head(2), attacks, 16000, seed=1, save_dir=tmp_path / "audio", save_n=1, measure=fake_measure)
    assert len(rows) == 2 * 2 * 2
    attack = rows[rows.kind == "attack"].reset_index(drop=True)
    noise = rows[rows.kind == "noise"].reset_index(drop=True)
    assert (attack.snr == noise.snr).all()
    assert (noise.pesq_wb - noise.snr).abs().max() < 0.5
    assert attack.pcm16_score.notna().all() and noise.get("pcm16_score", pd.Series([float("nan")])).isna().all()
    assert (tmp_path / "audio" / detector.name).exists()
    summary = perceptual_summary(rows)
    assert set(summary.kind) == {"attack", "noise"}
    assert pcm16(np.array([0.5, 1e-6], dtype=np.float32))[1] == 0.0
