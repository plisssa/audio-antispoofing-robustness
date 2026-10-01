import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def equal_error_rate(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])


def area_under_curve(scores, labels):
    return float(roc_auc_score(labels, scores))


def f1_at_threshold(scores, labels, threshold):
    predictions = (np.asarray(scores) >= threshold).astype(int)
    return float(f1_score(labels, predictions, zero_division=0))


def min_detection_cost(scores, labels, p_target=0.05, c_miss=1.0, c_fa=1.0):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    cost = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    normalizer = min(c_miss * p_target, c_fa * (1.0 - p_target))
    index = int(np.argmin(cost))
    return float(cost[index] / normalizer), float(thresholds[index])


def log_likelihood_ratio(scores):
    probability = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _cllr_from_llr(llr, labels):
    target = llr[labels == 1]
    nontarget = llr[labels == 0]
    if len(target) == 0 or len(nontarget) == 0:
        return float("nan")
    target_cost = np.mean(np.logaddexp(0.0, -target)) / np.log(2.0)
    nontarget_cost = np.mean(np.logaddexp(0.0, nontarget)) / np.log(2.0)
    return float(0.5 * (target_cost + nontarget_cost))


def cllr(scores, labels):
    return _cllr_from_llr(log_likelihood_ratio(scores), np.asarray(labels))


def min_cllr(scores, labels):
    from sklearn.isotonic import IsotonicRegression

    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    posterior = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6).fit_transform(
        np.asarray(scores, dtype=float), labels
    )
    return _cllr_from_llr(log_likelihood_ratio(posterior), labels)


def summary(scores, labels, threshold=None, dcf=None):
    eer, eer_threshold = equal_error_rate(scores, labels)
    operating_point = eer_threshold if threshold is None else threshold
    min_dcf, _ = min_detection_cost(scores, labels, **(dcf or {}))
    return {
        "eer": eer,
        "auc": area_under_curve(scores, labels),
        "f1": f1_at_threshold(scores, labels, operating_point),
        "min_dcf": min_dcf,
        "cllr": cllr(scores, labels),
        "min_cllr": min_cllr(scores, labels),
        "threshold": operating_point,
        "n": int(len(labels)),
    }


def bootstrap_delta_eer_ci(clean_scores, distorted_scores, labels, n_boot=1000, seed=1337, alpha=0.05):
    labels = np.asarray(labels)
    clean_scores = np.asarray(clean_scores, dtype=float)
    distorted_scores = np.asarray(distorted_scores, dtype=float)
    count = len(labels)
    if n_boot <= 0 or count == 0 or labels.sum() == 0 or labels.sum() == count:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        index = rng.integers(0, count, count)
        resampled = labels[index]
        if resampled.sum() == 0 or resampled.sum() == len(resampled):
            continue
        clean_eer, _ = equal_error_rate(clean_scores[index], resampled)
        distorted_eer, _ = equal_error_rate(distorted_scores[index], resampled)
        deltas.append(distorted_eer - clean_eer)
    if not deltas:
        return float("nan"), float("nan")
    return float(np.quantile(deltas, alpha / 2.0)), float(np.quantile(deltas, 1.0 - alpha / 2.0))


def delta_eer(eer_clean, eer_distorted):
    return float(eer_distorted - eer_clean)


def relative_degradation(clean_value, distorted_value):
    if abs(clean_value) < 1e-6:
        return float("nan")
    return float((distorted_value - clean_value) / abs(clean_value))


def attack_success_rate(labels, predictions_clean, predictions_attacked):
    labels = np.asarray(labels)
    correct_before = np.asarray(predictions_clean) == labels
    flipped = (np.asarray(predictions_attacked) != labels) & correct_before
    denominator = int(correct_before.sum())
    return float(flipped.sum() / denominator) if denominator else 0.0
