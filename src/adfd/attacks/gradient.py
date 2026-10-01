from .base import register_attack


def cross_entropy_gradient(detector, x, label):
    torch = detector.torch
    x = x.clone().detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        loss = torch.nn.functional.cross_entropy(detector.logits_tensor(x), label)
        gradient = torch.autograd.grad(loss, x, allow_unused=True)[0]
    if gradient is None:
        raise RuntimeError("gradient does not reach the input; frontend likely detached or under no_grad")
    return gradient.detach()


def margin_gradient(detector, x, label):
    torch = detector.torch
    x = x.clone().detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        logits = detector.logits_tensor(x)
        index = label.expand(logits.shape[0])[:, None]
        margin = logits.gather(1, index)[:, 0] - logits.scatter(1, index, float("-inf")).max(dim=1).values
        gradient = torch.autograd.grad(-margin.sum(), x, allow_unused=True)[0]
    if gradient is None:
        raise RuntimeError("gradient does not reach the input; frontend likely detached or under no_grad")
    return gradient.detach()


LOSS_GRADIENTS = {"ce": cross_entropy_gradient, "margin": margin_gradient}


def attack_objective(detector, x, label, loss="ce"):
    torch = detector.torch
    if loss == "margin":
        return -float(logit_margin(detector, x, label)[0])
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        return float(torch.nn.functional.cross_entropy(detector.logits_tensor(x), label))


@register_attack("fgsm")
def fgsm(detector, x, label, epsilon, **_):
    gradient = cross_entropy_gradient(detector, x, label)
    return (x + epsilon * gradient.sign()).clamp(-1.0, 1.0).detach()


@register_attack("pgd_eot")
def pgd_eot(detector, x, label, epsilon, steps=10, alpha_ratio=0.25, eot_samples=4, noise_snr_db=20.0):
    torch = detector.torch
    alpha = max(epsilon * alpha_ratio, epsilon / steps)
    original = x.clone().detach()
    power = float(torch.mean(torch.square(original))) + 1e-12
    sigma = float((power / (10.0 ** (float(noise_snr_db) / 10.0))) ** 0.5)
    adversarial = (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(-1.0, 1.0).detach()
    for _ in range(steps):
        accumulated = torch.zeros_like(adversarial)
        for _ in range(int(eot_samples)):
            perturbed = (adversarial + sigma * torch.randn_like(adversarial)).clamp(-1.0, 1.0)
            accumulated = accumulated + cross_entropy_gradient(detector, perturbed, label)
        adversarial = adversarial + alpha * (accumulated / float(eot_samples)).sign()
        adversarial = torch.min(torch.max(adversarial, original - epsilon), original + epsilon)
        adversarial = adversarial.clamp(-1.0, 1.0).detach()
    return adversarial


@register_attack("pgd_l2")
def pgd_l2(detector, x, label, epsilon, steps=10, alpha_ratio=0.25):
    torch = detector.torch
    radius = float(epsilon) * float(x.numel()) ** 0.5
    alpha = max(radius * alpha_ratio, radius / steps)
    original = x.clone().detach()
    start = torch.randn_like(original)
    start = start / start.norm().clamp_min(1e-12) * radius * 0.1
    adversarial = (original + start).clamp(-1.0, 1.0).detach()
    for _ in range(steps):
        gradient = cross_entropy_gradient(detector, adversarial, label)
        direction = gradient / gradient.norm().clamp_min(1e-12)
        delta = adversarial + alpha * direction - original
        norm = delta.norm()
        if float(norm) > radius:
            delta = delta * (radius / norm)
        adversarial = (original + delta).clamp(-1.0, 1.0).detach()
    return adversarial


@register_attack("spsa")
def spsa(detector, x, label, epsilon, steps=100, samples=32, alpha_ratio=0.25, delta_ratio=1.0, chunk=16):
    torch = detector.torch
    delta = float(epsilon) * delta_ratio
    alpha = max(epsilon * alpha_ratio, epsilon / steps)
    original = x.clone().detach()
    adversarial = original.clone()
    for _ in range(int(steps)):
        signs = torch.randint(0, 2, (int(samples), original.shape[1]), device=original.device).to(original.dtype) * 2 - 1
        probes = torch.cat([adversarial + delta * signs, adversarial - delta * signs]).clamp(-1.0, 1.0)
        losses = []
        with torch.no_grad():
            for start in range(0, probes.shape[0], int(chunk)):
                logits = detector.logits_tensor(probes[start:start + int(chunk)])
                losses.append(torch.nn.functional.cross_entropy(logits, label.expand(logits.shape[0]), reduction="none"))
        losses = torch.cat(losses)
        estimate = ((losses[:int(samples)] - losses[int(samples):]) / (2.0 * delta))[:, None] * signs
        adversarial = adversarial + alpha * estimate.mean(dim=0, keepdim=True).sign()
        adversarial = torch.min(torch.max(adversarial, original - epsilon), original + epsilon)
        adversarial = adversarial.clamp(-1.0, 1.0).detach()
    return adversarial


def square_fraction(p_init, iteration, iterations):
    scaled = int(iteration / iterations * 10000)
    for bound, divisor in ((10, 1), (50, 2), (200, 4), (500, 8), (1000, 16), (2000, 32), (4000, 64), (6000, 128), (8000, 256)):
        if scaled <= bound:
            return p_init / divisor
    return p_init / 512


def logit_margin(detector, x, label):
    torch = detector.torch
    with torch.no_grad():
        logits = detector.logits_tensor(x)
    index = label.expand(logits.shape[0])[:, None]
    true = logits.gather(1, index)[:, 0]
    others = logits.scatter(1, index, float("-inf")).max(dim=1).values
    return true - others


@register_attack("square")
def square(detector, x, label, epsilon, queries=1000, p_init=0.8, candidates=1):
    torch = detector.torch
    original = x.clone().detach()
    n = original.shape[1]
    signs = torch.randint(0, 2, original.shape, device=original.device).to(original.dtype) * 2 - 1
    adversarial = (original + epsilon * signs).clamp(-1.0, 1.0)
    best = float(logit_margin(detector, adversarial, label)[0])
    iterations = max(1, int(queries) // int(candidates))
    for iteration in range(1, iterations):
        width = min(n, max(1, int(round(square_fraction(p_init, iteration, iterations) * n))))
        batch = adversarial.repeat(int(candidates), 1)
        for k in range(int(candidates)):
            start = int(torch.randint(0, n - width + 1, (1,)))
            value = float(epsilon) * (1.0 if int(torch.randint(0, 2, (1,))) else -1.0)
            current = adversarial[0, start:start + width] - original[0, start:start + width]
            if bool(torch.all(current == value)):
                value = -value
            batch[k, start:start + width] = (original[0, start:start + width] + value).clamp(-1.0, 1.0)
        margins = logit_margin(detector, batch, label)
        k = int(torch.argmin(margins))
        if float(margins[k]) < best:
            best = float(margins[k])
            adversarial = batch[k:k + 1].clone()
    return adversarial.detach()


@register_attack("pgd_restarts")
def pgd_restarts(detector, x, label, epsilon, steps=100, restarts=3, alpha_ratio=0.025, loss="ce"):
    best, best_value = None, None
    for _ in range(int(restarts)):
        candidate = pgd(detector, x, label, epsilon, steps=steps, alpha_ratio=alpha_ratio, loss=loss)
        value = attack_objective(detector, candidate, label, loss)
        if best_value is None or value > best_value:
            best, best_value = candidate, value
    return best


SURROGATE_CACHE = {}


def surrogate_start(surrogate, x, label, epsilon, steps, restarts, alpha_ratio):
    key = (surrogate.name, round(float(epsilon), 12), int(label[0]), hash(x.detach().cpu().numpy().tobytes()))
    if key not in SURROGATE_CACHE:
        SURROGATE_CACHE[key] = pgd_restarts(surrogate, x, label, epsilon, steps=steps, restarts=restarts, alpha_ratio=alpha_ratio).cpu()
    return SURROGATE_CACHE[key].to(x.device)


@register_attack("pgd_warm")
def pgd_warm(detector, x, label, epsilon, surrogate=None, surrogate_steps=100, surrogate_restarts=3, steps=100, alpha_ratio=0.025):
    torch = detector.torch
    if surrogate is None:
        raise ValueError("pgd_warm needs a surrogate detector")
    if getattr(detector, "spoof_index", 1) == getattr(surrogate, "spoof_index", 1):
        surrogate_label = label
    else:
        surrogate_label = 1 - label
    original = x.clone().detach()
    adversarial = surrogate_start(surrogate, original, surrogate_label, epsilon, surrogate_steps, surrogate_restarts, alpha_ratio)
    alpha = max(epsilon * alpha_ratio, epsilon / max(int(steps), 1))
    for _ in range(int(steps)):
        gradient = cross_entropy_gradient(detector, adversarial, label)
        adversarial = adversarial + alpha * gradient.sign()
        adversarial = torch.min(torch.max(adversarial, original - epsilon), original + epsilon)
        adversarial = adversarial.clamp(-1.0, 1.0).detach()
    return adversarial


@register_attack("surrogate_transfer")
def surrogate_transfer(detector, x, label, epsilon, surrogate=None, surrogate_steps=100, surrogate_restarts=3, alpha_ratio=0.025):
    return pgd_warm(detector, x, label, epsilon, surrogate=surrogate, surrogate_steps=surrogate_steps, surrogate_restarts=surrogate_restarts, steps=0, alpha_ratio=alpha_ratio)


@register_attack("pgd_margin")
def pgd_margin(detector, x, label, epsilon, steps=100, restarts=3, alpha_ratio=0.025):
    return pgd_restarts(detector, x, label, epsilon, steps=steps, restarts=restarts, alpha_ratio=alpha_ratio, loss="margin")


@register_attack("pgd")
def pgd(detector, x, label, epsilon, steps=10, alpha_ratio=0.25, loss="ce"):
    torch = detector.torch
    gradient_of = LOSS_GRADIENTS[loss]
    alpha = max(epsilon * alpha_ratio, epsilon / steps)
    original = x.clone().detach()
    adversarial = (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(-1.0, 1.0).detach()
    for _ in range(steps):
        gradient = gradient_of(detector, adversarial, label)
        adversarial = adversarial + alpha * gradient.sign()
        adversarial = torch.min(torch.max(adversarial, original - epsilon), original + epsilon)
        adversarial = adversarial.clamp(-1.0, 1.0).detach()
    return adversarial
