import numpy as np

from .base import register, rng_for


@register("transmission")
def packet_loss(signal, sr, level, packet_ms=20, conceal="hold", salt=0):
    rng = rng_for(level, len(signal), salt)
    packet = max(1, int(sr * packet_ms / 1000))
    out = signal.copy()
    packets = len(signal) // packet
    dropped = rng.random(packets) < float(level)
    for index in range(packets):
        if not dropped[index]:
            continue
        start = index * packet
        end = start + packet
        if conceal == "hold" and start - packet >= 0:
            out[start:end] = out[start - packet:start][: end - start]
        else:
            out[start:end] = 0.0
    achieved = float(np.mean(dropped)) if packets else 0.0
    return out.astype(np.float32), {
        "packet_loss_rate": float(level),
        "achieved_loss_rate": achieved,
        "packet_ms": packet_ms,
    }
