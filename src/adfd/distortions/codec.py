import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..audio.io import load_audio, save_audio
from .base import register

FFMPEG = shutil.which("ffmpeg")
CONTAINER = {"mp3": "mp3", "opus": "opus", "aac": "m4a", "g711a": "wav", "g711u": "wav", "g722": "wav", "amrnb": "amr"}
ENCODER = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac", "g711a": "pcm_alaw", "g711u": "pcm_mulaw", "g722": "g722", "amrnb": "libopencore_amrnb"}
FIXED_RATE = {"g711a", "g711u", "g722"}
NARROWBAND = {"amrnb"}


def _run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _match_length(signal, length):
    if len(signal) >= length:
        return signal[:length]
    return np.pad(signal, (0, length - len(signal)))


def _encode_command(source, encoded, codec, level):
    command = [FFMPEG, "-y", "-i", str(source), "-c:a", ENCODER[codec]]
    if codec in NARROWBAND:
        command += ["-ar", "8000", "-ac", "1", "-b:a", f"{int(level)}k"]
    elif codec not in FIXED_RATE:
        command += ["-b:a", f"{int(level)}k"]
    return command + [str(encoded)]


@register("codec")
def transcode(signal, sr, level, codec="mp3"):
    if codec not in ENCODER:
        raise ValueError(f"unsupported codec '{codec}'")
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH; codec distortion requires ffmpeg")
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "in.wav"
        encoded = Path(directory) / f"enc.{CONTAINER[codec]}"
        decoded = Path(directory) / "out.wav"
        save_audio(source, signal, sr)
        _run(_encode_command(source, encoded, codec, level))
        _run([FFMPEG, "-y", "-i", str(encoded), "-ar", str(sr), "-ac", "1", str(decoded)])
        out, _ = load_audio(decoded, sr)
    out = _match_length(out, len(signal))
    return out.astype(np.float32), {"bitrate_kbps": int(level), "codec": codec, "applied": True}
