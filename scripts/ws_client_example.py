#!/usr/bin/env python3
"""
Stream a WAV file over the /v1/audio/transcriptions/ws WebSocket endpoint.

The endpoint accepts JSON frames `{"audio": "<base64 pcm16-le>", "final": bool}`
and replies with `{"type", "text", "token_confidence", "speech_prob", "samples"}`
per chunk.

The model expects 16 kHz mono PCM16. Anything else is resampled in-process via
scipy. Set `final: true` on the last chunk to make the server flush its buffer.

Usage:
    python ws_client_example.py                       # uses ../example.wav
    python ws_client_example.py path/to/audio.wav
    python ws_client_example.py audio.wav ws://host:8080
    python ws_client_example.py audio.wav ws://host:8080 100   # 100 ms chunks

Dependencies (install once):
    pip install websockets numpy scipy
"""
import argparse
import asyncio
import base64
import json
import sys
import wave
from math import gcd
from pathlib import Path

import numpy as np
import websockets
from scipy.signal import resample_poly


TARGET_SR = 16_000


def load_pcm16_mono_16k(wav_path: Path) -> bytes:
    """Read a WAV file and return raw PCM16-LE mono samples at 16 kHz."""
    with wave.open(str(wav_path), "rb") as w:
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    if sampwidth != 2:
        raise SystemExit(
            f"unsupported sample width {sampwidth*8} bit (need 16-bit PCM)"
        )
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nchannels > 1:
        samples = samples.reshape(-1, nchannels).mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        samples = resample_poly(samples, TARGET_SR // g, sr // g)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767).astype(np.int16).tobytes()


async def stream(wav_path: Path, base_url: str, chunk_ms: int) -> None:
    pcm = load_pcm16_mono_16k(wav_path)
    n_samples = len(pcm) // 2
    duration_s = n_samples / TARGET_SR
    chunk_bytes = (TARGET_SR * chunk_ms // 1000) * 2

    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions/ws?sample_rate={TARGET_SR}"
    print(
        f"connecting → {url}\n"
        f"audio: {wav_path}  duration={duration_s:.2f}s  chunk={chunk_ms}ms",
        flush=True,
    )

    async with websockets.connect(url, max_size=None) as ws:
        offsets = list(range(0, len(pcm), chunk_bytes))
        for i, off in enumerate(offsets):
            chunk = pcm[off : off + chunk_bytes]
            is_last = (off + chunk_bytes) >= len(pcm)
            await ws.send(
                json.dumps(
                    {
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "final": is_last,
                    }
                )
            )
            resp = json.loads(await ws.recv())
            kind = resp.get("type")
            sp = resp.get("speech_prob", 0.0)
            tc = resp.get("token_confidence", 0.0)
            text = resp.get("text", "")
            # In-place updating line, like a streaming caption.
            sys.stdout.write(
                f"\r[{i+1:03d}/{len(offsets):03d}] {kind:5s} "
                f"speech={sp:.2f} conf={tc:.2f} | {text[-90:]:<90}"
            )
            sys.stdout.flush()
            if kind == "final":
                print(f"\n\nFINAL: {text}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "wav",
        nargs="?",
        default=str(Path(__file__).parent.parent / "example.wav"),
        help="WAV file to stream (default: ../example.wav)",
    )
    p.add_argument(
        "base_url",
        nargs="?",
        default="ws://localhost:8080",
        help="ASR server base URL, ws:// or wss:// (default: ws://localhost:8080)",
    )
    p.add_argument(
        "chunk_ms",
        nargs="?",
        type=int,
        default=200,
        help="Chunk size in milliseconds (default: 200)",
    )
    args = p.parse_args()
    asyncio.run(stream(Path(args.wav), args.base_url, args.chunk_ms))


if __name__ == "__main__":
    main()
