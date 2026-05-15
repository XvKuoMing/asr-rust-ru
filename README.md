# asr-rust

High-performance Russian speech recognition in Rust, powered by the
[GigaAM](https://github.com/salute-developers/GigaAM) model (Conformer encoder +
RNN-Transducer decoder) running on GPU via the [Burn](https://burn.dev) framework
with the LibTorch backend.

## Performance

Benchmarked on a single GPU with 11.7 seconds of audio (`example.wav`):

| Batch size | Encoder | Decode | Per-request | Throughput |
|-----------:|--------:|-------:|------------:|-----------:|
| 1          | 38ms    | 35ms   | 73ms        | 14 req/s   |
| 4          | 42ms    | 38ms   | 20ms        | 50 req/s   |
| 8          | 52ms    | 43ms   | 12ms        | 83 req/s   |
| 16         | 86ms    | 79ms   | 10ms        | 96 req/s   |
| 32         | 175ms   | 146ms  | 10ms        | 98 req/s   |

- **RTF ~0.001x** at batch 16+ (processes audio ~1000x faster than real-time)
- GPU encoder handles batches efficiently; CPU decoder threads run in parallel

## Architecture

```
Audio (WAV/any ffmpeg format)
  → ffmpeg resample to 16 kHz mono
  → Mel spectrogram (64 bins, 320-sample FFT, 160 hop)
  → Conformer Encoder (16 layers, 768 dim, 16 heads, RoPE)  [GPU]
  → Encoder projection                                       [GPU → CPU]
  → RNNT Greedy Decode (LSTM prediction + joint network)     [CPU threads]
  → SentencePiece detokenization
  → Text
```

## Prerequisites

- **Rust** 1.75+ (tested with 1.85)
- **NVIDIA GPU** with CUDA 12.x+ drivers
- **ffmpeg** (for audio loading/resampling)
- **Python 3.12+** and [uv](https://docs.astral.sh/uv/) (only for weight conversion)

## Quick start

### 1. Convert model weights

Download the GigaAM v3_e2e_rnnt checkpoint and convert to safetensors:

```bash
cd scripts
uv sync
uv run python convert_model.py
```

This creates a `weights/` directory with:
- `v3_e2e_rnnt_encoder.safetensors`
- `v3_e2e_rnnt_head.safetensors`
- `v3_e2e_rnnt_tokenizer.model`
- `config.json`

### 2. Build

The project uses LibTorch with CUDA. The `.cargo/config.toml` sets
`TORCH_CUDA_VERSION=cu128` so `tch-rs` auto-downloads the correct LibTorch
distribution on first build:

```bash
cargo build --release
```

### 3. Run

Use the `run.sh` wrapper which auto-discovers the LibTorch shared libraries:

```bash
# Single inference (batch=1, 10 iterations)
./run.sh

# Custom: ./run.sh <batch_size> <num_batches> [audio_file]
./run.sh 1 10 example.wav   # sequential baseline
./run.sh 8 10 example.wav   # batched GPU inference
./run.sh 16 5 example.wav   # higher throughput
```

**Without the wrapper** (set `LD_LIBRARY_PATH` manually):

```bash
LIB_DIR=$(find target/release/build/torch-sys-*/out/libtorch/libtorch/lib -maxdepth 0 | head -1)
LD_LIBRARY_PATH="$LIB_DIR" ./target/release/asr-rust 8 10 example.wav
```

### 4. Docker

Build and run with full GPU support:

```bash
docker build -t asr-rust .
docker run --gpus all asr-rust
```

The multi-stage Dockerfile handles everything: weight conversion, LibTorch
download, Rust compilation, and produces a minimal runtime image.

## HTTP API

### `POST /v1/audio/transcriptions`

Multipart upload: `file` (any ffmpeg-supported audio), optional `model`, optional
`stream` (`true` → Server-Sent Events). Returns JSON or an SSE stream of
`transcript.text.delta` / `transcript.text.done` events. Full reference in
[API.md](API.md).

### `GET /v1/audio/transcriptions/ws` (WebSocket)

Streaming, chunked transcription over WebSocket. Designed for browser/edge
clients that want low-latency partials plus a probability signal that can be
used directly as a VAD.

**Audio format**

- PCM16 little-endian, mono
- **16 000 Hz only** (set via `?sample_rate=16000`; other rates → HTTP 400)
- Sent as base64 inside a JSON frame, **or** as a raw binary WebSocket frame
  (binary path skips the base64 round-trip and is preferred for native clients)

**Wire protocol**

Client → Server (text frame):

```json
{ "audio": "<base64 pcm16-le>", "final": false }
```

`final: true` flushes the server's audio buffer for the current segment —
useful when your VAD has decided the speaker just finished a turn.

Server → Client (text frame):

```json
{
  "type": "delta",
  "text": "привет мир",
  "token_confidence": 0.92,
  "speech_prob": 0.81,
  "samples": 12800
}
```

| Field              | Meaning                                                                |
|--------------------|------------------------------------------------------------------------|
| `type`             | `"delta"` for a partial, `"final"` for the post-flush emission         |
| `text`             | Transcript of the buffer so far                                        |
| `token_confidence` | Mean softmax probability of emitted (non-blank) tokens, `0` if none    |
| `speech_prob`      | Mean `1 − P(blank)` over every encoder frame — use as a VAD score      |
| `samples`          | Length of the audio buffer (16 kHz samples) that produced `text`       |

Errors come back as `{"type":"error","error":"..."}` text frames; the server
keeps the socket open so the client can recover.

**Usage example (Python)**

```python
import asyncio, base64, json, websockets

async def main():
    uri = "ws://localhost:8080/v1/audio/transcriptions/ws?sample_rate=16000"
    async with websockets.connect(uri) as ws:
        with open("speech.pcm", "rb") as f:           # raw PCM16-LE @ 16 kHz
            while chunk := f.read(3200):              # 100 ms chunks
                await ws.send(json.dumps({
                    "audio": base64.b64encode(chunk).decode(),
                    "final": False,
                }))
                resp = json.loads(await ws.recv())
                if resp["speech_prob"] > 0.5:
                    print(resp["text"], f"({resp['speech_prob']:.2f})")

asyncio.run(main())
```

A fully featured runnable client lives at
[`scripts/ws_client_example.py`](scripts/ws_client_example.py) — it accepts any
WAV (resampling to 16 kHz as needed) and prints a live caption with
`speech_prob` / `token_confidence` per chunk:

```bash
pip install websockets numpy scipy
python scripts/ws_client_example.py                          # default: example.wav, ws://localhost:8080
python scripts/ws_client_example.py my.wav ws://host:8080 100  # 100 ms chunks against a remote host
```

**Server-side per-chunk log line** (one INFO line per WS chunk):

```
WS chunk=12 18ms [audio=2.60s preprocess=4ms queue=1ms encoder=11ms decode=2ms
                  batch=1 speech_prob=0.312 token_conf=0.879 chars=27 rtf=0.0069x] final=false
```

The companion **livellm-proxy** ships a `transcription_ws` client that talks
this protocol with built-in VAD — see
`~/qalby/livellm-proxy/audio_ai/livellm.py` and the proxy's README.

## Project structure

```
src/
├── main.rs           # CLI entry point with batched benchmark
├── audio.rs          # WAV loading (via ffmpeg), mel spectrogram extraction
├── model/
│   ├── mod.rs        # GigaAMASR top-level model, weight loading + key remapping
│   ├── encoder.rs    # Conformer encoder (subsampling, RoPE, attention, conv, FFN)
│   └── decoder.rs    # RNNT head (LSTM cell, prediction network, joint network)
├── decoding.rs       # CPU RNNT greedy decoder, SentencePiece tokenizer, batched projection
└── schemas.rs        # Shared type definitions
scripts/
├── convert_model.py  # Download GigaAM checkpoint + convert to safetensors
└── pyproject.toml    # Python dependencies for conversion
```

## Configuration

### `.cargo/config.toml`

Controls the LibTorch CUDA version for auto-download:

```toml
[env]
TORCH_CUDA_VERSION = "cu128"
```

Change to `"cu126"` or `"cu130"` to match your CUDA driver.

### Manual LibTorch

If you prefer a manual LibTorch installation, download it and set:

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
cargo build --release
```

## License

MIT
