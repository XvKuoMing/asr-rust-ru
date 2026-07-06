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

`convert_model.py` downloads from the Sber CDN (geo-blocked in some regions).
To use a **local / fine-tuned** checkpoint instead:

```bash
uv run python convert_finetuned.py /path/to/your.ckpt /path/to/v3_e2e_rnnt_tokenizer.model ../weights
```

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

A `model` name ending in **`-lmcorr`** (e.g. `gigaam-lmcorr`) runs the
[embedded brand-correction LM](#brand-correction-lm) on the transcript; the
response then carries the corrected `text` plus the uncorrected `raw_text`.
Any other (or absent) model name transcribes without correction.

```bash
curl -F "file=@call.wav" -F "model=gigaam-lmcorr" localhost:8080/v1/audio/transcriptions
```
```json
{
  "text":     "Доброе утро, компания Водовоз Светлана. Как я могу к вам обращаться?",
  "raw_text": "Доброе утро, компания Вадова Светлана. Как я могу к вам обращаться?",
  "usage":    { "...": "..." }
}
```

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
  "peak_speech_prob": 0.97,
  "samples": 12800
}
```

| Field              | Meaning                                                                |
|--------------------|------------------------------------------------------------------------|
| `type`             | `"delta"` for a partial, `"final"` for the post-flush emission         |
| `text`             | Transcript of the buffer so far                                        |
| `raw_text`         | Uncorrected text; present on `final` frames when `?model=...-lmcorr`   |
| `token_confidence` | Mean softmax probability of emitted (non-blank) tokens, `0` if none    |
| `speech_prob`      | Mean `1 − P(blank)` over every encoder frame — a VAD score             |
| `peak_speech_prob` | Max per-frame `1 − P(blank)` — prefer this for noise filtering         |
| `samples`          | Length of the audio buffer (16 kHz samples) that produced `text`       |

Add `&model=gigaam-lmcorr` to the URL query to brand-correct `final` frames
(deltas always stream raw for latency).

**Filtering background noise with the probability fields.** `speech_prob` is a
*mean* over frames, so a chunk that is mostly silence with one short spoken word
scores low — a high threshold will drop exactly the short answers you care
about. Use `peak_speech_prob` (max over frames) as the primary gate, or simply
treat an empty `text` as "no speech": with the anti-deletion decoder an empty
transcript already means blank confidently won every frame.

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

## Brand correction (LM)

An embedded fine-tuned **ruT5** rewrites brand names the ASR mis-heard
(`вотоввос` → `Водовоз`, `сникерс` → `Snickers`, `аквариал` → `Аква Ареал`).
It runs fully in-process via [candle](https://github.com/huggingface/candle)
— GPU (bf16) with the `corrector-cuda` feature, CPU otherwise; no Python, no
sidecar. A catalog acceptance filter keeps only edits that turn
a phonetically-close span into a real catalog brand, so correction can never
delete content or replace a genuine word.

- **Opt-in per request**: model name ending in `-lmcorr` (multipart `model`
  field on POST, `?model=` on the WebSocket). Without the suffix the LM is
  bypassed entirely.
- **Model directory** (`CORRECTOR_DIR`, default `corrector/`): a standard HF T5
  export (`model.safetensors`, `config.json`, `tokenizer.json`) plus
  `brands.txt` (one canonical brand per line). Assemble it with:

```bash
python scripts/prepare_corrector.py <trained_hf_dir> <brands.txt> corrector [common_words.txt]
```

`common_words.txt` (optional, recommended) lists corpus-frequent words from
your own transcripts; the candidate screen treats them as real words and only
windows spans containing an *unknown* word — garbled brands are non-words.
Without it the screen fires on ordinary words and correction is slower.
To run the server with your own fine-tuned ASR checkpoint instead of the CDN
download, convert it with `scripts/convert_finetuned.py <ckpt> <tokenizer.model>`.

If the directory is missing the service runs normally and `-lmcorr` requests
get HTTP 400.

**Performance** (measured end-to-end on an RTX 5090, WSL2, fine-tuned
weights; native Linux is faster per kernel launch). Correction is *windowed*:
a non-word screen (`common_words.txt`) finds garble candidates — on real call
transcripts **~85% of sentences contain none and skip the LM entirely** — and
only a small window around the best-ranked spans (max 2) goes through T5.

| request path | server time |
|---|---|
| transcription only (5–7 s audio) | 115–180 ms (RTF ≈ 0.02–0.03×) |
| + correction, screen-skip (no garble candidates) | +8–30 ms |
| + correction, one window fixed by the LM | +~130 ms |
| CPU corrector fallback (no `corrector-cuda`) | +~2 s |

Whole app VRAM: **~2.1 GiB** (fp32 Conformer encoder + LibTorch context +
bf16 T5 corrector). Quality with windowing + screen on the synthetic
garbled-brand benchmark: 58.4% fix rate, 0.5% false positives, general val
WER 11.21 → 11.18 (the corrector strictly helps). Validated on real call
audio: a garbled company name (`компания Вадова` → `компания Водовоз`) is
fixed; correct brand mentions and genuine words that merely sound brand-like
(a real `аквариум`) are never touched. The Docker image builds with
`corrector-cuda` by default; a local GPU build needs `nvcc` (CUDA 12.8+ for
Blackwell/sm_120).

## Short-audio anti-deletion

RNNT greedy decoding collapses to the empty string when the blank class wins
every encoder frame — short clips whose only content is a number were silently
dropped. The decoder now applies a length-adaptive blank penalty on short clips
and, when a first pass emits nothing but the peak per-frame non-blank
probability says the clip contains speech, re-decodes with a penalty large
enough to emit. True silence still decodes to empty. Tunables live in
`decoding::AntiDeletion` (see `src/decoding.rs`); defaults are active out of
the box and are a no-op on normal-length audio.

## Project structure

```
src/
├── main.rs           # CLI entry point with batched benchmark
├── audio.rs          # WAV loading (via ffmpeg), mel spectrogram extraction
├── model/
│   ├── mod.rs        # GigaAMASR top-level model, weight loading + key remapping
│   ├── encoder.rs    # Conformer encoder (subsampling, RoPE, attention, conv, FFN)
│   └── decoder.rs    # RNNT head (LSTM cell, prediction network, joint network)
├── decoding.rs       # CPU RNNT greedy decoder (+ anti-deletion), SentencePiece tokenizer
├── corrector.rs      # Embedded T5 brand-correction LM (candle) + catalog filter
└── schemas.rs        # Shared type definitions
scripts/
├── convert_model.py      # Download GigaAM checkpoint + convert to safetensors
├── prepare_corrector.py  # Assemble the corrector model dir (CORRECTOR_DIR)
└── pyproject.toml        # Python dependencies for conversion
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
