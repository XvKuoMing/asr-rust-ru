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
