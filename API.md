# asr-rust API Reference

Complete reference for all public modules, structs, and functions.

## Table of contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Modules](#modules)
  - [`audio`](#audio)
  - [`model`](#model)
  - [`decoding`](#decoding)
- [Concurrency & batching](#concurrency--batching)
- [Thread safety](#thread-safety)
- [Integration example](#integration-example)

---

## Overview

The inference pipeline has four stages:

```
Audio bytes → audio::load_wav() → audio::extract_mel_spectrogram()
  → model.encoder.forward()  [GPU, supports batch]
  → decoding::precompute_enc_proj_batch()  [GPU→CPU transfer]
  → cpu_decoder.decode()  [CPU, one thread per sequence]
  → String
```

All types are generic over `B: burn::tensor::backend::Backend`.
The recommended backend is `burn::backend::LibTorch` with CUDA.

---

## Pipeline

### Initialization (once at startup)

```rust
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;

type Backend = LibTorch;

let device = LibTorchDevice::Cuda(0);

// 1. Load model weights (encoder + RNNT head) from safetensors
let model = model::GigaAMASR::<Backend>::load("weights", &device);

// 2. Load SentencePiece tokenizer
let tokenizer = decoding::Tokenizer::load("weights/v3_e2e_rnnt_tokenizer.model");

// 3. Extract decoder weights to CPU (done once, O(1) at inference time)
let cpu_decoder = decoding::CpuRnntDecoder::from_model(&model.head, tokenizer.blank_id());
```

### Inference (per request or batch)

```rust
use burn::tensor::{Tensor, TensorData};

// 1. Load & preprocess audio
let samples: Vec<f32> = audio::load_wav("input.wav");
let mel: Vec<f32> = audio::extract_mel_spectrogram(&samples);
let (n_mels, n_frames) = audio::mel_spectrogram_shape(samples.len());

// 2. Build tensors
let mel_tensor: Tensor<Backend, 3> =
    Tensor::from_data(TensorData::new(mel, [1, n_mels, n_frames]), &device);
let length: Tensor<Backend, 1> =
    Tensor::from_data(TensorData::from([n_frames as f32]), &device);

// 3. Encoder forward (GPU)
let (encoded, encoded_len) = model.encoder.forward(mel_tensor, length);

// 4. Project + transfer to CPU
let (enc_proj, seq_len) = decoding::precompute_enc_proj(&model.head, &encoded, &encoded_len);

// 5. Greedy decode (CPU)
let text: String = cpu_decoder.decode(&enc_proj, seq_len, &tokenizer);
```

---

## Modules

### `audio`

Audio loading and feature extraction. All preprocessing runs on CPU.

#### Constants

| Constant      | Value  | Description                          |
|---------------|--------|--------------------------------------|
| `SAMPLE_RATE` | 16000  | Expected sample rate in Hz           |

#### `audio::load_wav(path: &str) -> Vec<f32>`

Load an audio file and return mono 16kHz f32 samples normalized to `[-1, 1]`.

- Uses **ffmpeg** under the hood (any format ffmpeg supports works: wav, mp3, ogg, flac, etc.)
- Automatically resamples to 16kHz and converts to mono
- **Panics** if ffmpeg is not installed or the file cannot be decoded

#### `audio::extract_mel_spectrogram(samples: &[f32]) -> Vec<f32>`

Compute a log-mel spectrogram from raw samples.

- **Input**: `&[f32]` — mono 16kHz audio samples
- **Output**: `Vec<f32>` — flat row-major array of shape `[n_mels, n_frames]`
- Parameters: 64 mel bins, 320-sample FFT, 320 window length, 160 hop length, HTK mel scale
- Matches the GigaAM Python preprocessor exactly

#### `audio::mel_spectrogram_shape(num_samples: usize) -> (usize, usize)`

Compute the output dimensions `(n_mels, n_frames)` for a given number of input samples
without actually computing the spectrogram. Useful for pre-allocating tensors.

---

### `model`

Neural network definitions using the Burn framework.

#### Constants (model hyperparameters)

| Constant              | Value | Description                             |
|-----------------------|-------|-----------------------------------------|
| `FEAT_IN`             | 64    | Input mel spectrogram bins              |
| `N_LAYERS`            | 16    | Number of conformer layers              |
| `D_MODEL`             | 768   | Model hidden dimension                  |
| `SUBSAMPLING_FACTOR`  | 4     | Temporal downsampling factor            |
| `FF_EXPANSION_FACTOR` | 4     | Feed-forward expansion multiplier       |
| `N_HEADS`             | 16    | Number of attention heads               |
| `CONV_KERNEL_SIZE`    | 5     | Conformer convolution kernel size       |
| `SUBS_KERNEL_SIZE`    | 5     | Subsampling convolution kernel size     |
| `POS_EMB_MAX_LEN`    | 5000  | Maximum sequence length for RoPE        |
| `NUM_CLASSES`         | 1025  | Vocabulary size (1024 tokens + 1 blank) |
| `PRED_HIDDEN`         | 320   | RNNT prediction network hidden size     |
| `JOINT_HIDDEN`        | 320   | RNNT joint network hidden size          |

#### `model::GigaAMASR<B: Backend>`

Top-level ASR model combining the encoder and RNNT head.

**Fields** (public):
- `encoder: ConformerEncoder<B>` — the acoustic encoder
- `head: RNNTHead<B>` — the RNNT decoder head

**Methods**:

##### `GigaAMASR::new(device: &Device<B>) -> Self`

Create a new model with randomly initialized weights.

##### `GigaAMASR::load(weights_dir: &str, device: &Device<B>) -> Self`

Load a model from safetensors files. Expects the following files in `weights_dir`:
- `v3_e2e_rnnt_encoder.safetensors`
- `v3_e2e_rnnt_head.safetensors`

Handles key remapping for LSTM and joint network weights automatically.

---

#### `model::encoder::ConformerEncoder<B: Backend>`

16-layer Conformer encoder with rotary position embeddings.

##### `ConformerEncoder::forward(audio_signal, length) -> (encoded, encoded_len)`

| Parameter      | Type              | Shape           | Description                            |
|----------------|-------------------|-----------------|----------------------------------------|
| `audio_signal` | `Tensor<B, 3>`   | `[B, n_mels, T]` | Mel spectrogram batch                |
| `length`       | `Tensor<B, 1>`   | `[B]`           | Number of valid frames per sample      |
| **returns** `encoded`     | `Tensor<B, 3>` | `[B, d_model, T']` | Encoder output (subsampled)  |
| **returns** `encoded_len` | `Tensor<B, 1>` | `[B]`              | Valid output lengths         |

- **Supports batch > 1**: pass multiple mel spectrograms stacked on dim 0
- Automatically builds padding and attention masks when `B > 1`
- For variable-length inputs, pad to the longest sequence and provide true lengths

Internal architecture per layer:
1. Feed-forward (half-step residual)
2. Multi-head self-attention with rotary position embeddings (RoPE)
3. Depthwise convolution with GLU gating
4. Feed-forward (half-step residual)
5. Layer normalization

---

#### `model::decoder::RNNTHead<B: Backend>`

RNNT head containing the prediction network and joint network.

**Fields** (public):
- `decoder: RNNTDecoder<B>` — prediction network (embedding + LSTM)
- `joint: RNNTJoint<B>` — joint network (encoder proj + pred proj + output)

##### `RNNTDecoder::predict(x, h, c) -> (g, h_new, c_new)`

GPU-side prediction step (used for full RNNT training/loss, not used in fast inference path).

| Parameter | Type                         | Description                      |
|-----------|------------------------------|----------------------------------|
| `x`       | `Option<Tensor<B, 2, Int>>` | Previous token IDs or None       |
| `h`       | `Tensor<B, 2>`              | LSTM hidden state `[B, hidden]`  |
| `c`       | `Tensor<B, 2>`              | LSTM cell state `[B, hidden]`    |

##### `RNNTJoint::joint(encoder_out, decoder_out) -> Tensor<B, 4>`

GPU-side joint network (used for full RNNT training/loss).

| Parameter     | Type            | Shape              | Description          |
|---------------|-----------------|--------------------|----------------------|
| `encoder_out` | `Tensor<B, 3>` | `[B, T, enc_dim]`  | Encoder features     |
| `decoder_out` | `Tensor<B, 3>` | `[B, U, pred_dim]` | Prediction features  |
| **returns**   | `Tensor<B, 4>` | `[B, T, U, vocab]` | Log-softmax logits   |

---

### `decoding`

Tokenization and RNNT greedy decoding.

#### `decoding::Tokenizer`

Minimal SentencePiece tokenizer (protobuf parser, no external dependencies).

##### `Tokenizer::load(model_path: &str) -> Self`

Load a SentencePiece `.model` file.

##### `Tokenizer::decode(token_ids: &[usize]) -> String`

Convert a sequence of token IDs back to a UTF-8 string.
Handles the SentencePiece `▁` (U+2581) word boundary marker.

##### `Tokenizer::blank_id() -> usize`

Returns the blank token ID (= vocab_size = 1024).

---

#### `decoding::CpuRnntDecoder`

High-performance RNNT greedy decoder running entirely on CPU.
All weights are extracted from the GPU model once at construction and held as flat `f32` vectors.

##### `CpuRnntDecoder::from_model<B: Backend>(head: &RNNTHead<B>, blank_id: usize) -> Self`

Extract all decoder weights from the GPU model. Call once at startup.

##### `CpuRnntDecoder::decode(&self, enc_proj: &[f32], seq_len: usize, tokenizer: &Tokenizer) -> String`

Run RNNT greedy decoding on pre-computed encoder projections.

| Parameter   | Type            | Description                                            |
|-------------|-----------------|--------------------------------------------------------|
| `enc_proj`  | `&[f32]`        | Flat row-major `[seq_len, joint_hidden]` from GPU proj |
| `seq_len`   | `usize`         | Number of valid encoder time steps                     |
| `tokenizer` | `&Tokenizer`    | Tokenizer for ID → text conversion                     |
| **returns** | `String`        | Decoded transcription                                  |

- **Thread-safe**: `&self` is immutable; can be called from multiple threads simultaneously
- Max 10 symbols per encoder time step (prevents infinite loops on adversarial input)
- Includes manual LSTM cell, matrix-vector products, and argmax — no GPU calls

---

#### `decoding::precompute_enc_proj<B>(head, encoded, encoded_len) -> (Vec<f32>, usize)`

Single-sequence encoder projection on GPU, transferred to CPU.

| Parameter     | Type            | Shape               | Description              |
|---------------|-----------------|---------------------|--------------------------|
| `head`        | `&RNNTHead<B>` | —                   | The RNNT head            |
| `encoded`     | `&Tensor<B, 3>` | `[1, d_model, T']` | Encoder output           |
| `encoded_len` | `&Tensor<B, 1>` | `[1]`              | Valid length             |
| **returns**   | `(Vec<f32>, usize)` | —              | `(flat_projection, len)` |

#### `decoding::precompute_enc_proj_batch<B>(head, encoded, encoded_len) -> Vec<(Vec<f32>, usize)>`

Batched encoder projection — runs the linear layer once for the entire batch on GPU,
then splits into per-sequence CPU data.

| Parameter     | Type            | Shape               | Description              |
|---------------|-----------------|---------------------|--------------------------|
| `head`        | `&RNNTHead<B>` | —                   | The RNNT head            |
| `encoded`     | `&Tensor<B, 3>` | `[B, d_model, T']` | Batched encoder output   |
| `encoded_len` | `&Tensor<B, 1>` | `[B]`              | Per-sequence lengths     |
| **returns**   | `Vec<(Vec<f32>, usize)>` | —          | Per-sequence projections |

---

## Concurrency & batching

### Architecture

```
                        ┌──────────────────────┐
  Request pool ───────► │  Batch accumulator    │
  (audio files)         │  (pad to max length)  │
                        └──────────┬───────────┘
                                   │ [B, 64, T_max]
                        ┌──────────▼───────────┐
                        │  GPU Encoder          │
                        │  (single forward pass)│
                        └──────────┬───────────┘
                                   │ [B, 768, T']
                        ┌──────────▼───────────┐
                        │  GPU Enc Projection   │
                        │  (batched linear)     │
                        └──────────┬───────────┘
                                   │ split per sequence
                     ┌─────────────┼─────────────┐
                     │             │             │
              ┌──────▼──┐   ┌─────▼───┐   ┌─────▼───┐
              │ Thread 1 │   │ Thread 2 │   │ Thread N │
              │ CPU RNNT │   │ CPU RNNT │   │ CPU RNNT │
              │ decode   │   │ decode   │   │ decode   │
              └──────┬──┘   └─────┬───┘   └─────┬───┘
                     │            │             │
                     └─────────── ▼ ────────────┘
                              Vec<String>
```

### Batched inference

Stack multiple mel spectrograms into a single tensor and run the encoder once:

```rust
let mel_batch: Tensor<Backend, 3> = Tensor::cat(mel_tensors, 0); // [B, 64, T_max]
let len_batch: Tensor<Backend, 1> = Tensor::cat(len_tensors, 0); // [B]

let (encoded, encoded_len) = model.encoder.forward(mel_batch, len_batch);
let per_seq = decoding::precompute_enc_proj_batch(&model.head, &encoded, &encoded_len);
```

For variable-length inputs, **pad shorter spectrograms with zeros** to `T_max` and set each
element of `len_batch` to the true frame count. The encoder masks padded positions internally.

### Parallel CPU decoding

After the GPU stages, decode each sequence on a separate CPU thread:

```rust
let texts: Vec<String> = std::thread::scope(|s| {
    let handles: Vec<_> = per_seq
        .iter()
        .map(|(proj, len)| {
            let dec = &cpu_decoder;
            let tok = &tokenizer;
            s.spawn(move || dec.decode(proj, *len, tok))
        })
        .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
});
```

### Thread safety summary

| Component           | `Send` | `Sync` | Notes                                          |
|---------------------|--------|--------|-------------------------------------------------|
| `GigaAMASR`        | Yes    | Yes    | GPU model; use `Arc` for shared access          |
| `ConformerEncoder`  | Yes    | Yes    | Forward pass needs `&self` only                 |
| `Tokenizer`         | Yes    | Yes    | Immutable after load                            |
| `CpuRnntDecoder`    | Yes    | Yes    | `decode(&self, ..)` — no mutation, fully re-entrant |

### Recommended concurrency pattern for a web service

```rust
use std::sync::Arc;

let model = Arc::new(model::GigaAMASR::<Backend>::load("weights", &device));
let tokenizer = Arc::new(decoding::Tokenizer::load("weights/v3_e2e_rnnt_tokenizer.model"));
let cpu_decoder = Arc::new(decoding::CpuRnntDecoder::from_model(&model.head, tokenizer.blank_id()));

// In request handler (e.g., Actix-web):
// - Accumulate requests into batches (e.g., collect up to N or timeout after T ms)
// - Run batched encoder + projection on GPU (serialized via mutex or channel)
// - Fan out CPU decode across threads
// - Return individual results
```

### Choosing batch size

| Batch size | GPU utilization | Decode threads | Best for                        |
|------------|-----------------|----------------|---------------------------------|
| 1          | Low             | 1              | Lowest latency, single request  |
| 4–8        | Good            | 4–8            | Balanced latency and throughput |
| 16         | High            | 16             | High throughput                 |
| 32+        | Saturated       | 32+            | Maximum throughput (CPU-bound)  |

Beyond batch 16, throughput gains flatten as CPU decode becomes the bottleneck.

---

## Integration example

Minimal end-to-end example for a web service handler:

```rust
fn transcribe_batch(
    audio_paths: &[&str],
    model: &model::GigaAMASR<Backend>,
    cpu_decoder: &decoding::CpuRnntDecoder,
    tokenizer: &decoding::Tokenizer,
    device: &LibTorchDevice,
) -> Vec<String> {
    // 1. Preprocess all audio files
    let mut mels = Vec::new();
    let mut lengths = Vec::new();
    let mut max_frames = 0usize;

    for path in audio_paths {
        let samples = audio::load_wav(path);
        let mel = audio::extract_mel_spectrogram(&samples);
        let (n_mels, n_frames) = audio::mel_spectrogram_shape(samples.len());
        max_frames = max_frames.max(n_frames);
        mels.push((mel, n_mels, n_frames));
        lengths.push(n_frames as f32);
    }

    // 2. Pad and stack into batch tensors
    let batch_size = mels.len();
    let n_mels = mels[0].1;
    let mut batch_data = vec![0.0f32; batch_size * n_mels * max_frames];

    for (b, (mel, _, n_frames)) in mels.iter().enumerate() {
        for m in 0..n_mels {
            let src_off = m * n_frames;
            let dst_off = b * n_mels * max_frames + m * max_frames;
            batch_data[dst_off..dst_off + n_frames]
                .copy_from_slice(&mel[src_off..src_off + n_frames]);
        }
    }

    let mel_tensor: Tensor<Backend, 3> = Tensor::from_data(
        TensorData::new(batch_data, [batch_size, n_mels, max_frames]),
        device,
    );
    let len_tensor: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::from(lengths.as_slice()), device);

    // 3. GPU: encoder + projection (single batched call)
    let (encoded, encoded_len) = model.encoder.forward(mel_tensor, len_tensor);
    let per_seq = decoding::precompute_enc_proj_batch(&model.head, &encoded, &encoded_len);

    // 4. CPU: parallel decode
    std::thread::scope(|s| {
        let handles: Vec<_> = per_seq
            .iter()
            .map(|(proj, len)| s.spawn(|| cpu_decoder.decode(proj, *len, tokenizer)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}
```
