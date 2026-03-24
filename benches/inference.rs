use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

use asr_rust::{audio, decoding, model};

type Backend = LibTorch;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = (p / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64)
    }
}

fn print_stats(label: &str, values: &mut [f64]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = values.iter().sum::<f64>() / values.len() as f64;
    println!(
        "║  {:<10} │ avg {:>7.1}ms │ p50 {:>7.1}ms │ p95 {:>7.1}ms │ min {:>7.1}ms │ max {:>7.1}ms",
        label,
        avg,
        percentile(values, 50.0),
        percentile(values, 95.0),
        values[0],
        values[values.len() - 1],
    );
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();

    let batch_size: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1);
    let num_batches: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let audio_path = args.get(2).map(|s| s.as_str()).unwrap_or("example.wav");

    let total_requests = batch_size * num_batches;
    let device = LibTorchDevice::Cuda(0);

    println!("Loading model...");
    let t = Instant::now();
    let model = model::GigaAMASR::<Backend>::load("weights", &device);
    println!("Model loaded in {}ms", t.elapsed().as_millis());

    println!("Loading tokenizer...");
    let tokenizer = decoding::Tokenizer::load("weights/v3_e2e_rnnt_tokenizer.model");

    println!("Preparing CPU decoder...");
    let cpu_decoder = decoding::CpuRnntDecoder::from_model(&model.head, tokenizer.blank_id());

    println!("Loading audio: {}", audio_path);
    let samples = audio::load_wav(audio_path);
    let num_samples = samples.len();
    let audio_dur_s = num_samples as f64 / audio::SAMPLE_RATE as f64;
    println!(
        "Loaded {} samples ({:.2}s @ {}Hz)",
        num_samples, audio_dur_s, audio::SAMPLE_RATE
    );

    println!("Extracting mel spectrogram...");
    let mel = audio::extract_mel_spectrogram(&samples);
    let (n_mels, n_frames) = audio::mel_spectrogram_shape(num_samples);
    println!("Mel spectrogram: {}×{}", n_mels, n_frames);

    let mel_single: Tensor<Backend, 3> =
        Tensor::from_data(TensorData::new(mel, [1, n_mels, n_frames]), &device);
    let mel_batch: Tensor<Backend, 3> = mel_single.repeat_dim(0, batch_size);

    let len_single: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::from([n_frames as f32]), &device);
    let len_batch: Tensor<Backend, 1> = len_single.repeat_dim(0, batch_size);

    println!("Warmup (batch_size={})...", batch_size);
    {
        let dummy_mel: Tensor<Backend, 3> = Tensor::zeros([batch_size, n_mels, 320], &device);
        let dummy_len: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::from([20.0f32]), &device).repeat_dim(0, batch_size);
        let _ = model.encoder.forward(dummy_mel, dummy_len);
    }

    println!(
        "\nBenchmark: {} batches × {} requests/batch = {} total requests\n",
        num_batches, batch_size, total_requests
    );

    let mut batch_enc_ms = Vec::with_capacity(num_batches);
    let mut batch_proj_ms = Vec::with_capacity(num_batches);
    let mut batch_dec_ms = Vec::with_capacity(num_batches);
    let mut batch_total_ms = Vec::with_capacity(num_batches);
    let mut last_text = String::new();

    let wall_start = Instant::now();

    for i in 0..num_batches {
        let t0 = Instant::now();
        let (encoded, encoded_len) =
            model.encoder.forward(mel_batch.clone(), len_batch.clone());
        let _ = encoded.clone().into_data();
        let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let per_seq =
            decoding::precompute_enc_proj_batch(&model.head, &encoded, &encoded_len);
        let proj_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let texts: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = per_seq
                .iter()
                .map(|(enc_proj, seq_len)| {
                    let dec = &cpu_decoder;
                    let tok = &tokenizer;
                    s.spawn(move || dec.decode(enc_proj, *seq_len, tok))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let dec_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let total = enc_ms + proj_ms + dec_ms;

        println!(
            "  [{:>3}/{}]  encoder {:>6.1}ms │ proj {:>5.1}ms │ decode {:>6.1}ms │ batch {:>6.1}ms │ per-req {:>5.1}ms",
            i + 1, num_batches, enc_ms, proj_ms, dec_ms, total, total / batch_size as f64,
        );

        batch_enc_ms.push(enc_ms);
        batch_proj_ms.push(proj_ms);
        batch_dec_ms.push(dec_ms);
        batch_total_ms.push(total);
        last_text = texts.into_iter().next().unwrap_or_default();
    }

    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let mut req_latencies: Vec<f64> =
        batch_total_ms.iter().map(|t| t / batch_size as f64).collect();

    let avg_batch = batch_total_ms.iter().sum::<f64>() / num_batches as f64;
    let avg_req = avg_batch / batch_size as f64;

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Benchmark: {} batches × {} req/batch = {} requests",
        num_batches, batch_size, total_requests
    );
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Per-batch latency:");
    print_stats("Encoder", &mut batch_enc_ms);
    print_stats("Projection", &mut batch_proj_ms);
    print_stats("Decode", &mut batch_dec_ms);
    print_stats("Total", &mut batch_total_ms);
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Per-request (amortized over batch):");
    print_stats("Latency", &mut req_latencies);
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Wall time         : {:.0}ms", wall_ms);
    println!(
        "║  Throughput        : {:.1} req/sec",
        total_requests as f64 / (wall_ms / 1000.0)
    );
    println!(
        "║  Avg batch latency : {:.1}ms  ({:.1}ms per request)",
        avg_batch, avg_req
    );
    println!(
        "║  RTF               : {:.4}x  (per-request latency / audio duration)",
        avg_req / 1000.0 / audio_dur_s
    );
    println!("║  Audio             : {:.2}s  ({})", audio_dur_s, audio_path);
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Transcription     : {}", last_text);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}
