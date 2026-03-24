use std::f32::consts::PI;

use rustfft::{num_complex::Complex, FftPlanner};

pub const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 320;
const WIN_LENGTH: usize = 320;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 64;
const CENTER: bool = false;

pub fn load_wav(path: &str) -> Vec<f32> {
    // Use ffmpeg to decode + resample to 16kHz mono s16le, matching Python's load_audio
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-threads", "0",
            "-i", path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", &SAMPLE_RATE.to_string(),
            "-",
        ])
        .output()
        .expect("Failed to run ffmpeg. Make sure ffmpeg is installed.");

    assert!(
        output.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    output
        .stdout
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect()
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / size as f32).cos()))
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filterbank() -> Vec<Vec<f32>> {
    let fft_bins = N_FFT / 2 + 1;
    let f_max = SAMPLE_RATE as f32 / 2.0;

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * N_FFT as f32 / SAMPLE_RATE as f32)
        .collect();

    let mut filterbank = vec![vec![0.0f32; fft_bins]; N_MELS];

    for m in 0..N_MELS {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..fft_bins {
            let freq = k as f32;
            if freq >= left && freq <= center {
                filterbank[m][k] = (freq - left) / (center - left);
            } else if freq > center && freq <= right {
                filterbank[m][k] = (right - freq) / (right - center);
            }
        }
    }

    filterbank
}

pub fn extract_mel_spectrogram(samples: &[f32]) -> Vec<f32> {
    let window = hann_window(WIN_LENGTH);
    let filterbank = mel_filterbank();
    let fft_bins = N_FFT / 2 + 1;

    let signal = if CENTER {
        let pad = N_FFT / 2;
        let mut padded = vec![0.0f32; pad + samples.len() + pad];
        for i in 0..pad {
            padded[pad - 1 - i] = samples[i.min(samples.len() - 1)];
        }
        padded[pad..pad + samples.len()].copy_from_slice(samples);
        for i in 0..pad {
            let src_idx = samples.len().saturating_sub(1).saturating_sub(i);
            padded[pad + samples.len() + i] = samples[src_idx];
        }
        padded
    } else {
        samples.to_vec()
    };

    let n_frames = if CENTER {
        signal.len().saturating_sub(N_FFT) / HOP_LENGTH + 1
    } else {
        (signal.len().saturating_sub(WIN_LENGTH)) / HOP_LENGTH + 1
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // mel_spec: [n_mels, n_frames] stored row-major
    let mut mel_spec = vec![0.0f32; N_MELS * n_frames];

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_LENGTH;
        let mut buffer: Vec<Complex<f32>> = (0..N_FFT)
            .map(|i| {
                let val = if i < WIN_LENGTH && start + i < signal.len() {
                    signal[start + i] * window[i]
                } else {
                    0.0
                };
                Complex::new(val, 0.0)
            })
            .collect();

        fft.process(&mut buffer);

        // Power spectrum
        let power: Vec<f32> = buffer[..fft_bins].iter().map(|c| c.norm_sqr()).collect();

        // Apply mel filterbank + log
        for (mel_idx, filter) in filterbank.iter().enumerate() {
            let mut val: f32 = 0.0;
            for (k, &p) in power.iter().enumerate() {
                val += filter[k] * p;
            }
            val = val.clamp(1e-9, 1e9).ln();
            mel_spec[mel_idx * n_frames + frame_idx] = val;
        }
    }

    mel_spec
}

pub fn mel_spectrogram_shape(num_samples: usize) -> (usize, usize) {
    let n_frames = if CENTER {
        (num_samples + N_FFT) / HOP_LENGTH + 1
    } else {
        (num_samples.saturating_sub(WIN_LENGTH)) / HOP_LENGTH + 1
    };
    (N_MELS, n_frames)
}
