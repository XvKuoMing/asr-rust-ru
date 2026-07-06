use actix_multipart::Multipart;
use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer};
use base64::Engine as _;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;
use burn::tensor::{Tensor, TensorData};
use envconfig::Envconfig;
use futures::TryStreamExt;
use futures_util::StreamExt;
use serde::Deserialize;
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use asr_rust::config::AppConfig;
use asr_rust::schemas::*;
use asr_rust::{audio, corrector, decoding, model};

type Backend = LibTorch;

// ---------------------------------------------------------------------------
// Batch engine types
// ---------------------------------------------------------------------------

struct EncoderRequest {
    mel: Vec<f32>,
    n_mels: usize,
    n_frames: usize,
    submitted_at: Instant,
    reply: tokio::sync::oneshot::Sender<EncoderResult>,
}

struct EncoderResult {
    enc_proj: Vec<f32>,
    seq_len: usize,
    queue_ms: f64,
    encoder_ms: f64,
    batch_size: usize,
}

// ---------------------------------------------------------------------------
// Application state (shared across all handlers via Arc)
// ---------------------------------------------------------------------------

struct AppState {
    encoder_tx: tokio::sync::mpsc::Sender<EncoderRequest>,
    tokenizer: decoding::Tokenizer,
    cpu_decoder: decoding::CpuRnntDecoder,
    /// Embedded brand-correction LM; `None` when no corrector model directory
    /// was provided. Mutex because T5 generation mutates its kv-cache.
    corrector: Option<std::sync::Mutex<corrector::Corrector>>,
}

impl AppState {
    /// Does the requested `model` name opt into LM brand correction?
    fn wants_lmcorr(model: Option<&str>) -> bool {
        model.map(|m| m.trim().ends_with("-lmcorr")).unwrap_or(false)
    }

    /// Run brand correction; returns the corrected text, or the input as-is
    /// if the corrector is unavailable.
    fn correct(&self, text: &str) -> String {
        match &self.corrector {
            Some(c) => c.lock().expect("corrector mutex poisoned").correct(text),
            None => text.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Batch inference engine — runs on a dedicated OS thread, owns the model
// ---------------------------------------------------------------------------

/// Estimate GPU memory (bytes) consumed by a batch.
/// Accounts for input tensor plus intermediate activations across 16 conformer layers.
fn estimate_batch_bytes(batch_size: usize, n_mels: usize, max_frames: usize) -> usize {
    const ACTIVATION_MULTIPLIER: usize = 20;
    batch_size * n_mels * max_frames * std::mem::size_of::<f32>() * ACTIVATION_MULTIPLIER
}

fn batch_engine(
    mut rx: tokio::sync::mpsc::Receiver<EncoderRequest>,
    model: model::GigaAMASR<Backend>,
    device: LibTorchDevice,
    max_batch: usize,
    batch_timeout: Duration,
    max_vram_bytes: usize,
) {
    log::info!(
        "Batch engine ready: max_batch={}, timeout={}ms, max_vram={}MiB",
        max_batch,
        batch_timeout.as_millis(),
        max_vram_bytes / (1024 * 1024),
    );

    loop {
        let first = match rx.blocking_recv() {
            Some(req) => req,
            None => {
                log::info!("Batch engine shutting down");
                return;
            }
        };

        let mut batch = vec![first];

        if max_batch > 1 {
            std::thread::sleep(batch_timeout);
            while batch.len() < max_batch {
                match rx.try_recv() {
                    Ok(req) => batch.push(req),
                    Err(_) => break,
                }
            }
        }

        // VRAM-aware trimming: sort by n_frames ascending, drop longest first
        let n_mels = batch[0].n_mels;
        batch.sort_by_key(|r| r.n_frames);
        let mut fit_count = batch.len();
        while fit_count > 1 {
            let max_frames = batch[..fit_count].iter().map(|r| r.n_frames).max().unwrap();
            if estimate_batch_bytes(fit_count, n_mels, max_frames) <= max_vram_bytes {
                break;
            }
            fit_count -= 1;
        }
        let dropped: Vec<_> = batch.drain(fit_count..).collect();
        if !dropped.is_empty() {
            log::warn!(
                "VRAM budget exceeded: trimmed batch from {} to {} ({} dropped)",
                fit_count + dropped.len(),
                fit_count,
                dropped.len(),
            );
        }

        let batch_size = batch.len();
        let max_frames = batch.iter().map(|r| r.n_frames).max().unwrap();

        log::info!(
            "Batch: {} reqs, max_frames={}, estimated_vram={:.1}MiB",
            batch_size,
            max_frames,
            estimate_batch_bytes(batch_size, n_mels, max_frames) as f64 / (1024.0 * 1024.0),
        );

        // Process batch with panic recovery
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let mut padded = vec![0.0f32; batch_size * n_mels * max_frames];
            let mut lengths = Vec::with_capacity(batch_size);

            for (i, req) in batch.iter().enumerate() {
                let base = i * n_mels * max_frames;
                for m in 0..n_mels {
                    let src = &req.mel[m * req.n_frames..(m + 1) * req.n_frames];
                    let dst = base + m * max_frames;
                    padded[dst..dst + req.n_frames].copy_from_slice(src);
                }
                lengths.push(req.n_frames as f32);
            }

            let mel_tensor: Tensor<Backend, 3> = Tensor::from_data(
                TensorData::new(padded, [batch_size, n_mels, max_frames]),
                &device,
            );
            let len_tensor: Tensor<Backend, 1> = Tensor::from_data(
                TensorData::new(lengths, [batch_size]),
                &device,
            );

            let t_enc = Instant::now();
            let (encoded, encoded_len) = model.encoder.forward(mel_tensor, len_tensor);
            let per_seq =
                decoding::precompute_enc_proj_batch(&model.head, &encoded, &encoded_len);
            let encoder_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

            log::info!(
                "Encoder done: batch={}, encoder={:.0}ms ({:.1}ms/req)",
                batch_size,
                encoder_ms,
                encoder_ms / batch_size as f64,
            );

            for (req, (enc_proj, seq_len)) in batch.into_iter().zip(per_seq) {
                let elapsed = req.submitted_at.elapsed().as_secs_f64() * 1000.0;
                let _ = req.reply.send(EncoderResult {
                    enc_proj,
                    seq_len,
                    queue_ms: (elapsed - encoder_ms).max(0.0),
                    encoder_ms,
                    batch_size,
                });
            }
        }));

        if let Err(payload) = result {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            log::error!("Batch engine panic (recovering): {}", msg);
        }
    }
}

// ---------------------------------------------------------------------------
// POST /v1/audio/transcriptions
// ---------------------------------------------------------------------------

#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    request_body(content_type = "multipart/form-data", content = TranscribeRequest),
    responses(
        (status = 200, description = "Transcription result", body = TranscriptionResponse),
        (status = 400, description = "Missing audio file"),
    ),
    tag = "Audio"
)]
async fn transcribe(
    state: web::Data<AppState>,
    req: actix_web::HttpRequest,
    mut payload: Multipart,
) -> actix_web::Result<HttpResponse> {
    let request_start = Instant::now();
    let peer = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "-".into());

    // ── Parse multipart fields ──────────────────────────────────────
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut stream_mode = false;
    let mut model_name: Option<String> = None;

    while let Some(mut field) = payload.try_next().await? {
        let name = field.name().unwrap_or("").to_string();
        let mut buf = Vec::new();
        while let Some(chunk) = field.try_next().await? {
            buf.extend_from_slice(&chunk);
        }
        match name.as_str() {
            "file" => audio_bytes = Some(buf),
            "stream" => {
                let val = String::from_utf8_lossy(&buf);
                stream_mode = matches!(val.trim(), "true" | "True" | "1");
            }
            "model" => model_name = Some(String::from_utf8_lossy(&buf).trim().to_string()),
            _ => {}
        }
    }

    let audio_bytes =
        audio_bytes.ok_or_else(|| actix_web::error::ErrorBadRequest("missing 'file' field"))?;
    let upload_size = audio_bytes.len();

    let lmcorr = AppState::wants_lmcorr(model_name.as_deref());
    if lmcorr && state.corrector.is_none() {
        return Err(actix_web::error::ErrorBadRequest(
            "model '-lmcorr' requested but no corrector model is loaded \
             (set CORRECTOR_DIR to a valid model directory)",
        ));
    }

    log::info!(
        "{} POST /v1/audio/transcriptions stream={} lmcorr={} size={}B",
        peer,
        stream_mode,
        lmcorr,
        upload_size,
    );

    // ── Write to temp file (ffmpeg needs a path) ────────────────────
    let mut tmp =
        tempfile::NamedTempFile::new().map_err(actix_web::error::ErrorInternalServerError)?;
    tmp.write_all(&audio_bytes)
        .map_err(actix_web::error::ErrorInternalServerError)?;
    let tmp_path = tmp.path().to_string_lossy().to_string();

    // ── Phase 1: preprocess audio (CPU-bound) ───────────────────────
    let (mel, n_mels, n_frames, audio_dur_s, preprocess_ms) = web::block(move || {
        let t0 = Instant::now();
        let samples = audio::load_wav(&tmp_path);
        let num_samples = samples.len();
        let audio_dur_s = num_samples as f64 / audio::SAMPLE_RATE as f64;
        let mel = audio::extract_mel_spectrogram(&samples);
        let (n_mels, n_frames) = audio::mel_spectrogram_shape(num_samples);
        let preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;
        (mel, n_mels, n_frames, audio_dur_s, preprocess_ms)
    })
    .await
    .map_err(actix_web::error::ErrorInternalServerError)?;

    // ── Phase 2: submit to batch engine, await result ───────────────
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    state
        .encoder_tx
        .send(EncoderRequest {
            mel,
            n_mels,
            n_frames,
            submitted_at: Instant::now(),
            reply: reply_tx,
        })
        .await
        .map_err(|_| actix_web::error::ErrorInternalServerError("batch engine unavailable"))?;

    let enc = reply_rx
        .await
        .map_err(|_| actix_web::error::ErrorInternalServerError("batch engine error"))?;

    // ── Phase 3: decode ─────────────────────────────────────────────
    if stream_mode {
        handle_streaming(state, enc, audio_dur_s, preprocess_ms, peer, request_start, lmcorr).await
    } else {
        handle_non_streaming(
            state,
            enc,
            audio_dur_s,
            preprocess_ms,
            n_frames,
            peer,
            request_start,
            lmcorr,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Non-streaming handler
// ---------------------------------------------------------------------------

async fn handle_non_streaming(
    state: web::Data<AppState>,
    enc: EncoderResult,
    audio_dur_s: f64,
    preprocess_ms: f64,
    n_frames: usize,
    peer: String,
    request_start: Instant,
    lmcorr: bool,
) -> actix_web::Result<HttpResponse> {
    let result = web::block(move || {
        let t = Instant::now();
        let text = state
            .cpu_decoder
            .decode(&enc.enc_proj, enc.seq_len, &state.tokenizer);
        let decode_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let (text, raw_text) = if lmcorr {
            let corrected = state.correct(&text);
            (corrected, Some(text))
        } else {
            (text, None)
        };
        let correct_ms = t.elapsed().as_secs_f64() * 1000.0;

        let output_tokens = text.split_whitespace().count();
        let total_ms = request_start.elapsed().as_secs_f64() * 1000.0;
        let rtf = (total_ms / 1000.0) / audio_dur_s;

        log::info!(
            "{} 200 {:.0}ms [audio={:.2}s preprocess={:.0}ms queue={:.0}ms \
             encoder={:.0}ms decode={:.0}ms correct={:.0}ms batch={} tokens={} rtf={:.4}x]",
            peer,
            total_ms,
            audio_dur_s,
            preprocess_ms,
            enc.queue_ms,
            enc.encoder_ms,
            decode_ms,
            correct_ms,
            enc.batch_size,
            output_tokens,
            rtf,
        );

        TranscriptionResponse {
            text,
            raw_text,
            usage: Usage {
                usage_type: "tokens".into(),
                input_tokens: n_frames,
                input_token_details: InputTokenDetails {
                    text_tokens: 0,
                    audio_tokens: n_frames,
                },
                output_tokens,
                total_tokens: n_frames + output_tokens,
            },
        }
    })
    .await
    .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(result))
}

// ---------------------------------------------------------------------------
// Streaming SSE handler
// ---------------------------------------------------------------------------

async fn handle_streaming(
    state: web::Data<AppState>,
    enc: EncoderResult,
    audio_dur_s: f64,
    preprocess_ms: f64,
    peer: String,
    request_start: Instant,
    lmcorr: bool,
) -> actix_web::Result<HttpResponse> {
    log::info!(
        "{} SSE started [audio={:.2}s preprocess={:.0}ms queue={:.0}ms \
         encoder={:.0}ms batch={}]",
        peer,
        audio_dur_s,
        preprocess_ms,
        enc.queue_ms,
        enc.encoder_ms,
        enc.batch_size,
    );

    let (tx, rx) = tokio::sync::mpsc::channel::<String>(256);

    let peer2 = peer.clone();
    tokio::task::spawn_blocking(move || {
        let decode_start = Instant::now();
        let mut token_count: usize = 0;
        let full_text = state.cpu_decoder.decode_streaming(
            &enc.enc_proj,
            enc.seq_len,
            &state.tokenizer,
            |_token_id, piece| {
                token_count += 1;
                let event = TranscriptTextDelta {
                    event_type: "transcript.text.delta".into(),
                    delta: piece.to_string(),
                };
                let line = format!("data: {}\n\n", serde_json::to_string(&event).unwrap());
                tx.blocking_send(line).ok();
            },
        );

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = request_start.elapsed().as_secs_f64() * 1000.0;
        let rtf = (total_ms / 1000.0) / audio_dur_s;

        log::info!(
            "{} SSE done {:.0}ms [decode={:.0}ms tokens={} rtf={:.4}x]",
            peer2,
            total_ms,
            decode_ms,
            token_count,
            rtf,
        );

        // deltas stream raw for latency; the final text carries the correction
        let (text, raw_text) = if lmcorr {
            let corrected = state.correct(&full_text);
            (corrected, Some(full_text))
        } else {
            (full_text, None)
        };
        let done = TranscriptTextDone {
            event_type: "transcript.text.done".into(),
            text,
            raw_text,
        };
        let line = format!("data: {}\n\n", serde_json::to_string(&done).unwrap());
        tx.blocking_send(line).ok();
    });

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<_, actix_web::Error>(web::Bytes::from(event)), rx))
    });

    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .append_header(("Cache-Control", "no-cache"))
        .streaming(stream))
}

// ---------------------------------------------------------------------------
// GET /v1/audio/transcriptions/ws — streaming WebSocket transcription
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WsQuery {
    /// Input sample rate for the PCM16 stream. Defaults to 16000 Hz.
    sample_rate: Option<u32>,
    /// Model name; a `-lmcorr` suffix enables brand correction on finals.
    model: Option<String>,
}

struct WsTranscribe {
    text: String,
    token_confidence: f32,
    speech_prob: f32,
    peak_speech_prob: f32,
    samples: usize,
    preprocess_ms: f64,
    queue_ms: f64,
    encoder_ms: f64,
    decode_ms: f64,
    batch_size: usize,
}

/// Run encoder + RNNT decode on a single contiguous audio buffer.
async fn transcribe_buffer(
    state: &web::Data<AppState>,
    samples: Vec<f32>,
) -> Result<WsTranscribe, String> {
    let num_samples = samples.len();
    if num_samples < audio::SAMPLE_RATE / 25 {
        // < 40 ms — not enough for a single mel frame; treat as silence.
        return Ok(WsTranscribe {
            text: String::new(),
            token_confidence: 0.0,
            speech_prob: 0.0,
            peak_speech_prob: 0.0,
            samples: num_samples,
            preprocess_ms: 0.0,
            queue_ms: 0.0,
            encoder_ms: 0.0,
            decode_ms: 0.0,
            batch_size: 0,
        });
    }

    let state_clone = state.clone();

    let t_pre = Instant::now();
    let (mel, n_mels, n_frames) = web::block(move || {
        let mel = audio::extract_mel_spectrogram(&samples);
        let (n_mels, n_frames) = audio::mel_spectrogram_shape(samples.len());
        (mel, n_mels, n_frames)
    })
    .await
    .map_err(|e| format!("preprocess error: {e}"))?;
    let preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    if n_frames == 0 {
        return Ok(WsTranscribe {
            text: String::new(),
            token_confidence: 0.0,
            speech_prob: 0.0,
            peak_speech_prob: 0.0,
            samples: num_samples,
            preprocess_ms,
            queue_ms: 0.0,
            encoder_ms: 0.0,
            decode_ms: 0.0,
            batch_size: 0,
        });
    }

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    state_clone
        .encoder_tx
        .send(EncoderRequest {
            mel,
            n_mels,
            n_frames,
            submitted_at: Instant::now(),
            reply: reply_tx,
        })
        .await
        .map_err(|_| "batch engine unavailable".to_string())?;

    let enc = reply_rx
        .await
        .map_err(|_| "batch engine error".to_string())?;

    let state_clone = state_clone.clone();
    let t_dec = Instant::now();
    let (text, token_conf, speech_prob, peak_speech_prob) = web::block(move || {
        state_clone.cpu_decoder.decode_with_probability(
            &enc.enc_proj,
            enc.seq_len,
            &state_clone.tokenizer,
        )
    })
    .await
    .map_err(|e| format!("decode error: {e}"))?;
    let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

    Ok(WsTranscribe {
        text,
        token_confidence: token_conf,
        speech_prob,
        peak_speech_prob,
        samples: num_samples,
        preprocess_ms,
        queue_ms: enc.queue_ms,
        encoder_ms: enc.encoder_ms,
        decode_ms,
        batch_size: enc.batch_size,
    })
}

#[utoipa::path(
    get,
    path = "/v1/audio/transcriptions/ws",
    params(
        ("sample_rate" = Option<u32>, Query, description = "Input PCM16 sample rate (Hz). Default 16000."),
    ),
    responses(
        (status = 101, description = "WebSocket upgrade. See README for the JSON wire protocol."),
    ),
    tag = "Audio"
)]
async fn transcribe_ws(
    state: web::Data<AppState>,
    req: HttpRequest,
    body: web::Payload,
    query: web::Query<WsQuery>,
) -> actix_web::Result<HttpResponse> {
    let peer = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "-".into());

    let sample_rate = query.sample_rate.unwrap_or(audio::SAMPLE_RATE as u32);
    if sample_rate != audio::SAMPLE_RATE as u32 {
        return Ok(HttpResponse::BadRequest().json(WsErrorResponse {
            event_type: "error".into(),
            error: format!(
                "unsupported sample_rate {sample_rate}: only {} Hz is supported",
                audio::SAMPLE_RATE
            ),
        }));
    }

    let lmcorr = AppState::wants_lmcorr(query.model.as_deref());
    if lmcorr && state.corrector.is_none() {
        return Ok(HttpResponse::BadRequest().json(WsErrorResponse {
            event_type: "error".into(),
            error: "model '-lmcorr' requested but no corrector model is loaded \
                    (set CORRECTOR_DIR to a valid model directory)"
                .into(),
        }));
    }

    let (response, mut session, mut msg_stream) = actix_ws::handle(&req, body)?;
    log::info!(
        "{} WS /v1/audio/transcriptions/ws connected lmcorr={}",
        peer,
        lmcorr
    );

    actix_web::rt::spawn(async move {
        let mut buffer: Vec<f32> = Vec::with_capacity(audio::SAMPLE_RATE * 5);
        let b64 = base64::engine::general_purpose::STANDARD;
        let peer_log = peer.clone();
        let mut chunk_idx: usize = 0;

        while let Some(Ok(msg)) = msg_stream.next().await {
            match msg {
                actix_ws::Message::Text(text) => {
                    let chunk: TranscriptionChunkRequest =
                        match serde_json::from_str(text.as_ref()) {
                            Ok(v) => v,
                            Err(e) => {
                                let _ = session
                                    .text(
                                        serde_json::to_string(&WsErrorResponse {
                                            event_type: "error".into(),
                                            error: format!("invalid JSON: {e}"),
                                        })
                                        .unwrap(),
                                    )
                                    .await;
                                continue;
                            }
                        };

                    let pcm = match b64.decode(chunk.audio.as_bytes()) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = session
                                .text(
                                    serde_json::to_string(&WsErrorResponse {
                                        event_type: "error".into(),
                                        error: format!("base64 decode error: {e}"),
                                    })
                                    .unwrap(),
                                )
                                .await;
                            continue;
                        }
                    };
                    buffer.extend(audio::pcm16_le_to_f32(&pcm));

                    let snapshot = buffer.clone();
                    let t_req = Instant::now();
                    let r = match transcribe_buffer(&state, snapshot).await {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = session
                                .text(
                                    serde_json::to_string(&WsErrorResponse {
                                        event_type: "error".into(),
                                        error: e,
                                    })
                                    .unwrap(),
                                )
                                .await;
                            continue;
                        }
                    };
                    let total_ms = t_req.elapsed().as_secs_f64() * 1000.0;
                    let audio_s = r.samples as f64 / audio::SAMPLE_RATE as f64;
                    let rtf = if audio_s > 0.0 { (total_ms / 1000.0) / audio_s } else { 0.0 };
                    log::info!(
                        "{} WS chunk={} {}ms [audio={:.2}s preprocess={:.0}ms \
                         queue={:.0}ms encoder={:.0}ms decode={:.0}ms batch={} \
                         speech_prob={:.3} token_conf={:.3} chars={} rtf={:.4}x] final={}",
                        peer,
                        chunk_idx,
                        total_ms as u64,
                        audio_s,
                        r.preprocess_ms,
                        r.queue_ms,
                        r.encoder_ms,
                        r.decode_ms,
                        r.batch_size,
                        r.speech_prob,
                        r.token_confidence,
                        r.text.chars().count(),
                        rtf,
                        chunk.r#final,
                    );

                    let event_type = if chunk.r#final { "final" } else { "delta" };
                    // correct finals only: deltas stay raw for latency
                    let (text, raw_text) = if lmcorr && chunk.r#final && !r.text.is_empty() {
                        let state = state.clone();
                        let raw = r.text.clone();
                        let corrected =
                            web::block(move || state.correct(&raw)).await.unwrap_or_else(
                                |e| {
                                    log::warn!("correction task failed: {e}");
                                    r.text.clone()
                                },
                            );
                        (corrected, Some(r.text))
                    } else {
                        (r.text, None)
                    };
                    let resp = TranscriptionChunkResponse {
                        event_type: event_type.into(),
                        text,
                        raw_text,
                        token_confidence: r.token_confidence,
                        speech_prob: r.speech_prob,
                        peak_speech_prob: r.peak_speech_prob,
                        samples: r.samples,
                    };
                    if session.text(serde_json::to_string(&resp).unwrap()).await.is_err() {
                        break;
                    }
                    if chunk.r#final {
                        buffer.clear();
                    }
                    chunk_idx += 1;
                }
                actix_ws::Message::Binary(bin) => {
                    buffer.extend(audio::pcm16_le_to_f32(&bin));
                    let snapshot = buffer.clone();
                    let t_req = Instant::now();
                    let r = match transcribe_buffer(&state, snapshot).await {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = session
                                .text(
                                    serde_json::to_string(&WsErrorResponse {
                                        event_type: "error".into(),
                                        error: e,
                                    })
                                    .unwrap(),
                                )
                                .await;
                            continue;
                        }
                    };
                    let total_ms = t_req.elapsed().as_secs_f64() * 1000.0;
                    let audio_s = r.samples as f64 / audio::SAMPLE_RATE as f64;
                    let rtf = if audio_s > 0.0 { (total_ms / 1000.0) / audio_s } else { 0.0 };
                    log::info!(
                        "{} WS chunk={} {}ms [audio={:.2}s preprocess={:.0}ms \
                         queue={:.0}ms encoder={:.0}ms decode={:.0}ms batch={} \
                         speech_prob={:.3} token_conf={:.3} chars={} rtf={:.4}x] binary",
                        peer,
                        chunk_idx,
                        total_ms as u64,
                        audio_s,
                        r.preprocess_ms,
                        r.queue_ms,
                        r.encoder_ms,
                        r.decode_ms,
                        r.batch_size,
                        r.speech_prob,
                        r.token_confidence,
                        r.text.chars().count(),
                        rtf,
                    );
                    let resp = TranscriptionChunkResponse {
                        event_type: "delta".into(),
                        text: r.text,
                        raw_text: None,
                        token_confidence: r.token_confidence,
                        speech_prob: r.speech_prob,
                        peak_speech_prob: r.peak_speech_prob,
                        samples: r.samples,
                    };
                    if session.text(serde_json::to_string(&resp).unwrap()).await.is_err() {
                        break;
                    }
                    chunk_idx += 1;
                }
                actix_ws::Message::Ping(p) => {
                    let _ = session.pong(&p).await;
                }
                actix_ws::Message::Close(reason) => {
                    let _ = session.close(reason).await;
                    break;
                }
                _ => {}
            }
        }
        log::info!("{} WS closed", peer_log);
    });

    Ok(response)
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

#[utoipa::path(
    get,
    path = "/health",
    responses((status = 200, description = "Service is healthy", body = HealthResponse)),
    tag = "System"
)]
async fn health() -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".into(),
    })
}

// ---------------------------------------------------------------------------
// Warmup — run a dummy forward pass to JIT-compile CUDA kernels
// ---------------------------------------------------------------------------

fn warmup(model: &model::GigaAMASR<Backend>, device: &LibTorchDevice) {
    let t = Instant::now();
    let dummy_mel: Tensor<Backend, 3> = Tensor::zeros([1, model::FEAT_IN, 320], device);
    let dummy_len: Tensor<Backend, 1> = Tensor::from_data(TensorData::from([20.0f32]), device);
    let _ = model.encoder.forward(dummy_mel, dummy_len);
    log::info!(
        "Warmup done in {:.0}ms",
        t.elapsed().as_secs_f64() * 1000.0
    );
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let config = AppConfig::init_from_env().expect("Failed to load config from environment");
    log::info!("Config: {:?}", config);

    let device = LibTorchDevice::Cuda(0);

    log::info!("Loading model from {}/ …", config.weights_dir);
    let model = model::GigaAMASR::<Backend>::load(&config.weights_dir, &device);

    let tokenizer_path = format!("{}/v3_e2e_rnnt_tokenizer.model", config.weights_dir);
    let tokenizer = decoding::Tokenizer::load(&tokenizer_path);
    let cpu_decoder = decoding::CpuRnntDecoder::from_model(&model.head, tokenizer.blank_id());

    let corrector = if std::path::Path::new(&config.corrector_dir).is_dir() {
        match corrector::Corrector::load(&config.corrector_dir) {
            Ok(c) => {
                log::info!(
                    "Brand corrector loaded from {}/ ({} catalog brands) — \
                     request a model name ending in '-lmcorr' to enable it",
                    config.corrector_dir,
                    c.catalog_len(),
                );
                Some(std::sync::Mutex::new(c))
            }
            Err(e) => {
                log::error!(
                    "Brand corrector present at {}/ but failed to load: {e}",
                    config.corrector_dir
                );
                None
            }
        }
    } else {
        log::info!(
            "No corrector model at {}/ — '-lmcorr' requests will be rejected",
            config.corrector_dir
        );
        None
    };

    log::info!("Warming up …");
    warmup(&model, &device);

    let max_batch = config.batch_size;
    let batch_timeout = Duration::from_millis(config.batch_timeout_ms);
    let max_vram_bytes = config.max_batch_vram_mb * 1024 * 1024;
    let (encoder_tx, encoder_rx) = tokio::sync::mpsc::channel(256);

    std::thread::spawn(move || {
        batch_engine(encoder_rx, model, device, max_batch, batch_timeout, max_vram_bytes);
    });

    let state = web::Data::new(AppState {
        encoder_tx,
        tokenizer,
        cpu_decoder,
        corrector,
    });

    let bind = (config.host.clone(), config.port);
    log::info!("Listening on http://{}:{}", bind.0, bind.1);
    log::info!("API docs at http://{}:{}/docs/", bind.0, bind.1);

    #[derive(OpenApi)]
    #[openapi(
        paths(transcribe, transcribe_ws, health),
        components(schemas(
            TranscribeRequest,
            TranscriptionResponse,
            Usage,
            InputTokenDetails,
            TranscriptTextDelta,
            TranscriptTextDone,
            TranscriptionChunkRequest,
            TranscriptionChunkResponse,
            WsErrorResponse,
            HealthResponse,
        ))
    )]
    struct ApiDoc;

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/v1/audio/transcriptions", web::post().to(transcribe))
            .route("/v1/audio/transcriptions/ws", web::get().to(transcribe_ws))
            .route("/health", web::get().to(health))
            .service(
                SwaggerUi::new("/docs/{_:.*}")
                    .url("/api-docs/openapi.json", ApiDoc::openapi()),
            )
    })
    .bind(bind)?
    .run()
    .await
}
