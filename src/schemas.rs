use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// ---------------------------------------------------------------------------
// POST /v1/audio/transcriptions — request (multipart, doc-only schema)
// ---------------------------------------------------------------------------

/// Multipart form fields for audio transcription.
#[derive(ToSchema)]
pub struct TranscribeRequest {
    /// Audio file (any format ffmpeg supports: wav, mp3, ogg, flac, …)
    #[schema(format = Binary)]
    pub file: String,

    /// Model identifier. A name ending in `-lmcorr` (e.g. `gigaam-lmcorr`)
    /// enables the embedded brand-correction LM on the transcript; any other
    /// (or absent) name transcribes without correction.
    #[schema(default = "gigaam")]
    pub model: Option<String>,

    /// Set `true` to receive Server-Sent Events instead of a single JSON blob.
    #[schema(default = json!(false))]
    pub stream: Option<bool>,
}

// ---------------------------------------------------------------------------
// Non-streaming JSON response
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionResponse {
    pub text: String,
    /// Uncorrected ASR output; present only when brand correction ran
    /// (`model` ended in `-lmcorr`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    #[serde(rename = "type")]
    pub usage_type: String,
    pub input_tokens: usize,
    pub input_token_details: InputTokenDetails,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct InputTokenDetails {
    pub text_tokens: usize,
    pub audio_tokens: usize,
}

// ---------------------------------------------------------------------------
// Streaming SSE event payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptTextDelta {
    #[serde(rename = "type")]
    pub event_type: String,
    pub delta: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptTextDone {
    #[serde(rename = "type")]
    pub event_type: String,
    pub text: String,
    /// Uncorrected ASR output; present only when brand correction ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
}

// ---------------------------------------------------------------------------
// WebSocket transcription protocol — /v1/audio/transcriptions/ws
// ---------------------------------------------------------------------------

/// Client → Server: a chunk of PCM16 audio encoded as base64.
///
/// The audio must be little-endian signed 16-bit PCM, mono, at the sample rate
/// declared on the URL query string (default 16000 Hz). Chunks are accumulated
/// server-side; each chunk triggers an incremental transcription pass and a
/// `TranscriptionChunkResponse` reply.
///
/// A frame with `final = true` flushes the current segment and resets the
/// internal audio buffer — useful when the client's VAD decides the speaker
/// has finished a turn.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TranscriptionChunkRequest {
    /// Base64-encoded PCM16-LE audio.
    pub audio: String,
    /// Set `true` to flush the segment and reset the server buffer afterwards.
    #[serde(default)]
    pub r#final: bool,
}

/// Server → Client: incremental transcription for one chunk.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TranscriptionChunkResponse {
    /// Discriminator: `"delta"` for partial, `"final"` for end-of-segment.
    #[serde(rename = "type")]
    pub event_type: String,
    /// Text decoded from the audio buffer up to this chunk.
    pub text: String,
    /// Average softmax probability of the emitted (non-blank) tokens. 0.0 if
    /// nothing was emitted (likely silence).
    pub token_confidence: f32,
    /// Mean `1 − P(blank)` over every encoder frame in this chunk — a VAD
    /// signal (≈0 on silence). NOTE: being a mean it under-reports short
    /// speech; a lone spoken number diluted across silent frames scores low.
    pub speech_prob: f32,
    /// Max per-frame `1 − P(blank)` in this chunk. Stays high for brief
    /// speech the mean washes out — prefer this for noise filtering.
    pub peak_speech_prob: f32,
    /// Uncorrected ASR output; present only on `final` chunks when brand
    /// correction ran (`model=...-lmcorr` on the URL query).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
    /// Length (in 16 kHz samples) of the audio buffer that produced `text`.
    pub samples: usize,
}

/// Server → Client: protocol-level error.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct WsErrorResponse {
    #[serde(rename = "type")]
    pub event_type: String,
    pub error: String,
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
}
