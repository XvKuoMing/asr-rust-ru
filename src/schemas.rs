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

    /// Model identifier — accepted but ignored (only GigaAM available).
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
    /// Mean `1 − P(blank)` over every encoder frame in this chunk. Use this
    /// as a VAD signal — close to 0 on silence, close to 1 on speech.
    pub speech_prob: f32,
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
