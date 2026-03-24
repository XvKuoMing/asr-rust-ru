use serde::Serialize;
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
// Health
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
}
