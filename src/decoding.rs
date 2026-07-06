use burn::tensor::{backend::Backend, Tensor};

use crate::model::decoder::RNNTHead;

const MAX_SYMBOLS_PER_STEP: usize = 10;

// ---------------------------------------------------------------------------
// SentencePiece tokenizer (vocab parsed once at load time)
// ---------------------------------------------------------------------------

pub struct Tokenizer {
    pieces: Vec<String>,
}

impl Tokenizer {
    pub fn load(model_path: &str) -> Self {
        let data = std::fs::read(model_path).expect("Failed to read tokenizer model");
        let pieces = parse_spm_vocab(&data);
        Self { pieces }
    }

    pub fn blank_id(&self) -> usize {
        self.pieces.len()
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            result.push_str(&self.decode_token(id, result.is_empty()));
        }
        result
    }

    /// Decode a single token ID to its text piece.
    /// Handles the SentencePiece `▁` word-boundary marker.
    pub fn decode_token(&self, token_id: usize, is_first: bool) -> String {
        if token_id >= self.pieces.len() {
            return String::new();
        }
        let piece = &self.pieces[token_id];
        if let Some(rest) = piece.strip_prefix('\u{2581}') {
            if is_first {
                rest.to_string()
            } else {
                format!(" {rest}")
            }
        } else {
            piece.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-side RNNT decoder — all weights held as flat f32 arrays
// ---------------------------------------------------------------------------

/// Short-audio / anti-deletion knobs for greedy RNNT decoding.
///
/// RNNT greedy decode collapses a whole utterance to the empty string when the
/// blank class wins on every encoder frame — the failure mode behind "an
/// isolated number in a short 8 kHz clip is ignored". These knobs bias the
/// blank logit down (especially on short clips) and, as a hard backstop, force
/// a re-decode that emits at least one token whenever the clip actually
/// contains speech. Genuine silence still decodes to the empty string.
#[derive(Clone, Copy, Debug)]
pub struct AntiDeletion {
    /// Subtracted from the blank logit at every decode step, all utterances.
    pub blank_penalty: f32,
    /// Encoder-frame count at/below which a clip is treated as "short".
    /// GigaAM's encoder emits ~25 frames/s, so 40 ≈ 1.6 s.
    pub short_frames: usize,
    /// Extra blank penalty added on top of `blank_penalty` for short clips.
    pub short_extra_penalty: f32,
    /// Force at least one emitted token when the clip contains speech.
    pub guarantee_nonempty: bool,
    /// VAD gate for the guarantee: only force emission if the peak per-frame
    /// non-blank probability `max_t (1 - softmax(logits)[blank])` reaches this.
    /// Pure silence sits near 0; clearly-spoken short clips are well above it.
    /// It is a precision/recall knob: lower = recover more dropped numbers but
    /// risk emitting on ambiguous background noise; higher = safer on noise but
    /// may leave the most degraded numbers empty. Tune with eval_short_audio.py.
    pub speech_prob_thresh: f32,
    /// Hard cap on the escalated blank penalty used by the guarantee.
    pub max_penalty: f32,
}

impl Default for AntiDeletion {
    fn default() -> Self {
        Self {
            blank_penalty: 0.0,
            short_frames: 40,
            short_extra_penalty: 2.5,
            guarantee_nonempty: true,
            speech_prob_thresh: 0.2,
            max_penalty: 15.0,
        }
    }
}

pub struct CpuRnntDecoder {
    embed: Vec<f32>,
    lstm_ih_w: Vec<f32>,
    lstm_ih_b: Vec<f32>,
    lstm_hh_w: Vec<f32>,
    lstm_hh_b: Vec<f32>,
    pred_w: Vec<f32>,
    pred_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    pred_hidden: usize,
    joint_hidden: usize,
    num_classes: usize,
    blank_id: usize,
    anti_del: AntiDeletion,
}

impl CpuRnntDecoder {
    /// Extract weights from the Burn GPU model once at startup.
    pub fn from_model<B: Backend>(head: &RNNTHead<B>, blank_id: usize) -> Self {
        let pred_hidden = head.decoder.lstm.hidden_size;

        let embed = head.decoder.embed.weight.val().into_data().to_vec::<f32>().unwrap();

        let lstm_ih_w = head.decoder.lstm.weight_ih.weight.val().into_data().to_vec::<f32>().unwrap();
        let lstm_ih_b = head.decoder.lstm.weight_ih.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap();
        let lstm_hh_w = head.decoder.lstm.weight_hh.weight.val().into_data().to_vec::<f32>().unwrap();
        let lstm_hh_b = head.decoder.lstm.weight_hh.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap();

        let pred_w = head.joint.pred.weight.val().into_data().to_vec::<f32>().unwrap();
        let pred_b = head.joint.pred.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap();
        let out_w  = head.joint.output.weight.val().into_data().to_vec::<f32>().unwrap();
        let out_b  = head.joint.output.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap();

        let joint_hidden = pred_b.len();
        let num_classes = out_b.len();

        Self {
            embed, lstm_ih_w, lstm_ih_b, lstm_hh_w, lstm_hh_b,
            pred_w, pred_b, out_w, out_b,
            pred_hidden, joint_hidden, num_classes, blank_id,
            anti_del: AntiDeletion::default(),
        }
    }

    /// Override the anti-deletion / short-audio decoding knobs.
    pub fn set_anti_deletion(&mut self, cfg: AntiDeletion) {
        self.anti_del = cfg;
    }

    /// Current anti-deletion configuration.
    pub fn anti_deletion(&self) -> AntiDeletion {
        self.anti_del
    }

    /// Run RNNT greedy decoding entirely on CPU.
    /// `enc_proj` is the pre-computed `joint.enc(encoded)` as flat f32,
    ///  row-major [seq_len, joint_hidden].
    pub fn decode(&self, enc_proj: &[f32], seq_len: usize, tokenizer: &Tokenizer) -> String {
        self.decode_inner(enc_proj, seq_len, tokenizer, |_, _| {})
    }

    /// Run decoding and additionally compute speech-probability metrics.
    ///
    /// Returns `(text, token_confidence, speech_prob, peak_speech_prob)`:
    /// - `token_confidence`: mean softmax probability of every emitted non-blank
    ///   token (0.0 when no tokens were emitted).
    /// - `speech_prob`: mean `1 - P(blank)` across every encoder frame in the
    ///   chunk — a VAD signal (close to 0 on silence, close to 1 on speech).
    ///   Being a MEAN it under-reports short speech: one spoken number diluted
    ///   across silent frames scores low.
    /// - `peak_speech_prob`: max per-frame `1 - P(blank)` — stays high for a
    ///   brief word the mean washes out; the better gate for noise filtering.
    pub fn decode_with_probability(
        &self,
        enc_proj: &[f32],
        seq_len: usize,
        tokenizer: &Tokenizer,
    ) -> (String, f32, f32, f32) {
        let jh = self.joint_hidden;
        let ph = self.pred_hidden;

        let mut hyp: Vec<usize> = Vec::new();
        let mut h = vec![0.0f32; ph];
        let mut c = vec![0.0f32; ph];
        let mut last_label: Option<usize> = None;

        let mut emb = vec![0.0f32; ph];
        let mut gates = vec![0.0f32; 4 * ph];
        let mut h_new = vec![0.0f32; ph];
        let mut c_new = vec![0.0f32; ph];
        let mut pred_proj = vec![0.0f32; jh];
        let mut combined = vec![0.0f32; jh];
        let mut logits = vec![0.0f32; self.num_classes];
        let mut probs = vec![0.0f32; self.num_classes];

        let mut token_prob_sum: f32 = 0.0;
        let mut token_count: usize = 0;
        let mut nonblank_mass_sum: f32 = 0.0;
        let mut frame_count: usize = 0;

        let ad = self.anti_del; // Copy
        let base_penalty = if seq_len <= ad.short_frames {
            ad.blank_penalty + ad.short_extra_penalty
        } else {
            ad.blank_penalty
        };
        let mut best_margin = f32::NEG_INFINITY;
        let mut peak_prob = 0.0f32;

        for t in 0..seq_len {
            let f_proj = &enc_proj[t * jh..(t + 1) * jh];
            let mut not_blank = true;
            let mut symbols = 0usize;
            let mut blank_recorded_this_frame = false;

            while not_blank && symbols < MAX_SYMBOLS_PER_STEP {
                match last_label {
                    Some(id) => emb.copy_from_slice(&self.embed[id * ph..(id + 1) * ph]),
                    None => emb.iter_mut().for_each(|v| *v = 0.0),
                }

                self.lstm_cell(&emb, &h, &c, &mut gates, &mut h_new, &mut c_new);
                matvec_row(&self.pred_w, &self.pred_b, &h_new, &mut pred_proj, ph, jh);

                for i in 0..jh {
                    combined[i] = (f_proj[i] + pred_proj[i]).max(0.0);
                }

                matvec_row(&self.out_w, &self.out_b, &combined, &mut logits, jh, self.num_classes);
                softmax(&logits, &mut probs);
                // anti-deletion: metrics (below) stay on RAW probs; bias only
                // the argmax decision away from blank.
                let margin = max_excluding(&logits, self.blank_id) - logits[self.blank_id];
                if margin > best_margin {
                    best_margin = margin;
                }
                let nb_prob = 1.0 - probs[self.blank_id];
                if nb_prob > peak_prob {
                    peak_prob = nb_prob;
                }
                logits[self.blank_id] -= base_penalty;
                let k = argmax(&logits);

                if !blank_recorded_this_frame {
                    nonblank_mass_sum += 1.0 - probs[self.blank_id];
                    frame_count += 1;
                    blank_recorded_this_frame = true;
                }

                if k == self.blank_id {
                    not_blank = false;
                } else {
                    h.copy_from_slice(&h_new);
                    c.copy_from_slice(&c_new);
                    hyp.push(k);
                    last_label = Some(k);
                    token_prob_sum += probs[k];
                    token_count += 1;
                    symbols += 1;
                }
            }
        }

        let text = tokenizer.decode(&hyp);
        let token_confidence = if token_count > 0 {
            token_prob_sum / token_count as f32
        } else {
            0.0
        };
        let speech_prob = if frame_count > 0 {
            nonblank_mass_sum / frame_count as f32
        } else {
            0.0
        };

        // Hard non-empty guarantee (short-audio number drops): blank swept the
        // whole clip, but the peak non-blank probability reached
        // `speech_prob_thresh` ⇒ real (quiet/short) speech ⇒ re-decode at a
        // penalty large enough to flip the best frame. Forced tokens report
        // confidence 0.0. (speech_prob, the per-frame mean, is left as measured.)
        if ad.guarantee_nonempty && hyp.is_empty() && peak_prob >= ad.speech_prob_thresh {
            let pen = ((-best_margin) + 0.5).max(base_penalty).min(ad.max_penalty);
            let (hyp2, _, _) = self.greedy_pass(enc_proj, seq_len, tokenizer, pen, &mut |_, _| {});
            return (tokenizer.decode(&hyp2), 0.0, speech_prob, peak_prob);
        }

        (text, token_confidence, speech_prob, peak_prob)
    }

    /// Like [`decode`], but calls `on_token(token_id, text_piece)` for each
    /// emitted token — suitable for SSE streaming.
    pub fn decode_streaming<F>(
        &self,
        enc_proj: &[f32],
        seq_len: usize,
        tokenizer: &Tokenizer,
        on_token: F,
    ) -> String
    where
        F: FnMut(usize, &str),
    {
        self.decode_inner(enc_proj, seq_len, tokenizer, on_token)
    }

    fn decode_inner<F>(
        &self,
        enc_proj: &[f32],
        seq_len: usize,
        tokenizer: &Tokenizer,
        mut on_token: F,
    ) -> String
    where
        F: FnMut(usize, &str),
    {
        let ad = self.anti_del; // Copy
        // Length-adaptive blank penalty: short clips (isolated numbers etc.)
        // get an extra nudge away from all-blank / empty output.
        let base = if seq_len <= ad.short_frames {
            ad.blank_penalty + ad.short_extra_penalty
        } else {
            ad.blank_penalty
        };

        let (hyp, best_margin, peak_prob) =
            self.greedy_pass(enc_proj, seq_len, tokenizer, base, &mut on_token);

        // Hard non-empty guarantee: the greedy pass emitted nothing, but the
        // peak per-frame non-blank probability reached `speech_prob_thresh` —
        // real (quiet/short) speech, not silence. Re-decode at a blank penalty
        // large enough to flip the best frame so an isolated number/word is
        // never dropped. Genuine silence has near-zero non-blank probability, so
        // it fails the gate and correctly stays empty. Because the first pass
        // emitted nothing, no tokens were streamed, so replaying through
        // `on_token` cannot double-emit.
        if ad.guarantee_nonempty && hyp.is_empty() && peak_prob >= ad.speech_prob_thresh {
            let pen = ((-best_margin) + 0.5).max(base).min(ad.max_penalty);
            let (hyp2, _, _) =
                self.greedy_pass(enc_proj, seq_len, tokenizer, pen, &mut on_token);
            return tokenizer.decode(&hyp2);
        }

        tokenizer.decode(&hyp)
    }

    /// One greedy RNNT pass at a fixed `blank_penalty` (subtracted from the
    /// blank logit before every `argmax`). Streams emitted tokens through
    /// `on_token`. Returns `(emitted_ids, best_nonblank_margin, peak_nonblank_prob)`:
    ///   * `best_nonblank_margin` = max over steps of
    ///     `(max_nonblank_logit - blank_logit)` on the RAW logits — sizes the
    ///     penalty needed to flip the best frame.
    ///   * `peak_nonblank_prob` = max over steps of `(1 - softmax(logits)[blank])`
    ///     on the RAW logits — the VAD signal that gates the non-empty guarantee.
    fn greedy_pass<F>(
        &self,
        enc_proj: &[f32],
        seq_len: usize,
        tokenizer: &Tokenizer,
        blank_penalty: f32,
        on_token: &mut F,
    ) -> (Vec<usize>, f32, f32)
    where
        F: FnMut(usize, &str),
    {
        let jh = self.joint_hidden;
        let ph = self.pred_hidden;

        let mut hyp: Vec<usize> = Vec::new();
        let mut h = vec![0.0f32; ph];
        let mut c = vec![0.0f32; ph];
        let mut last_label: Option<usize> = None;

        let mut emb = vec![0.0f32; ph];
        let mut gates = vec![0.0f32; 4 * ph];
        let mut h_new = vec![0.0f32; ph];
        let mut c_new = vec![0.0f32; ph];
        let mut pred_proj = vec![0.0f32; jh];
        let mut combined = vec![0.0f32; jh];
        let mut logits = vec![0.0f32; self.num_classes];

        let mut best_margin = f32::NEG_INFINITY;
        let mut peak_prob = 0.0f32;

        for t in 0..seq_len {
            let f_proj = &enc_proj[t * jh..(t + 1) * jh];
            let mut not_blank = true;
            let mut symbols = 0usize;

            while not_blank && symbols < MAX_SYMBOLS_PER_STEP {
                match last_label {
                    Some(id) => emb.copy_from_slice(&self.embed[id * ph..(id + 1) * ph]),
                    None => emb.iter_mut().for_each(|v| *v = 0.0),
                }

                self.lstm_cell(&emb, &h, &c, &mut gates, &mut h_new, &mut c_new);
                matvec_row(&self.pred_w, &self.pred_b, &h_new, &mut pred_proj, ph, jh);

                for i in 0..jh {
                    combined[i] = (f_proj[i] + pred_proj[i]).max(0.0);
                }

                matvec_row(&self.out_w, &self.out_b, &combined, &mut logits, jh, self.num_classes);

                // anti-deletion: measure speechiness on RAW logits (both the
                // logit margin, for penalty sizing, and the softmax VAD prob,
                // for the guarantee gate), then bias the blank class down.
                let blank_logit = logits[self.blank_id];
                let margin = max_excluding(&logits, self.blank_id) - blank_logit;
                if margin > best_margin {
                    best_margin = margin;
                }
                let nb_prob = nonblank_prob(&logits, self.blank_id);
                if nb_prob > peak_prob {
                    peak_prob = nb_prob;
                }
                logits[self.blank_id] = blank_logit - blank_penalty;

                let k = argmax(&logits);

                if k == self.blank_id {
                    not_blank = false;
                } else {
                    h.copy_from_slice(&h_new);
                    c.copy_from_slice(&c_new);

                    let piece = tokenizer.decode_token(k, hyp.is_empty());
                    on_token(k, &piece);

                    hyp.push(k);
                    last_label = Some(k);
                    symbols += 1;
                }
            }
        }

        (hyp, best_margin, peak_prob)
    }

    /// Compute LSTM gates and produce h_new, c_new without mutating h/c.
    #[inline]
    fn lstm_cell(
        &self,
        input: &[f32],
        h: &[f32],
        c: &[f32],
        gates: &mut [f32],
        h_new: &mut [f32],
        c_new: &mut [f32],
    ) {
        let ph = self.pred_hidden;
        let gs = 4 * ph;

        // Fused bias init
        for j in 0..gs {
            gates[j] = self.lstm_ih_b[j] + self.lstm_hh_b[j];
        }

        // Cache-friendly: iterate over input rows, accumulate into gates
        for i in 0..ph {
            let xi = input[i];
            let hi = h[i];
            let ih_row = &self.lstm_ih_w[i * gs..(i + 1) * gs];
            let hh_row = &self.lstm_hh_w[i * gs..(i + 1) * gs];
            for j in 0..gs {
                gates[j] += xi * ih_row[j] + hi * hh_row[j];
            }
        }

        for i in 0..ph {
            let ig = sigmoid(gates[i]);
            let fg = sigmoid(gates[ph + i]);
            let cg = gates[2 * ph + i].tanh();
            let og = sigmoid(gates[3 * ph + i]);
            c_new[i] = fg * c[i] + ig * cg;
            h_new[i] = og * c_new[i].tanh();
        }
    }
}

/// Pre-compute encoder projection on GPU, return as CPU flat Vec (single-sequence).
#[allow(dead_code)]
pub fn precompute_enc_proj<B: Backend>(
    head: &RNNTHead<B>,
    encoded: &Tensor<B, 3>,
    encoded_len: &Tensor<B, 1>,
) -> (Vec<f32>, usize) {
    // encoded: [1, d_model, T] -> [1, T, d_model]
    let encoded_bt = encoded.clone().swap_dims(1, 2);
    let enc_proj = head.joint.enc.forward(encoded_bt); // [1, T, joint_hidden]
    let seq_len = encoded_len.clone().into_data().to_vec::<f32>().unwrap()[0] as usize;
    let data = enc_proj.into_data().to_vec::<f32>().unwrap();
    (data, seq_len)
}

/// Batched version: run encoder projection for all batch elements on GPU,
/// then split into per-sequence CPU data.
pub fn precompute_enc_proj_batch<B: Backend>(
    head: &RNNTHead<B>,
    encoded: &Tensor<B, 3>,
    encoded_len: &Tensor<B, 1>,
) -> Vec<(Vec<f32>, usize)> {
    let [batch_size, _d_model, max_t] = encoded.dims();
    let encoded_bt = encoded.clone().swap_dims(1, 2); // [B, T, d_model]
    let enc_proj = head.joint.enc.forward(encoded_bt); // [B, T, joint_hidden]
    let jh = enc_proj.dims()[2];

    let lens: Vec<f32> = encoded_len.clone().into_data().to_vec::<f32>().unwrap();
    let all_data: Vec<f32> = enc_proj.into_data().to_vec::<f32>().unwrap();

    (0..batch_size)
        .map(|b| {
            let start = b * max_t * jh;
            let end = start + max_t * jh;
            (all_data[start..end].to_vec(), lens[b] as usize)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tiny math helpers
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax — writes probabilities into `out`.
#[inline]
fn softmax(logits: &[f32], out: &mut [f32]) {
    let mut max_v = logits[0];
    for &v in &logits[1..] {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - max_v).exp();
        out[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv;
    }
}

/// Cache-friendly row-major matvec: out = bias + input @ W
/// W is [in_len, out_len] in row-major layout.
#[inline]
fn matvec_row(w: &[f32], bias: &[f32], input: &[f32], out: &mut [f32], in_len: usize, out_len: usize) {
    out[..out_len].copy_from_slice(&bias[..out_len]);
    for i in 0..in_len {
        let x = input[i];
        let row = &w[i * out_len..(i + 1) * out_len];
        for j in 0..out_len {
            out[j] += x * row[j];
        }
    }
}

/// Max value over `v` excluding index `skip` (used to measure how close the
/// best non-blank class came to blank).
#[inline]
fn max_excluding(v: &[f32], skip: usize) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for (i, &val) in v.iter().enumerate() {
        if i != skip && val > best {
            best = val;
        }
    }
    best
}

/// `1 - softmax(logits)[blank_id]`, numerically stable — the per-frame
/// probability that SOMETHING (a non-blank token) should be emitted here. Near
/// 0 on silence, high on speech; the VAD signal that gates the non-empty guard.
#[inline]
fn nonblank_prob(logits: &[f32], blank_id: usize) -> f32 {
    let mut mx = f32::NEG_INFINITY;
    for &v in logits {
        if v > mx {
            mx = v;
        }
    }
    let mut sum = 0.0f32;
    for &v in logits {
        sum += (v - mx).exp();
    }
    let p_blank = (logits[blank_id] - mx).exp() / sum;
    1.0 - p_blank
}

#[inline]
fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > best_v {
            best_v = val;
            best_i = i;
        }
    }
    best_i
}

// ---------------------------------------------------------------------------
// SentencePiece protobuf parser
// ---------------------------------------------------------------------------

fn parse_spm_vocab(data: &[u8]) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (field_number, wire_type, new_pos) = read_proto_tag(data, pos);
        pos = new_pos;
        if field_number == 1 && wire_type == 2 {
            let (msg_data, new_pos) = read_proto_bytes(data, pos);
            pos = new_pos;
            pieces.push(parse_sentence_piece(msg_data));
        } else {
            pos = skip_proto_field(data, pos, wire_type);
        }
    }
    pieces
}

fn parse_sentence_piece(data: &[u8]) -> String {
    let mut pos = 0;
    let mut piece = String::new();
    while pos < data.len() {
        let (field_number, wire_type, new_pos) = read_proto_tag(data, pos);
        pos = new_pos;
        if field_number == 1 && wire_type == 2 {
            let (bytes, new_pos) = read_proto_bytes(data, pos);
            pos = new_pos;
            piece = String::from_utf8_lossy(bytes).to_string();
        } else {
            pos = skip_proto_field(data, pos, wire_type);
        }
    }
    piece
}

fn read_proto_tag(data: &[u8], pos: usize) -> (u32, u32, usize) {
    let (val, new_pos) = read_varint(data, pos);
    ((val >> 3) as u32, (val & 0x7) as u32, new_pos)
}

fn read_varint(data: &[u8], mut pos: usize) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= data.len() { break; }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 { break; }
    }
    (result, pos)
}

fn read_proto_bytes(data: &[u8], pos: usize) -> (&[u8], usize) {
    let (len, new_pos) = read_varint(data, pos);
    let end = new_pos + len as usize;
    (&data[new_pos..end.min(data.len())], end)
}

fn skip_proto_field(data: &[u8], pos: usize, wire_type: u32) -> usize {
    match wire_type {
        0 => read_varint(data, pos).1,
        1 => pos + 8,
        2 => { let (len, p) = read_varint(data, pos); p + len as usize }
        5 => pos + 4,
        _ => data.len(),
    }
}
