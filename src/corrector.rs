//! Embedded brand-correction LM.
//!
//! A fine-tuned ruT5 (T5-base) runs in-process via candle and rewrites an ASR
//! hypothesis; a catalog acceptance filter then keeps ONLY the edits that turn
//! a phonetically-close span into a real catalog brand. Everything else —
//! deletions, insertions, free-form rewrites — is reverted to the raw
//! hypothesis, so correction can never lose content or clobber a genuine word.
//!
//! Activated per request by the OpenAI `model` field: a name ending in
//! `-lmcorr` enables correction; any other (or absent) name bypasses it.
//!
//! The model directory (`CORRECTOR_DIR`, default `corrector/`) holds the
//! standard HF export: `model.safetensors`, `config.json`, `tokenizer.json`,
//! plus `brands.txt` (one canonical brand phrase per line; `# comments` and
//! `canonical | alias` lines allowed — only the canonical column is used).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5;
use tokenizers::Tokenizer as HfTokenizer;

/// Task prefix the corrector was fine-tuned with. Must match training.
const PREFIX: &str = "исправь: ";
/// Generation cap (tokens). Transcript chunks are short sentences.
const MAX_NEW_TOKENS: usize = 192;
/// Max normalized edit distance between the raw span and a brand, relative to
/// the brand length, for an edit to count as "phonetically close".
const REL_DIST: f32 = 0.45;

pub struct Corrector {
    model: t5::T5ForConditionalGeneration,
    tokenizer: HfTokenizer,
    device: Device,
    eos_token_id: u32,
    decoder_start_token_id: u32,
    /// (normalized, canonical-cased) brand phrases, longest-normalized first.
    catalog: Vec<(String, String)>,
    /// Space-stripped catalog keys (plus Cyrillic transliterations of Latin
    /// brands), grouped by char length — the pre-filter screen.
    screen: Vec<Vec<String>>,
    /// Corpus-frequent words (optional `common_words.txt`). Garbled brands are
    /// non-words, so a span made of known real words is skipped as a candidate
    /// unless it is nearly an exact brand match.
    common: std::collections::HashSet<String>,
    /// Curated ambiguous words (optional `ambiguous_words.txt`): real words the
    /// ASR emits when a brand was spoken (аквариум ~ Аква Ареал). These ARE
    /// windowed — with wide context, so the LM can decide from the phrasing.
    ambiguous: std::collections::HashSet<String>,
}

impl Corrector {
    /// Load the corrector from `dir`. Returns Err with a reason if any file is
    /// missing or malformed — the caller decides whether that is fatal.
    pub fn load(dir: &str) -> Result<Self, String> {
        let cfg_raw = std::fs::read_to_string(format!("{dir}/config.json"))
            .map_err(|e| format!("read {dir}/config.json: {e}"))?;
        let mut config: t5::Config =
            serde_json::from_str(&cfg_raw).map_err(|e| format!("parse config.json: {e}"))?;
        config.use_cache = true; // kv-cache greedy decode

        // GPU when built with the `corrector-cuda` feature (falls back to CPU
        // if no device is present); plain CPU otherwise.
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        log::info!(
            "corrector device: {}",
            if device.is_cuda() { "cuda:0" } else { "cpu" }
        );
        // bf16 on GPU halves memory traffic per decode step (greedy argmax is
        // insensitive to the precision loss); f32 on CPU (no native bf16).
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[format!("{dir}/model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| format!("load model.safetensors: {e}"))?
        };
        // Some exports carry a separately-trained lm_head while the config
        // still claims tied embeddings (HF silently unties in that case; if we
        // honored the config the logits would come from the wrong matrix).
        if config.tie_word_embeddings && vb.contains_tensor("lm_head.weight") {
            log::info!("corrector: untying word embeddings (checkpoint has its own lm_head)");
            config.tie_word_embeddings = false;
        }
        let model = t5::T5ForConditionalGeneration::load(vb, &config)
            .map_err(|e| format!("build T5: {e}"))?;

        let tokenizer = HfTokenizer::from_file(format!("{dir}/tokenizer.json"))
            .map_err(|e| format!("load tokenizer.json: {e}"))?;

        let catalog = load_catalog(&format!("{dir}/brands.txt"))?;
        if catalog.is_empty() {
            return Err(format!("{dir}/brands.txt contains no usable brands"));
        }
        let screen = build_screen(&catalog);
        let common: std::collections::HashSet<String> =
            std::fs::read_to_string(format!("{dir}/common_words.txt"))
                .map(|d| {
                    d.lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty() && !l.starts_with('#'))
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
        let mut common = common;
        if !common.is_empty() {
            // catalog brand words are real words too — a span is a garble
            // candidate only if it contains at least one UNKNOWN word
            for (n, _) in &catalog {
                for w in n.split_whitespace() {
                    common.insert(w.to_string());
                }
            }
        }
        if common.is_empty() {
            log::warn!(
                "no {dir}/common_words.txt — the candidate screen will fire on \
                 ordinary words and correction windows will be larger/slower"
            );
        }
        // "word -> Brand" lines; only the word side matters for the screen
        // (the LM + catalog filter produce/validate the brand).
        let ambiguous: std::collections::HashSet<String> =
            std::fs::read_to_string(format!("{dir}/ambiguous_words.txt"))
                .map(|d| {
                    d.lines()
                        .filter_map(|l| l.trim().split(" -> ").next())
                        .filter(|w| !w.is_empty() && !w.starts_with('#'))
                        .map(normalize)
                        .collect()
                })
                .unwrap_or_default();
        if !ambiguous.is_empty() {
            log::info!("corrector: {} curated ambiguous words", ambiguous.len());
        }

        Ok(Self {
            eos_token_id: config.eos_token_id as u32,
            decoder_start_token_id: config
                .decoder_start_token_id
                .unwrap_or(config.pad_token_id) as u32,
            model,
            tokenizer,
            device,
            catalog,
            screen,
            common,
            ambiguous,
        })
    }

    /// Pre-filter: does `text` contain any span phonetically close to a
    /// catalog brand? Texts with nothing brand-like skip T5 generation
    /// entirely (micro- vs multi-hundred-ms latency).
    pub fn has_brand_candidate(&self, text: &str) -> bool {
        let words: Vec<String> = normalize(text)
            .split_whitespace()
            .map(|w| w.to_string())
            .collect();
        !self.candidate_spans(&words).is_empty()
    }

    /// Candidate spans `(start, end, ratio)` — word-index ranges (1–4 adjacent
    /// words, spaces stripped) within the acceptance filter's distance of a
    /// catalog brand, with the best relative edit distance found (`d / len`).
    /// The bound is exactly the acceptance bound, so no span the constraint
    /// could later accept is ever missed; `ratio` ranks how garble-like a span
    /// is (true garbles score low, borderline screen fires near `REL_DIST`).
    fn candidate_spans(&self, words: &[String]) -> Vec<(usize, usize, f32)> {
        let mut spans: Vec<(usize, usize, f32)> = Vec::new();
        for n in 1..=4usize {
            for (i, win) in words.windows(n).enumerate() {
                let span: String = win.concat();
                let sl = span.chars().count();
                if sl < 3 {
                    continue;
                }
                // screen keys are grouped by length; a key of length L is
                // reachable only if |L - sl| <= limit(L) — scan the buckets
                // that can possibly match.
                let mut best: Option<f32> = None;
                for (len, bucket) in self.screen.iter().enumerate() {
                    let limit = screen_limit(len);
                    if len == 0 || len.abs_diff(sl) > limit {
                        continue;
                    }
                    for key in bucket {
                        if levenshtein_within(&span, key, limit) {
                            let ratio = levenshtein(&span, key) as f32 / len as f32;
                            if best.map(|b| ratio < b).unwrap_or(true) {
                                best = Some(ratio);
                            }
                        }
                    }
                }
                if let Some(r) = best {
                    // ratio 0 = span already IS the brand: nothing to fix.
                    // A span made entirely of corpus-known real words is not a
                    // garble (garbles are non-words) unless it is nearly exact.
                    let has_ambiguous =
                        win.iter().any(|w| self.ambiguous.contains(w.as_str()));
                    let all_common = !self.common.is_empty()
                        && win.iter().all(|w| {
                            w.chars().count() < 3 || self.common.contains(w.as_str())
                        });
                    if r > 0.0 && (has_ambiguous || !all_common) {
                        spans.push((i, i + n, r));
                    }
                }
            }
        }
        spans
    }

    pub fn catalog_len(&self) -> usize {
        self.catalog.len()
    }

    /// Correct `text`: brand-candidate pre-filter, then WINDOWED LM correction.
    ///
    /// Instead of regenerating the whole sentence, only a ±`WINDOW_CONTEXT`-word
    /// window around each candidate span is passed through T5 — latency scales
    /// with the window (~10 words), not the sentence. Corrected windows are
    /// spliced back; the acceptance filter runs per window, so anything except
    /// a close-garble→catalog-brand edit is reverted exactly as before.
    /// Texts with nothing brand-like return immediately. On any generation
    /// error the raw text is returned unchanged.
    pub fn correct(&mut self, text: &str) -> String {
        const WINDOW_CONTEXT: usize = 1; // words of context on each side
        const MERGE_GAP: usize = 3; // merge windows closer than this
        const MAX_WINDOWS: usize = 2; // best-ranked windows processed per text

        if text.trim().is_empty() {
            return text.to_string();
        }
        let raw_words: Vec<&str> = text.split_whitespace().collect();
        let norm_words: Vec<String> = raw_words.iter().map(|w| normalize(w)).collect();
        let spans = self.candidate_spans(&norm_words);
        if spans.is_empty() {
            return text.to_string();
        }

        // merge candidate spans into padded windows, tracking the best
        // (lowest) distance ratio inside each window. Ambiguous words get wide
        // context: the surrounding phrasing is what disambiguates them.
        const AMBIGUOUS_CONTEXT: usize = 4;
        let mut windows: Vec<(usize, usize, f32)> = Vec::new();
        let mut sorted = spans;
        sorted.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        for (s, e, r) in sorted {
            let ctx = if norm_words[s..e]
                .iter()
                .any(|w| self.ambiguous.contains(w.as_str()))
            {
                AMBIGUOUS_CONTEXT
            } else {
                WINDOW_CONTEXT
            };
            let ws = s.saturating_sub(ctx);
            let we = (e + ctx).min(raw_words.len());
            match windows.last_mut() {
                Some((_, le, lr)) if ws <= *le + MERGE_GAP => {
                    *le = (*le).max(we);
                    *lr = lr.min(r);
                }
                _ => windows.push((ws, we, r)),
            }
        }

        // latency budget: process only the most garble-like windows (true
        // garbles rank near 0.2–0.3; borderline screen fires near REL_DIST)
        if windows.len() > MAX_WINDOWS {
            windows.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
            windows.truncate(MAX_WINDOWS);
        }
        windows.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

        let mut out: Vec<String> = raw_words.iter().map(|w| w.to_string()).collect();
        // right-to-left so earlier indices stay valid while splicing
        for (ws, we, _) in windows.into_iter().rev() {
            let window_raw = raw_words[ws..we].join(" ");
            match self.generate(&window_raw) {
                Ok(p) => {
                    let corrected = constrained_apply(&window_raw, &p, &self.catalog);
                    if corrected != window_raw {
                        out.splice(ws..we, corrected.split_whitespace().map(String::from));
                    }
                }
                Err(e) => {
                    log::warn!("corrector generation failed ({e}); keeping raw window");
                }
            }
        }
        out.join(" ")
    }

    /// Greedy seq2seq generation of the LM proposal.
    fn generate(&mut self, text: &str) -> Result<String, String> {
        let enc = self
            .tokenizer
            .encode(format!("{PREFIX}{text}"), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let input_ids = Tensor::new(enc.get_ids(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("input tensor: {e}"))?;

        self.model.clear_kv_cache();
        let encoder_output = self
            .model
            .encode(&input_ids)
            .map_err(|e| format!("encode: {e}"))?;

        // a correction is ≈ the input length; the margin absorbs split brands
        let max_new = (enc.get_ids().len() + 16).min(MAX_NEW_TOKENS);
        let mut output_ids: Vec<u32> = vec![self.decoder_start_token_id];
        for step in 0..max_new {
            let decoder_input = if step == 0 {
                Tensor::new(output_ids.as_slice(), &self.device)
            } else {
                Tensor::new(&[*output_ids.last().unwrap()], &self.device)
            }
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("decoder tensor: {e}"))?;

            let logits = self
                .model
                .decode(&decoder_input, &encoder_output)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| format!("decode: {e}"))?;
            let logits: Vec<f32> = logits
                .to_dtype(DType::F32)
                .and_then(|t| t.to_vec1())
                .map_err(|e| format!("logits: {e}"))?;

            let next = argmax_f32(&logits) as u32;
            if next == self.eos_token_id {
                break;
            }
            output_ids.push(next);
        }

        self.tokenizer
            .decode(&output_ids[1..], true)
            .map_err(|e| format!("detokenize: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Catalog acceptance filter (pure text logic)
// ---------------------------------------------------------------------------

/// Catalog brands that are also common Russian words — never auto-correct to
/// these (they collide with pronouns/greetings/ordinary nouns).
const STOP: &[&str] = &[
    "я", "добрый", "любимый", "рич", "чудо-ягода", "монарх", "эгоист",
    "домашний", "крымская", "боровая", "бархатные", "подводный", "сады",
    "мия", "аквариум",
];

fn load_catalog(path: &str) -> Result<Vec<(String, String)>, String> {
    let data =
        std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut cat: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for ln in data.lines() {
        let ln = ln.trim();
        if ln.is_empty() || ln.starts_with('#') {
            continue;
        }
        let canon = ln.split(" | ").next().unwrap_or("").trim();
        let n = normalize(canon);
        if n.is_empty()
            || STOP.contains(&n.as_str())
            || (!n.contains(' ') && n.chars().count() < 4)
            || !seen.insert(n.clone())
        {
            continue;
        }
        cat.push((n, canon.to_string()));
    }
    // longest normalized form first so multi-word brands match before parts
    cat.sort_by_key(|(n, _)| std::cmp::Reverse(n.chars().count()));
    Ok(cat)
}

/// Lowercase, ё→е, keep only Cyrillic а–я, Latin a–z, digits and spaces —
/// mirrors the normalization the corrector was trained/evaluated with, extended
/// with Latin so Latin catalog brands ("Snickers") are comparable too.
fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let c = match c {
            'ё' | 'Ё' => 'е',
            _ => c,
        };
        for lc in c.to_lowercase() {
            if ('а'..='я').contains(&lc) || lc.is_ascii_lowercase() || lc.is_ascii_digit() {
                out.push(lc);
            } else if lc == ' ' && !out.ends_with(' ') {
                out.push(' ');
            }
        }
    }
    out.trim().to_string()
}

/// Pre-filter distance bound for a screen key of length `len` — exactly the
/// acceptance filter's `REL_DIST` bound: any span the constraint could later
/// accept is within it, and anything looser only costs false positives (texts
/// that pay for T5 generation and then have every edit rejected anyway).
fn screen_limit(len: usize) -> usize {
    ((REL_DIST * len as f32).round() as usize).max(1)
}

/// Space-stripped catalog keys (plus Cyrillic transliterations of Latin
/// brands), grouped into buckets by char length for cheap length gating.
fn build_screen(catalog: &[(String, String)]) -> Vec<Vec<String>> {
    let mut keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (n, _) in catalog {
        let joined: String = n.chars().filter(|c| *c != ' ').collect();
        if joined.chars().any(|c| c.is_ascii_lowercase()) {
            keys.insert(translit_to_cyrillic(&joined));
        }
        keys.insert(joined);
    }
    let max_len = keys.iter().map(|k| k.chars().count()).max().unwrap_or(0);
    let mut buckets = vec![Vec::new(); max_len + 1];
    for k in keys {
        let l = k.chars().count();
        buckets[l].push(k);
    }
    buckets
}

/// Is `levenshtein(a, b) <= limit`? Banded DP with early exit — O(len·limit)
/// instead of O(len²), and bails as soon as the whole band exceeds `limit`.
fn levenshtein_within(a: &str, b: &str, limit: usize) -> bool {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > limit {
        return false;
    }
    let big = limit + 1;
    let mut prev: Vec<usize> = (0..=b.len()).map(|j| j.min(big)).collect();
    let mut cur = vec![big; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = (i + 1).min(big);
        let lo = (i + 1).saturating_sub(limit).max(1);
        let hi = (i + 1 + limit).min(b.len());
        let mut row_min = cur[0];
        for j in lo..=hi {
            let cost = usize::from(*ca != b[j - 1]);
            let v = (prev[j - 1] + cost)
                .min(prev[j] + 1)
                .min(cur[j - 1] + 1)
                .min(big);
            cur[j] = v;
            if v < row_min {
                row_min = v;
            }
        }
        if lo > 1 {
            cur[lo - 1] = big;
        }
        if row_min > limit {
            return false;
        }
        std::mem::swap(&mut prev, &mut cur);
        for v in cur.iter_mut() {
            *v = big;
        }
    }
    prev[b.len()] <= limit
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Latin → Cyrillic phonetic transliteration (digraphs first). The ASR writes
/// spoken foreign brands in Cyrillic, so phonetic closeness between a raw span
/// ("сникерс") and a Latin catalog brand ("snickers") must be measured on the
/// brand's Cyrillic rendering. Mirrors the training-data generator.
const TRANSLIT: &[(&str, &str)] = &[
    ("shch", "щ"), ("sch", "ш"), ("sh", "ш"), ("tch", "ч"), ("ch", "ч"),
    ("kh", "х"), ("zh", "ж"), ("th", "т"), ("ph", "ф"), ("ck", "к"),
    ("qu", "кв"), ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"),
    ("ai", "ай"), ("ay", "ей"), ("ey", "ей"), ("oy", "ой"), ("ya", "я"),
    ("yu", "ю"), ("yo", "е"), ("ja", "джа"), ("je", "дже"), ("jo", "джо"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"),
    ("g", "г"), ("h", "х"), ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"),
    ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"),
    ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("x", "кс"),
    ("y", "и"), ("z", "з"),
];

fn translit_to_cyrillic(s: &str) -> String {
    let mut out = s.to_string();
    for (lat, cyr) in TRANSLIT {
        out = out.replace(lat, cyr);
    }
    out
}

/// If `cand_norm` (LM proposal span) is a catalog brand (space-insensitively)
/// and `orig_norm` (raw span) is phonetically close to it, return the canonical
/// cased brand. Closeness for Latin brands is measured against their Cyrillic
/// transliteration as well (the ASR writes spoken foreign names in Cyrillic).
fn match_brand<'a>(
    cand_norm: &str,
    orig_norm: &str,
    catalog: &'a [(String, String)],
) -> Option<&'a str> {
    if cand_norm.is_empty() {
        return None;
    }
    let cand_joined: String = cand_norm.chars().filter(|c| *c != ' ').collect();
    let hit = catalog.iter().find(|(n, _)| {
        n == cand_norm || n.chars().filter(|c| *c != ' ').eq(cand_joined.chars())
    })?;
    let brand_joined: String = hit.0.chars().filter(|c| *c != ' ').collect();
    let orig_joined: String = orig_norm.chars().filter(|c| *c != ' ').collect();
    let mut d = levenshtein(&orig_joined, &brand_joined);
    let mut blen = brand_joined.chars().count();
    if brand_joined.chars().any(|c| c.is_ascii_lowercase()) {
        let cyr = translit_to_cyrillic(&brand_joined);
        let d2 = levenshtein(&orig_joined, &cyr);
        if d2 < d {
            d = d2;
            blen = cyr.chars().count();
        }
    }
    let limit = ((REL_DIST * blen as f32).round() as usize).max(1);
    (d <= limit).then_some(hit.1.as_str())
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Op {
    Equal,
    Replace,
    Other, // insert / delete — never accepted
}

/// Word-level opcodes between two normalized token slices, via LCS.
fn opcodes(a: &[String], b: &[String]) -> Vec<(Op, usize, usize, usize, usize)> {
    let (n, m) = (a.len(), b.len());
    // LCS length table
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    // walk the table, emitting equal runs and change blocks
    let mut ops = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    let (mut bi, mut bj) = (0usize, 0usize); // start of a pending change block
    let flush = |ops: &mut Vec<(Op, usize, usize, usize, usize)>,
                     bi: usize,
                     i: usize,
                     bj: usize,
                     j: usize| {
        if bi == i && bj == j {
            return;
        }
        let op = if bi < i && bj < j { Op::Replace } else { Op::Other };
        ops.push((op, bi, i, bj, j));
    };
    while i < n && j < m {
        if a[i] == b[j] {
            flush(&mut ops, bi, i, bj, j);
            let (si, sj) = (i, j);
            while i < n && j < m && a[i] == b[j] {
                i += 1;
                j += 1;
            }
            ops.push((Op::Equal, si, i, sj, j));
            bi = i;
            bj = j;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    flush(&mut ops, bi, n, bj, m);
    ops
}

/// Accept only replace-blocks that map a phonetically-close raw span to a real
/// catalog brand; keep the raw hypothesis for everything else.
pub fn constrained_apply(raw_hyp: &str, lm_output: &str, catalog: &[(String, String)]) -> String {
    let h: Vec<&str> = raw_hyp.split_whitespace().collect();
    let c: Vec<&str> = lm_output.split_whitespace().collect();
    let hn: Vec<String> = h.iter().map(|t| normalize(t)).collect();
    let cn: Vec<String> = c.iter().map(|t| normalize(t)).collect();

    let mut out: Vec<String> = Vec::with_capacity(h.len());
    for (op, i1, i2, j1, j2) in opcodes(&hn, &cn) {
        match op {
            Op::Equal => out.extend(h[i1..i2].iter().map(|s| s.to_string())),
            Op::Replace => {
                let cand = cn[j1..j2].join(" ");
                let orig = hn[i1..i2].join(" ");
                if let Some(brand) = match_brand(cand.trim(), orig.trim(), catalog) {
                    // carry trailing punctuation of the raw span
                    let last = h[i2 - 1];
                    let tail: String = last
                        .chars()
                        .skip_while(|ch| ch.is_alphanumeric())
                        .collect();
                    out.push(format!("{brand}{tail}"));
                } else {
                    out.extend(h[i1..i2].iter().map(|s| s.to_string()));
                }
            }
            Op::Other => out.extend(h[i1..i2].iter().map(|s| s.to_string())),
        }
    }
    out.join(" ")
}

fn argmax_f32(v: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat() -> Vec<(String, String)> {
        [
            "Водовоз",
            "Аква Ареал",
            "Snickers",
            "Сенежская",
        ]
        .iter()
        .map(|b| (normalize(b), b.to_string()))
        .collect()
    }

    #[test]
    fn normalize_keeps_cyrillic_latin_digits() {
        assert_eq!(normalize("Аква Ареал, 0.5Л!"), "аква ареал 05л");
        assert_eq!(normalize("Ёлки"), "елки");
        assert_eq!(normalize("Snickers"), "snickers");
    }

    #[test]
    fn accepts_close_brand_fix() {
        let out = constrained_apply(
            "компания Вотовос, здравствуйте",
            "компания Водовоз, здравствуйте",
            &cat(),
        );
        assert_eq!(out, "компания Водовоз, здравствуйте");
    }

    #[test]
    fn accepts_merged_multiword_brand() {
        let out = constrained_apply(
            "добавьте аквареал в заказ",
            "добавьте Аква Ареал в заказ",
            &cat(),
        );
        assert_eq!(out, "добавьте Аква Ареал в заказ");
    }

    #[test]
    fn accepts_translit_to_latin_brand() {
        let out = constrained_apply(
            "один сникерс пожалуйста",
            "один Snickers пожалуйста",
            &cat(),
        );
        assert_eq!(out, "один Snickers пожалуйста");
    }

    #[test]
    fn reverts_truncation() {
        let out = constrained_apply(
            "сумма 20 300 рублей телефон 925",
            "сумма 20 300",
            &cat(),
        );
        assert_eq!(out, "сумма 20 300 рублей телефон 925");
    }

    #[test]
    fn rejects_far_word_to_brand() {
        // "онежская" -> "Сенежская" is 1 edit on an 9-char brand: accepted;
        // but a genuinely distant word must be kept.
        let out = constrained_apply(
            "улица Пушкина дом три",
            "улица Сенежская дом три",
            &cat(),
        );
        assert_eq!(out, "улица Пушкина дом три");
    }

    #[test]
    fn keeps_plain_rewrites_out() {
        let out = constrained_apply(
            "добрый день компания",
            "Добрый день компания",
            &cat(),
        );
        assert_eq!(out, "добрый день компания");
    }

    #[test]
    fn banded_levenshtein_matches_full() {
        let words = ["", "а", "водовоз", "вотовос", "сникерс", "snickers",
                     "аквариал", "акваареал", "мама", "рама"];
        for a in words {
            for b in words {
                let d = levenshtein(a, b);
                for limit in 0..6 {
                    assert_eq!(
                        levenshtein_within(a, b, limit),
                        d <= limit,
                        "a={a} b={b} limit={limit} d={d}"
                    );
                }
            }
        }
    }

    #[test]
    fn screen_finds_garbles_and_skips_plain_text() {
        let screen = build_screen(&cat());
        let c = |text: &str| {
            // standalone screen check mirroring has_brand_candidate
            let words: Vec<String> =
                normalize(text).split_whitespace().map(|w| w.to_string()).collect();
            for n in 1..=4usize {
                for win in words.windows(n) {
                    let span: String = win.concat();
                    if span.chars().count() < 3 {
                        continue;
                    }
                    let sl = span.chars().count();
                    for (len, bucket) in screen.iter().enumerate() {
                        let limit = screen_limit(len);
                        if len.abs_diff(sl) > limit {
                            continue;
                        }
                        for key in bucket {
                            if levenshtein_within(&span, key, limit) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        };
        assert!(c("компания Вотовос слушает"));          // garbled brand
        assert!(c("один сникерс пожалуйста"));            // translit of Latin brand
        assert!(c("добавьте аква ареал в заказ"));        // exact multiword brand
        assert!(!c("да хорошо спасибо до свидания"));     // nothing brand-like
        assert!(!c("привезите завтра к девяти утра"));    // nothing brand-like
    }
}
