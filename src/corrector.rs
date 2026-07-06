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

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[format!("{dir}/model.safetensors")],
                DType::F32,
                &device,
            )
            .map_err(|e| format!("load model.safetensors: {e}"))?
        };
        let model = t5::T5ForConditionalGeneration::load(vb, &config)
            .map_err(|e| format!("build T5: {e}"))?;

        let tokenizer = HfTokenizer::from_file(format!("{dir}/tokenizer.json"))
            .map_err(|e| format!("load tokenizer.json: {e}"))?;

        let catalog = load_catalog(&format!("{dir}/brands.txt"))?;
        if catalog.is_empty() {
            return Err(format!("{dir}/brands.txt contains no usable brands"));
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
        })
    }

    pub fn catalog_len(&self) -> usize {
        self.catalog.len()
    }

    /// Correct `text`: LM proposal + catalog acceptance filter.
    /// On any generation error the raw text is returned unchanged.
    pub fn correct(&mut self, text: &str) -> String {
        if text.trim().is_empty() {
            return text.to_string();
        }
        match self.generate(text) {
            Ok(proposal) => constrained_apply(text, &proposal, &self.catalog),
            Err(e) => {
                log::warn!("corrector generation failed ({e}); returning raw text");
                text.to_string()
            }
        }
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

        let mut output_ids: Vec<u32> = vec![self.decoder_start_token_id];
        for step in 0..MAX_NEW_TOKENS {
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
}
