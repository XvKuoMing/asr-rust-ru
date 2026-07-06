//! Smoke-test / benchmark for the embedded brand corrector (no server).
//!
//! Usage:
//!     cargo run --release --example corrector_demo [CORRECTOR_DIR] ["text"]
//!     cargo run --release --features corrector-cuda --example corrector_demo
//!
//! Loads the corrector model directory (default `corrector/`, see
//! `scripts/prepare_corrector.py`), prints raw -> corrected for sample
//! utterances (or the one you pass), then reports steady-state latency
//! (median of repeated runs after a warmup pass).

use asr_rust::corrector::Corrector;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "corrector".to_string());
    let custom = args.next();

    let t0 = std::time::Instant::now();
    let mut c = match Corrector::load(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load corrector from {dir}/: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "corrector loaded from {dir}/ ({} catalog brands) in {:.1}s\n",
        c.catalog_len(),
        t0.elapsed().as_secs_f64()
    );

    // (label, text). The last two contain nothing brand-like and should take
    // the pre-filter fast path (microseconds, no T5 generation).
    let samples: Vec<(&str, String)> = match custom {
        Some(s) => vec![("custom", s)],
        None => vec![
            ("brand-garble ", "Добрый день, компания Вотовос, меня зовут Ирина.".into()),
            ("brand-merged ", "Добавьте, пожалуйста, аквариал ноль пять, шесть бутылочек.".into()),
            ("brand-translit", "Один сникерс и вода архыз газированная.".into()),
            ("numbers      ", "Сумма заказа 20 300 рублей, телефон 925 9010.".into()),
            ("no-brand     ", "Да, хорошо, спасибо большое, до свидания.".into()),
            ("no-brand-long", "Привезите, пожалуйста, завтра к девяти утра, я буду дома после восьми, домофон работает.".into()),
        ],
    };

    // warmup (first run pays kernel/JIT/cache init)
    for (_, s) in &samples {
        let _ = c.correct(s);
    }

    const REPS: usize = 5;
    println!("steady-state latency (median of {REPS} runs):\n");
    for (label, s) in &samples {
        let mut times = Vec::with_capacity(REPS);
        let mut out = String::new();
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            out = c.correct(s);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[REPS / 2];
        println!("[{label}] {med:8.1} ms  raw : {s}");
        println!("{:>25}corr: {out}\n", "");
    }
}
