//! Smoke-test the embedded brand corrector without starting the server.
//!
//! Usage:
//!     cargo run --example corrector_demo [CORRECTOR_DIR] ["text to correct"]
//!
//! Loads the corrector model directory (default `corrector/`, see
//! `scripts/prepare_corrector.py`) and prints raw -> corrected for a few
//! sample utterances (or the one you pass).

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

    let samples: Vec<String> = match custom {
        Some(s) => vec![s],
        None => [
            "Добрый день, компания Вотовос, меня зовут Ирина.",
            "Добавьте, пожалуйста, аквариал ноль пять, шесть бутылочек.",
            "Один сникерс и вода архыз газированная.",
            "Сумма заказа 20 300 рублей, телефон 925 9010.",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    };

    for s in samples {
        let t = std::time::Instant::now();
        let out = c.correct(&s);
        println!("raw : {s}");
        println!("corr: {out}   ({:.0} ms)\n", t.elapsed().as_secs_f64() * 1000.0);
    }
}
