#!/usr/bin/env python3
"""Assemble the corrector model directory the server loads (CORRECTOR_DIR).

Usage:
    python prepare_corrector.py <trained_hf_dir> <brands_catalog.txt> [out_dir]

<trained_hf_dir>  fine-tuned T5 corrector export (from the training repo, e.g.
                  checkpoints/brand_corrector_v2): model.safetensors,
                  config.json, tokenizer.json are required.
<brands_catalog>  brand catalog, one canonical phrase per line
                  ('canonical | alias' lines and '#' comments allowed).
[out_dir]         destination (default ./corrector, next to ./weights).

The server enables correction for requests whose model name ends in '-lmcorr'.
"""
import shutil
import sys
from pathlib import Path

REQUIRED = ["model.safetensors", "config.json", "tokenizer.json"]
OPTIONAL = ["generation_config.json", "tokenizer_config.json", "special_tokens_map.json"]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    brands = Path(sys.argv[2])
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "corrector")

    missing = [f for f in REQUIRED if not (src / f).exists()]
    if missing:
        sys.exit(f"error: {src} is missing {missing}")
    if not brands.exists():
        sys.exit(f"error: {brands} not found")

    out.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED + [f for f in OPTIONAL if (src / f).exists()]:
        shutil.copy2(src / f, out / f)
    shutil.copy2(brands, out / "brands.txt")
    print(f"corrector dir ready: {out.resolve()}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:28s} {p.stat().st_size:>12,} B")


if __name__ == "__main__":
    main()
