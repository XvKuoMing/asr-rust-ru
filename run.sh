#!/usr/bin/env bash
set -euo pipefail

LIB_DIR=$(find target/release/build/torch-sys-*/out/libtorch/libtorch/lib -maxdepth 0 2>/dev/null | head -1)
if [ -z "$LIB_DIR" ]; then
    echo "error: LibTorch libs not found. Run 'cargo build --release' first." >&2
    exit 1
fi

export LD_LIBRARY_PATH="${LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ $# -eq 0 ]; then
    exec ./target/release/asr-rust
else
    exec "$@"
fi
