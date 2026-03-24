# ---------------------------------------------------------------------------
# Stage 1 — Convert PyTorch checkpoint → safetensors + tokenizer
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS converter
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY scripts/ scripts/
RUN cd scripts && uv sync --locked && uv run python convert_model.py
# weights/ is created at /app/weights/

# ---------------------------------------------------------------------------
# Stage 2 — Download LibTorch CUDA distribution
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS libtorch
RUN apt-get update && apt-get install -y --no-install-recommends wget unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q -O /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.0%2Bcu128.zip" \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

# ---------------------------------------------------------------------------
# Stage 3 — Build the Rust binary against the pre-downloaded LibTorch
# ---------------------------------------------------------------------------
FROM rust:slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=libtorch /opt/libtorch /opt/libtorch
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
RUN cargo build --release

# ---------------------------------------------------------------------------
# Stage 4 — Minimal runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=libtorch   /opt/libtorch/lib  /opt/libtorch/lib
COPY --from=builder    /app/target/release/asr-rust /usr/local/bin/asr-rust
COPY --from=converter  /app/weights        /app/weights
COPY example.wav /app/example.wav

ENV LD_LIBRARY_PATH=/opt/libtorch/lib
WORKDIR /app

CMD ["asr-rust"]
