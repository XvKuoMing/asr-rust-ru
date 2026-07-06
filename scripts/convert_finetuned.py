#!/usr/bin/env python3
"""Convert a LOCAL (fine-tuned) GigaAM v3_e2e_rnnt checkpoint to the weights/
layout the Rust server loads — for when you have your own .ckpt and don't want
(or can't, geo-block) download from the Sber CDN like convert_model.py does.

Usage:
    python convert_finetuned.py <checkpoint.ckpt> <tokenizer.model> [out_dir]

<checkpoint.ckpt>   PyTorch-Lightning or native ckpt whose state_dict keys are
                    prefixed encoder. / head. (as produced by GigaAM training).
<tokenizer.model>   the v3_e2e_rnnt SentencePiece model (e.g. from
                    ~/.cache/gigaam/v3_e2e_rnnt_tokenizer.model).
[out_dir]           default ./weights
"""
import os
import shutil
import sys
import warnings

import torch
from safetensors.torch import save_file

MODEL_NAME = "v3_e2e_rnnt"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ckpt_path, tok_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "weights"
    os.makedirs(out, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    encoder, head, skipped = {}, {}, []
    for key, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            skipped.append(key)
            continue
        t = tensor.contiguous().float()
        if key.startswith("encoder."):
            encoder[key[len("encoder."):]] = t
        elif key.startswith("head."):
            head[key[len("head."):]] = t
        else:
            skipped.append(key)  # preprocessor etc. — Rust computes mel itself

    if not encoder or not head:
        sys.exit(f"error: expected encoder./head. keys, got prefixes: "
                 f"{sorted({k.split('.')[0] for k in sd})}")

    save_file(encoder, os.path.join(out, f"{MODEL_NAME}_encoder.safetensors"))
    save_file(head, os.path.join(out, f"{MODEL_NAME}_head.safetensors"))
    shutil.copy2(tok_path, os.path.join(out, f"{MODEL_NAME}_tokenizer.model"))
    print(f"weights ready in {os.path.abspath(out)}:")
    print(f"  encoder tensors: {len(encoder)} | head tensors: {len(head)} | skipped: {len(skipped)}")
    for f in sorted(os.listdir(out)):
        print(f"  {f:40s} {os.path.getsize(os.path.join(out, f)):>13,} B")


if __name__ == "__main__":
    main()
