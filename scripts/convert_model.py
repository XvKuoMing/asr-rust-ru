"""
Download v3_e2e_rnnt checkpoint + tokenizer, extract config,
convert state_dict to safetensors, and write everything to weights/.
"""

import hashlib
import json
import os
import shutil
import urllib.request
import warnings

import torch
from safetensors.torch import save_file
from tqdm import tqdm

URL_DIR = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
MODEL_NAME = "v3_e2e_rnnt"
EXPECTED_HASH = "2730de7545ac43ad256485a462b0a27a"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
CACHE_DIR = os.path.expanduser("~/.cache/gigaam")


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  Already cached: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {url} ...")
    with urllib.request.urlopen(url) as src, open(dest, "wb") as out:
        total = int(src.info().get("Content-Length", 0))
        with tqdm(total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
            while True:
                buf = src.read(8192)
                if not buf:
                    break
                out.write(buf)
                bar.update(len(buf))


def md5(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def omegaconf_to_dict(cfg):
    """Recursively convert OmegaConf to plain dict/list."""
    from omegaconf import DictConfig, ListConfig
    if isinstance(cfg, DictConfig):
        return {k: omegaconf_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, ListConfig):
        return [omegaconf_to_dict(v) for v in cfg]
    return cfg


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1) Download checkpoint
    ckpt_path = os.path.join(CACHE_DIR, f"{MODEL_NAME}.ckpt")
    print("Step 1: Download checkpoint")
    download(f"{URL_DIR}/{MODEL_NAME}.ckpt", ckpt_path)

    h = md5(ckpt_path)
    assert h == EXPECTED_HASH, f"Hash mismatch: {h} != {EXPECTED_HASH}"
    print(f"  Checksum OK: {h}")

    # 2) Download tokenizer
    tok_path = os.path.join(CACHE_DIR, f"{MODEL_NAME}_tokenizer.model")
    print("Step 2: Download tokenizer")
    download(f"{URL_DIR}/{MODEL_NAME}_tokenizer.model", tok_path)
    shutil.copy2(tok_path, os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_tokenizer.model"))
    print(f"  Copied tokenizer to weights/")

    # 3) Load checkpoint
    print("Step 3: Load checkpoint and extract config")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    cfg_dict = omegaconf_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=2, default=str))

    with open(os.path.join(WEIGHTS_DIR, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)
    print(f"  Wrote config.json")

    # 4) Convert state_dict to safetensors (split encoder / head)
    print("Step 4: Convert to safetensors")
    sd = ckpt["state_dict"]

    encoder_tensors = {}
    head_tensors = {}
    preprocessor_tensors = {}
    skipped = []

    for key, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            skipped.append(key)
            continue
        t = tensor.contiguous().float()
        if key.startswith("encoder."):
            encoder_tensors[key[len("encoder."):]] = t
        elif key.startswith("head."):
            head_tensors[key[len("head."):]] = t
        elif key.startswith("preprocessor."):
            preprocessor_tensors[key[len("preprocessor."):]] = t
        else:
            skipped.append(key)

    print(f"  Encoder keys: {len(encoder_tensors)}")
    print(f"  Head keys:    {len(head_tensors)}")
    print(f"  Preproc keys: {len(preprocessor_tensors)}")
    if skipped:
        print(f"  Skipped keys: {skipped}")

    enc_path = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_encoder.safetensors")
    head_path = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_head.safetensors")

    save_file(encoder_tensors, enc_path)
    print(f"  Wrote {enc_path}")
    save_file(head_tensors, head_path)
    print(f"  Wrote {head_path}")

    if preprocessor_tensors:
        pp_path = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_preprocessor.safetensors")
        save_file(preprocessor_tensors, pp_path)
        print(f"  Wrote {pp_path}")

    # 5) Print state_dict key listing for debugging weight loading
    print("\n=== Encoder keys ===")
    for k in sorted(encoder_tensors.keys()):
        print(f"  {k}: {list(encoder_tensors[k].shape)}")

    print("\n=== Head keys ===")
    for k in sorted(head_tensors.keys()):
        print(f"  {k}: {list(head_tensors[k].shape)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
