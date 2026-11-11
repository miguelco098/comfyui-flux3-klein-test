"""
Make sure the FLUX.3 Image Klein files exist before ComfyUI starts.

Files go to the network volume (/runpod-volume/models/...), the same place the
FLUX.2 endpoint already keeps its models, so they are downloaded once and every
later worker starts without downloading anything. Without a network volume they
go to the container's own /comfyui/models (re-downloaded on every cold start).

The Hugging Face repo is gated: accept the FLUX Kommunity License at
https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy and pass a
read token as HF_TOKEN (use a RunPod secret: {{ RUNPOD_SECRET_hf_token }}).

Env:
  HF_TOKEN               read token for the gated repo (required for download)
  FLUX3_HF_REPO          default black-forest-labs/flux-3-image-klein-comfy
  FLUX3_DOWNLOAD         comma list of "<repo path>" to fetch; default below
  FLUX3_SKIP_DOWNLOAD    true -> only report what is missing
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import time

REPO = os.environ.get("FLUX3_HF_REPO", "black-forest-labs/flux-3-image-klein-comfy")

# repo path -> ComfyUI folder name on the volume (extra_model_paths.yaml maps
# unet -> diffusion_models and clip -> text_encoders)
DEFAULT_FILES = [
    "diffusion_models/flux3-image-klein-base-bf16.safetensors",
    "diffusion_models/flux3-image-klein-bf16.safetensors",
    "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    "vae/flux2-vae.safetensors",
]
FOLDER_FOR_PREFIX = {
    "diffusion_models": "unet",
    "text_encoders": "clip",
    "vae": "vae",
    "loras": "loras",
}
LOCK_STALE_SECONDS = 45 * 60


def log(msg: str) -> None:
    print(f"flux3-fetch - {msg}", flush=True)


def models_root() -> str:
    if os.path.isdir("/runpod-volume"):
        return "/runpod-volume/models"
    return "/comfyui/models"


def target_path(root: str, repo_path: str) -> str:
    prefix, name = repo_path.split("/", 1)
    folder = FOLDER_FOR_PREFIX.get(prefix, prefix)
    if root == "/comfyui/models":  # inside the container use native names
        folder = prefix
    return os.path.join(root, folder, os.path.basename(name))


def _acquire_lock(path: str) -> bool:
    """Cross-worker lock on the shared volume (O_EXCL file + staleness check)."""
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {time.time()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(path)
            except FileNotFoundError:
                continue
            if age > LOCK_STALE_SECONDS:
                log(f"removing stale lock {path} ({int(age)}s old)")
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                continue
            return False


def download(repo_path: str, dest: str, token: str) -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(dest), f".flux3-tmp-{socket.gethostname()}")
    os.makedirs(tmp_dir, exist_ok=True)
    started = time.time()
    log(f"downloading {REPO}/{repo_path} -> {dest}")
    got = hf_hub_download(
        repo_id=REPO,
        filename=repo_path,
        token=token,
        local_dir=tmp_dir,
    )
    os.replace(got, dest)  # atomic: other workers never see a partial file
    shutil.rmtree(tmp_dir, ignore_errors=True)
    size_gb = os.path.getsize(dest) / 1e9
    log(f"done {os.path.basename(dest)} ({size_gb:.1f} GB in {time.time() - started:.0f}s)")


def main() -> int:
    root = models_root()
    files = [
        f.strip()
        for f in os.environ.get("FLUX3_DOWNLOAD", ",".join(DEFAULT_FILES)).split(",")
        if f.strip()
    ]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    skip = os.environ.get("FLUX3_SKIP_DOWNLOAD", "false").lower() == "true"

    missing = [(f, target_path(root, f)) for f in files if not os.path.isfile(target_path(root, f))]
    for f in files:
        state = "MISSING" if any(m[0] == f for m in missing) else "ok"
        log(f"{state:7} {target_path(root, f)}")
    if not missing:
        log("all FLUX.3 files present")
        return 0
    if skip:
        log("FLUX3_SKIP_DOWNLOAD=true, not downloading")
        return 0
    if not token or token.startswith("{{"):
        log(
            "ERROR: HF_TOKEN is not set (or the RunPod secret did not resolve). "
            "Accept the license on huggingface.co/black-forest-labs/flux-3-image-klein-comfy, "
            "create a read token, save it as RunPod secret 'hf_token' and set the endpoint "
            "env var HF_TOKEN={{ RUNPOD_SECRET_hf_token }}. Jobs will fail the model pre-flight "
            "check until the files exist."
        )
        return 0  # don't crash-loop the worker; the handler reports missing files

    for repo_path, dest in missing:
        lock = dest + ".lock"
        while not os.path.isfile(dest):
            if _acquire_lock(lock):
                try:
                    if not os.path.isfile(dest):
                        download(repo_path, dest, token)
                except Exception as exc:  # keep the worker alive, report clearly
                    log(f"ERROR downloading {repo_path}: {exc!r}")
                    if "401" in str(exc) or "403" in str(exc) or "gated" in str(exc).lower():
                        log("-> the token can't read the gated repo: accept the license with the same HF account")
                    break
                finally:
                    try:
                        os.remove(lock)
                    except FileNotFoundError:
                        pass
            else:
                log(f"another worker is downloading {os.path.basename(dest)}, waiting...")
                time.sleep(15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
