"""
Drop-in translation of FLUX.2 Klein ComfyUI API workflows to FLUX.3 Image Klein.

Lets the same workflow JSON that runs on the FLUX.2 endpoint run unchanged on
this endpoint, so A/B tests compare models and not workflow differences.

Only nodes that clearly reference FLUX.2 are touched:
  * UNETLoader / UnetLoaderGGUF whose unet_name contains "flux-2"/"flux2"
      -> unet_name = FLUX.3 checkpoint (base or distilled)
  * CLIPLoader with type "flux2"
      -> type "flux3", clip_name = Qwen3-VL 4B text encoder
  * KSampler / KSamplerAdvanced (only when the variant is "distilled", or
    when steps/cfg/sampler/scheduler overrides are given explicitly)

Everything else (prompts, seeds, resolution, ReferenceLatent, VAE, SaveImage)
is left exactly as sent. FLUX.3 uses the same flux2-vae and the same
128-channel latent, so EmptyFlux2LatentImage / ReferenceLatent keep working.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

FLUX3_UNET_BASE_DEFAULT = "flux3-image-klein-base-bf16.safetensors"
FLUX3_UNET_DISTILLED_DEFAULT = "flux3-image-klein-bf16.safetensors"
FLUX3_CLIP_DEFAULT = "qwen3vl_4b_int8_convrot.safetensors"
DISTILLED_STEPS_DEFAULT = 16
DISTILLED_CFG = 1.0

UNET_LOADERS = {"UNETLoader", "UnetLoaderGGUF"}
SAMPLERS = {"KSampler", "KSamplerAdvanced"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _looks_flux2(name: object) -> bool:
    if not isinstance(name, str):
        return False
    n = name.lower()
    return ("flux-2" in n or "flux2" in n or "flux_2" in n) and "flux3" not in n


@dataclass
class RewriteOptions:
    enabled: bool = True
    variant: str = "base"  # "base" | "distilled"
    unet: str | None = None
    clip: str | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler_name: str | None = None
    scheduler: str | None = None

    @property
    def unet_name(self) -> str:
        if self.unet:
            return self.unet
        if self.variant == "distilled":
            return os.environ.get("FLUX3_UNET_DISTILLED", FLUX3_UNET_DISTILLED_DEFAULT)
        return os.environ.get("FLUX3_UNET_BASE", FLUX3_UNET_BASE_DEFAULT)

    @property
    def clip_name(self) -> str:
        return self.clip or os.environ.get("FLUX3_CLIP", FLUX3_CLIP_DEFAULT)


def options_from(job_flux3: object) -> RewriteOptions:
    """Build options from env defaults, then the per-request `input.flux3` dict."""
    opts = RewriteOptions(
        enabled=_env_bool("FLUX3_REWRITE", True),
        variant=os.environ.get("FLUX3_VARIANT", "base").strip().lower() or "base",
    )
    if os.environ.get("FLUX3_STEPS"):
        opts.steps = int(os.environ["FLUX3_STEPS"])
    if os.environ.get("FLUX3_CFG"):
        opts.cfg = float(os.environ["FLUX3_CFG"])

    if isinstance(job_flux3, dict):
        if "enabled" in job_flux3:
            opts.enabled = bool(job_flux3["enabled"])
        if job_flux3.get("variant"):
            opts.variant = str(job_flux3["variant"]).strip().lower()
        for key in ("unet", "clip", "sampler_name", "scheduler"):
            if job_flux3.get(key):
                setattr(opts, key, str(job_flux3[key]))
        if job_flux3.get("steps") is not None:
            opts.steps = int(job_flux3["steps"])
        if job_flux3.get("cfg") is not None:
            opts.cfg = float(job_flux3["cfg"])

    if opts.variant not in ("base", "distilled"):
        raise ValueError(
            f"flux3.variant must be 'base' or 'distilled', got {opts.variant!r}"
        )
    return opts


@dataclass
class RewriteResult:
    workflow: dict
    changes: list[str] = field(default_factory=list)

    @property
    def touched_model(self) -> bool:
        return any(c.startswith(("unet:", "clip:")) for c in self.changes)


def rewrite_workflow(workflow: dict, opts: RewriteOptions) -> RewriteResult:
    """Return a translated copy of `workflow` and a human-readable change list."""
    if not opts.enabled or not isinstance(workflow, dict):
        return RewriteResult(workflow=workflow)

    wf = copy.deepcopy(workflow)
    changes: list[str] = []

    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        if ctype in UNET_LOADERS and _looks_flux2(inputs.get("unet_name")):
            old = inputs["unet_name"]
            # GGUF loader can't read safetensors checkpoints -> plain UNETLoader
            node["class_type"] = "UNETLoader"
            node["inputs"] = {"unet_name": opts.unet_name, "weight_dtype": "default"}
            changes.append(f"unet:{node_id}:{old}->{opts.unet_name}")

        elif ctype == "CLIPLoader" and str(inputs.get("type", "")).lower() == "flux2":
            old = inputs.get("clip_name")
            inputs["clip_name"] = opts.clip_name
            inputs["type"] = "flux3"
            changes.append(f"clip:{node_id}:{old}->{opts.clip_name}")

    # Sampler settings: only adjust when we actually switched the model, or
    # when the caller explicitly asked for overrides.
    model_switched = any(c.startswith("unet:") for c in changes)
    explicit = any(
        v is not None for v in (opts.steps, opts.cfg, opts.sampler_name, opts.scheduler)
    )
    if model_switched or explicit:
        steps = opts.steps
        cfg = opts.cfg
        if opts.variant == "distilled":
            steps = steps if steps is not None else DISTILLED_STEPS_DEFAULT
            cfg = cfg if cfg is not None else DISTILLED_CFG
        for node_id, node in wf.items():
            if not isinstance(node, dict) or node.get("class_type") not in SAMPLERS:
                continue
            inputs = node.get("inputs") or {}
            for key, value in (
                ("steps", steps),
                ("cfg", cfg),
                ("sampler_name", opts.sampler_name),
                ("scheduler", opts.scheduler),
            ):
                if value is None or isinstance(inputs.get(key), list):
                    continue  # unset, or wired from another node
                if inputs.get(key) != value:
                    changes.append(f"{key}:{node_id}:{inputs.get(key)}->{value}")
                    inputs[key] = value
            if node.get("class_type") == "KSamplerAdvanced" and steps is not None:
                end = inputs.get("end_at_step")
                if isinstance(end, int) and end > steps:
                    inputs["end_at_step"] = 10000

    return RewriteResult(workflow=wf, changes=changes)
