"""
Thin wrapper around worker-comfyui's handler (shipped as /handler_base.py).

Same request/response contract as the FLUX.2 endpoint:
    {"input": {"workflow": {...API workflow...}, "images": [...optional...]}}

Extra, optional request field:
    "flux3": {"enabled": true, "variant": "base" | "distilled",
              "steps": 16, "cfg": 1.0, "sampler_name": "euler",
              "scheduler": "simple", "unet": "...", "clip": "..."}

Extra response field:
    "flux3": {"variant", "rewritten", "changes", "handler_seconds"}
"""

import time

import runpod

import handler_base as base
from flux3_rewrite import options_from, rewrite_workflow


def handler(job):
    started = time.time()
    job_input = job.get("input")
    info = {"variant": None, "rewritten": False, "changes": []}

    if isinstance(job_input, dict) and isinstance(job_input.get("workflow"), dict):
        try:
            opts = options_from(job_input.get("flux3"))
        except ValueError as exc:
            return {"error": str(exc)}
        result = rewrite_workflow(job_input["workflow"], opts)
        job_input["workflow"] = result.workflow
        info.update(
            variant=opts.variant,
            rewritten=bool(result.changes),
            changes=result.changes,
        )
        if result.changes:
            print(f"flux3 - rewrote workflow ({opts.variant}): {'; '.join(result.changes)}")
        job_input.pop("flux3", None)

    output = base.handler(job)

    if isinstance(output, dict):
        info["handler_seconds"] = round(time.time() - started, 2)
        output["flux3"] = info
    return output


if __name__ == "__main__":
    print("worker-comfyui (flux3) - Starting handler...")
    runpod.serverless.start({"handler": handler})
