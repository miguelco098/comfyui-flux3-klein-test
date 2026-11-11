#!/usr/bin/env python3
"""
A/B test: FLUX.2 Klein endpoint vs FLUX.3 Klein endpoint, same workflow,
same prompts, same seeds. Standard library only.

Usage
-----
  export RUNPOD_API_KEY=...            # never hard-code it
  python ab_test.py \
      --flux2-endpoint afn0htiebzt6i2 \
      --flux3-endpoint <NEW_ENDPOINT_ID> \
      --workflow flux2_klein_base_t2i_api.json \
      --prompts prompts_example.txt \
      --variants base,distilled \
      --out ab_results

--workflow is the FLUX.2 API workflow your agent already sends. The FLUX.3
endpoint rewrites it server-side (models + text encoder; for "distilled" also
steps/cfg), so both sides get the identical graph, prompt, seed and size.

Outputs (in --out):
  results.csv     one row per prompt x model: status, execution/delay/total seconds, cost
  summary.json    averages per model + estimated cost per 1000 images
  report.html     side-by-side gallery for visual review
  <n>_<model>.png the images
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://api.runpod.ai/v2"


def call(method: str, url: str, key: str, body: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {e.read()[:500]!r}") from None
        except urllib.error.URLError:
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def run_job(endpoint: str, key: str, payload: dict, poll_s: float, max_wait_s: int) -> dict:
    started = time.time()
    job = call("POST", f"{API}/{endpoint}/run", key, payload)
    job_id = job["id"]
    while True:
        st = call("GET", f"{API}/{endpoint}/status/{job_id}", key)
        if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            st["_wall_s"] = round(time.time() - started, 2)
            st["_job_id"] = job_id
            return st
        if time.time() - started > max_wait_s:
            call("POST", f"{API}/{endpoint}/cancel/{job_id}", key)
            return {"status": "CLIENT_TIMEOUT", "_job_id": job_id, "_wall_s": max_wait_s}
        time.sleep(poll_s)


def set_prompt(workflow: dict, node: str | None, prompt: str, seed: int | None) -> dict:
    wf = json.loads(json.dumps(workflow))
    if node is None:  # first CLIPTextEncode with non-empty text = positive prompt
        for nid, n in wf.items():
            if n.get("class_type") == "CLIPTextEncode" and isinstance(n["inputs"].get("text"), str) and n["inputs"]["text"].strip():
                node = nid
                break
    if node is None or node not in wf:
        raise SystemExit("Could not find the positive prompt node; pass --prompt-node")
    wf[node]["inputs"]["text"] = prompt
    if seed is not None:
        for n in wf.values():
            if n.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                key = "seed" if "seed" in n["inputs"] else "noise_seed"
                if not isinstance(n["inputs"].get(key), list):
                    n["inputs"][key] = seed
            if n.get("class_type") == "RandomNoise" and not isinstance(n["inputs"].get("noise_seed"), list):
                n["inputs"]["noise_seed"] = seed
    return wf


def save_images(status: dict, path_prefix: str) -> list[str]:
    out = status.get("output") or {}
    saved = []
    for i, img in enumerate(out.get("images") or []):
        name = f"{path_prefix}{'' if i == 0 else f'_{i}'}.png"
        if img.get("type") == "base64":
            with open(name, "wb") as f:
                f.write(base64.b64decode(img["data"]))
            saved.append(name)
        elif img.get("type") == "s3_url":
            saved.append(img["data"])
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flux2-endpoint", required=True)
    ap.add_argument("--flux3-endpoint", required=True)
    ap.add_argument("--workflow", required=True, help="FLUX.2 API workflow JSON")
    ap.add_argument("--prompts", required=True, help="text file, one prompt per line")
    ap.add_argument("--prompt-node", default=None, help="node id of the positive prompt (auto-detected)")
    ap.add_argument("--seed", type=int, default=123456789, help="base seed; prompt i uses seed+i")
    ap.add_argument("--variants", default="base", help="FLUX.3 variants to test: base,distilled")
    ap.add_argument("--distilled-steps", type=int, default=None, help="override steps for distilled")
    ap.add_argument("--images", default=None, help="optional JSON file with worker-comfyui 'images' list (reference images)")
    ap.add_argument("--rate-per-hour", type=float, default=1.79, help="$/h of the GPU tier, for cost estimates")
    ap.add_argument("--parallel", type=int, default=2, help="concurrent jobs per endpoint")
    ap.add_argument("--max-wait", type=int, default=1800)
    ap.add_argument("--out", default="ab_results")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("Set RUNPOD_API_KEY in the environment.", file=sys.stderr)
        return 2

    with open(args.workflow) as f:
        workflow = json.load(f)
    with open(args.prompts) as f:
        prompts = [p.strip() for p in f if p.strip() and not p.startswith("#")]
    extra_images = None
    if args.images:
        with open(args.images) as f:
            extra_images = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    arms = [("flux2", args.flux2_endpoint, None)]
    for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
        flux3 = {"variant": v}
        if v == "distilled" and args.distilled_steps:
            flux3["steps"] = args.distilled_steps
        arms.append((f"flux3_{v}", args.flux3_endpoint, flux3))

    tasks = []
    for i, prompt in enumerate(prompts):
        wf = set_prompt(workflow, args.prompt_node, prompt, args.seed + i)
        for name, endpoint, flux3 in arms:
            payload = {"input": {"workflow": wf}}
            if extra_images:
                payload["input"]["images"] = extra_images
            if flux3:
                payload["input"]["flux3"] = flux3
            tasks.append((i, prompt, name, endpoint, payload))

    rows = []

    def work(t):
        i, prompt, name, endpoint, payload = t
        print(f"[{name}] prompt {i} queued", flush=True)
        st = run_job(endpoint, key, payload, poll_s=2.0, max_wait_s=args.max_wait)
        files = save_images(st, os.path.join(args.out, f"{i:03d}_{name}")) if st.get("status") == "COMPLETED" else []
        exec_s = (st.get("executionTime") or 0) / 1000
        delay_s = (st.get("delayTime") or 0) / 1000
        row = {
            "prompt_idx": i,
            "model": name,
            "status": st.get("status"),
            "execution_s": round(exec_s, 2),
            "delay_s": round(delay_s, 2),
            "wall_s": st.get("_wall_s"),
            "cost_usd": round(exec_s * args.rate_per_hour / 3600, 5),
            "image": files[0] if files else "",
            "job_id": st.get("_job_id"),
            "error": json.dumps((st.get("output") or {}).get("errors") or st.get("error") or "")[:300],
            "flux3_changes": "; ".join(((st.get("output") or {}).get("flux3") or {}).get("changes") or []),
            "prompt": prompt,
        }
        print(f"[{name}] prompt {i} {row['status']} exec={row['execution_s']}s delay={row['delay_s']}s", flush=True)
        return row

    # interleave arms so cold starts hit both endpoints similarly
    with ThreadPoolExecutor(max_workers=max(1, args.parallel) * len(arms)) as pool:
        rows = list(pool.map(work, tasks))

    rows.sort(key=lambda r: (r["prompt_idx"], r["model"]))
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for name, _, _ in arms:
        ok = [r for r in rows if r["model"] == name and r["status"] == "COMPLETED"]
        ex = [r["execution_s"] for r in ok]
        summary[name] = {
            "completed": len(ok),
            "failed": sum(1 for r in rows if r["model"] == name and r["status"] != "COMPLETED"),
            "exec_mean_s": round(statistics.mean(ex), 2) if ex else None,
            "exec_median_s": round(statistics.median(ex), 2) if ex else None,
            "cost_per_1000_usd": round(statistics.mean(ex) * args.rate_per_hour / 3600 * 1000, 2) if ex else None,
        }
    summary["_note"] = (
        "execution_s comes from RunPod (billed GPU time per job). The first job per worker "
        "includes model loading; judge speed on the median. Cost assumes "
        f"${args.rate_per_hour}/h for every job."
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    names = [a[0] for a in arms]
    cells = []
    for i, prompt in enumerate(prompts):
        tds = []
        for n in names:
            r = next((r for r in rows if r["prompt_idx"] == i and r["model"] == n), None)
            img = os.path.basename(r["image"]) if r and r["image"] and not r["image"].startswith("http") else (r["image"] if r else "")
            tds.append(
                f"<td><div class=m>{n} · {r['status'] if r else ''} · {r['execution_s'] if r else ''}s</div>"
                + (f"<a href='{html.escape(img)}'><img src='{html.escape(img)}'></a>" if img else "<div class=x>sin imagen</div>")
                + "</td>"
            )
        cells.append(f"<tr><th colspan={len(names)}>{i}: {html.escape(prompt)}</th></tr><tr>{''.join(tds)}</tr>")
    with open(os.path.join(args.out, "report.html"), "w") as f:
        f.write(
            "<!doctype html><meta charset=utf-8><title>FLUX.2 vs FLUX.3 A/B</title>"
            "<style>body{font:14px system-ui;margin:16px;background:#111;color:#eee}"
            "table{border-collapse:collapse;width:100%}th{text-align:left;padding:12px 4px 4px;color:#9cf}"
            "td{vertical-align:top;padding:4px}img{width:100%;border-radius:6px}.m{color:#aaa;margin:2px 0}"
            "pre{background:#222;padding:8px;border-radius:6px}</style>"
            f"<h1>FLUX.2 Klein vs FLUX.3 Klein</h1><pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
            f"<table>{''.join(cells)}</table>"
        )

    print(json.dumps(summary, indent=2))
    print(f"\nReport: {os.path.join(args.out, 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
