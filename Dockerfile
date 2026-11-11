# FLUX.3 Image Klein worker for RunPod Serverless (A/B test twin of the
# FLUX.2 Klein endpoint "COMFYUI WORKFLOW V2").
#
# Same base image family and the same request/response contract as the
# FLUX.2 worker. Models live on the network volume, like the FLUX.2 endpoint.
FROM runpod/worker-comfyui:5.11.0-base

# --- 1. runpod SDK -----------------------------------------------------------
# worker-comfyui pins runpod~=1.7.12, which breaks job-take against the current
# RunPod backend ("Job has missing field(s): id or input."). Same fix as the
# FLUX.2 repo.
RUN uv pip install --upgrade "runpod>=1.12.0"

# --- 2. ComfyUI >= 0.41 ------------------------------------------------------
# The base image ships ComfyUI 0.34.0, which has no FLUX.3 support
# (the flux3 model + "flux3" text-encoder type landed in 0.41.0).
ARG COMFYUI_VERSION=v0.45.2
RUN set -eux; \
    cd /tmp; \
    wget -q "https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/${COMFYUI_VERSION}.tar.gz" -O comfy.tgz; \
    mkdir comfy-new; \
    tar -xzf comfy.tgz -C comfy-new --strip-components=1; \
    # Overlay the new code on /comfyui. Folders the base image set up
    # (models, custom_nodes, user, input, output, .venv, extra_model_paths.yaml)
    # are not shipped from the tarball, so they stay exactly as they were.
    rm -rf /tmp/comfy-new/models /tmp/comfy-new/custom_nodes /tmp/comfy-new/user \
           /tmp/comfy-new/input /tmp/comfy-new/output; \
    # drop old code packages first so no stale modules survive the upgrade
    rm -rf /comfyui/comfy /comfyui/comfy_extras /comfyui/comfy_api \
           /comfyui/comfy_api_nodes /comfyui/comfy_execution /comfyui/comfy_config \
           /comfyui/app /comfyui/api_server /comfyui/middleware /comfyui/utils \
           /comfyui/alembic_db /comfyui/blueprints; \
    cp -a /tmp/comfy-new/. /comfyui/; \
    rm -rf /tmp/comfy-new /tmp/comfy.tgz; \
    uv pip install --python /opt/venv/bin/python -r /comfyui/requirements.txt \
        "transformers<5" "huggingface-hub<1"; \
    cd /comfyui && timeout 300 /opt/venv/bin/python main.py --quick-test-for-ci --cpu; \
    grep -q "flux3" /comfyui/nodes.py

# --- 3. handler wrapper + model fetcher --------------------------------------
# Keep worker-comfyui's handler as a module and put our wrapper in front of it.
RUN mv /handler.py /handler_base.py
COPY handler.py flux3_rewrite.py fetch_models.py /
COPY start_flux3.sh /start_flux3.sh
COPY flux2_klein_base_t2i_api.json flux3_klein_base_t2i_api.json flux3_klein_distilled_t2i_api.json /workflows/
RUN chmod +x /start_flux3.sh \
    && /opt/venv/bin/python -c "import ast,sys;[ast.parse(open(f).read(),f) for f in ('/handler.py','/handler_base.py','/flux3_rewrite.py','/fetch_models.py')]" \
    && cd / && /opt/venv/bin/python -c "import flux3_rewrite, fetch_models"

ENV FLUX3_REWRITE=true \
    FLUX3_VARIANT=base \
    HF_HUB_DISABLE_TELEMETRY=1

CMD ["/start_flux3.sh"]
