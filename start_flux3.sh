#!/usr/bin/env bash
# Fetch FLUX.3 Klein files to the network volume (first cold start only),
# then hand over to worker-comfyui's normal start script.
cd / && /opt/venv/bin/python -u /fetch_models.py || echo "flux3-fetch - fetch step failed, continuing"
exec /start.sh
