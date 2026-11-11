# FLUX.3 Image Klein — worker de prueba A/B (RunPod Serverless + ComfyUI)

Gemelo del endpoint de producción **COMFYUI WORKFLOW V2** (`afn0htiebzt6i2`, FLUX.2 Klein),
con el mismo contrato de entrada/salida, pero corriendo **FLUX.3 Image Klein**.
Sirve para comparar calidad, tiempo y costo con las mismas solicitudes.

## Qué usa hoy producción (medido en los logs el 10–11 nov 2026)

| | Producción (FLUX.2) |
|---|---|
| Modelo | `flux-2-klein-base-4b.safetensors` (BF16, 7.2 GB) |
| Text encoder | `qwen_3_4b.safetensors` (tipo `flux2`) |
| VAE | `flux2-vae.safetensors` |
| Sampler | KSampler, 30 pasos, CFG 3.5, euler / simple |
| Resolución | 1920×1088 |
| Tiempo medio | ~46 s por imagen (707 jobs en 7 días) |
| Modelos en | network volume `VIDEOS YT` (`/runpod-volume/models/unet|clip|vae`) |

PuLID **no** se usa en producción (ni el nodo ni sus pesos están instalados), así que el cambio a FLUX.3 no pierde ninguna función.

## Qué hace este worker

1. **ComfyUI v0.45.2** sobre `runpod/worker-comfyui:5.11.0-base` (la base trae ComfyUI 0.34, que no soporta FLUX.3; el soporte llegó en 0.41).
2. **Descarga automática de modelos al volumen** en el primer arranque (solo una vez; queda en el mismo volumen que FLUX.2):
   - `unet/flux3-image-klein-base-bf16.safetensors` — FLUX.3 Klein base (equivalente directo a tu modelo actual)
   - `unet/flux3-image-klein-bf16.safetensors` — FLUX.3 Klein destilado (pocos pasos, más barato)
   - `clip/qwen3vl_4b_int8_convrot.safetensors` — text encoder Qwen3-VL 4B
   - `vae/flux2-vae.safetensors` — ya existe en el volumen; se reutiliza
3. **Modo drop-in**: si recibe un workflow de FLUX.2, lo traduce solo a FLUX.3:
   - `UNETLoader` con un modelo `flux-2*` → modelo FLUX.3
   - `CLIPLoader` tipo `flux2` → `qwen3vl_4b_int8_convrot`, tipo `flux3`
   - con `variant: "distilled"` además: `steps=16`, `cfg=1.0`
   - todo lo demás (prompt, seed, resolución, `ReferenceLatent`, VAE, SaveImage) queda igual
   - workflows que ya son FLUX.3, o de otros modelos (Wan, etc.), no se tocan
4. La respuesta trae un bloque extra `flux3` con los cambios aplicados.

## API (igual que producción)

```json
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run
{
  "input": {
    "workflow": { ...el mismo workflow API de FLUX.2 que ya mandas... },
    "images": [ ...opcional, igual que hoy... ],
    "flux3": { "variant": "base" }
  }
}
```

`flux3` es opcional:

| campo | default | uso |
|---|---|---|
| `variant` | `base` (o env `FLUX3_VARIANT`) | `base` = comparación justa con tu FLUX.2 base; `distilled` = versión rápida |
| `steps`, `cfg` | sin cambio (base) / 16 y 1.0 (distilled) | forzar valores |
| `sampler_name`, `scheduler` | sin cambio | forzar valores |
| `unet`, `clip` | ver arriba | usar otro archivo |
| `enabled` | `true` | `false` = mandar el workflow tal cual |

También puedes mandar directamente los workflows FLUX.3 `*_api.json` del repo.

## Variables de entorno del endpoint

| variable | valor |
|---|---|
| `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` (secret con tu token de lectura de Hugging Face) |
| `NETWORK_VOLUME_DEBUG` | `true` (igual que producción) |
| `FLUX3_VARIANT` | `base` (opcional) |

El repo de Hugging Face es **restringido**: antes hay que aceptar la licencia FLUX Kommunity en
https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy con la misma cuenta del token.

> Licencia: uso comercial permitido para organizaciones pequeñas, pero exige revisar/filtrar los resultados
> y etiquetarlos como contenido generado por IA donde la ley lo pida. Organizaciones grandes necesitan licencia de BFL.

## Configuración del endpoint (copiada de producción)

- Deploy from GitHub, Dockerfile en la raíz
- GPUs: 24 GB Pro + 32 GB Pro, 1 GPU por worker
- Max workers 4, active workers 0, idle timeout 20 s
- Escalado: queue delay, 4 s
- Timeout de ejecución: 4000 s
- Network volume: `VIDEOS YT` (EU-RO-1)
- Disco del contenedor: 20 GB (producción usa 5 GB; la imagen nueva trae ComfyUI actualizado)

**Primer arranque:** descarga ~27 GB al volumen (unos 5–15 min). Lanza un job de calentamiento antes de medir.

## Prueba A/B para tu agente

```bash
export RUNPOD_API_KEY=...   # nunca en el código
python ab_test.py \
  --flux2-endpoint afn0htiebzt6i2 \
  --flux3-endpoint <ID_DEL_ENDPOINT_NUEVO> \
  --workflow <el workflow FLUX.2 que tu agente ya usa, en formato API> \
  --prompts prompts_example.txt \
  --variants base,distilled \
  --out ab_results
```

El script manda **el mismo workflow, prompt y seed** a los dos endpoints. Genera:

- `results.csv` — estado, segundos facturados, espera y costo por imagen
- `summary.json` — medias y costo por 1000 imágenes
- `report.html` — galería lado a lado

### Protocolo recomendado

1. **Calentar:** 1 job a cada endpoint y descartarlo (incluye la carga del modelo y, en FLUX.3, la descarga inicial).
2. **Muestra:** 20–30 prompts reales de tus escenas de YouTube, no prompts genéricos.
3. **Comparar:**
   - calidad visual en `report.html` (anatomía, manos, texto, coherencia entre escenas)
   - `exec_median_s` y `cost_per_1000_usd`
   - fallos
4. **Decidir:**
   - Migrar si FLUX.3 base gana en calidad y no cuesta más.
   - O probar `distilled` si la calidad es comparable y sale más barato.

Si alguna escena usa imágenes de referencia (`ReferenceLatent`), esas también funcionan con FLUX.3. Inclúyelas en la muestra con `--images`.

## Estructura

```
Dockerfile                           imagen del worker
handler.py                           wrapper del handler de worker-comfyui (rewrite + info flux3)
flux3_rewrite.py                     traducción FLUX.2 -> FLUX.3
fetch_models.py                      descarga de modelos al volumen (con lock entre workers)
start_flux3.sh                       descarga modelos y arranca el worker normal
flux2_klein_base_t2i_api.json        workflow API de referencia (equivalente a producción)
flux3_klein_base_t2i_api.json        workflow API FLUX.3 base
flux3_klein_distilled_t2i_api.json   workflow API FLUX.3 destilado
ab_test.py, prompts_example.txt      prueba A/B (solo librería estándar)
test_rewrite.py                      tests unitarios (python -m unittest test_rewrite.py)
```

## Validación hecha antes de entregar

- 12/12 tests unitarios.
- Contra un **ComfyUI v0.45.2 real** con la misma estructura de volumen (`models/unet`, `models/clip`):
  - los 5 workflows (FLUX.2 de referencia, FLUX.3 base, FLUX.3 destilado y los dos reescritos desde FLUX.2) pasan la validación de ComfyUI y el chequeo de modelos de worker-comfyui 5.11.0
  - el handler completo reescribe, encola y devuelve el bloque `flux3`
- El paso del Dockerfile que actualiza ComfyUI conserva modelos, nodos custom y la config del volumen.
- **No probado aquí** (requiere GPU y los pesos reales): la inferencia en sí. Se valida con el job de calentamiento en RunPod.
