# Yaskira/01 — Modal RAW con Medium + WhisperX (corrida 36587713782)

**Fecha:** 2026-09-29. **Rama / SHA probado:** `feat/editorial-engine-v2-whole-video-rebuild` @ `36ee48fa61032ea04a4ebf3fe482bcf58293e1f4`.
**Workflow:** `.github/workflows/cutsell-video00-modal-raw.yml` (`workflow_dispatch`), inputs `source_key=Yaskira/01.mp4`, `asr_provider=faster-whisper-medium-whisperx`.
**Corrida:** <https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36587713782> (job `109472885639`). Autorización de prueba pagada: concedida previamente por la propietaria; una sola corrida.

## Verificación previa al lanzamiento (offline, sin costo)

- `cutsell_worker/medium_whisperx_asr.py` implementa el proveedor `faster-whisper-medium-whisperx`; `universal_clean_cut_validation._validation_asr` lo despacha; `modal_video00_full_benchmark.py` añade el sidecar WhisperX 3.8.6 a la imagen cuando `CUTSELL_VALIDATION_ASR_PROVIDER` es ese proveedor (variable definida a nivel de job en el workflow).
- Los flags que fija el paso «Pin Yaskira01 Medium WhisperX and Gemini 3.5 audiovisual review» existen en `cutsell_worker` (`CUTSELL_EDITORIAL_ENGINE_V2`, `CUTSELL_V2_NATIVE_SELECTION_ENABLED`, `CUTSELL_HYBRID_LLM_ENABLED`, `CUTSELL_WATCH_LISTEN_AV_ENABLED`, `CUTSELL_WATCH_LISTEN_AV_MAX_EDIT_USD`, `CUTSELL_ASR_MODEL`).
- Tests locales: `tests/test_medium_whisperx_asr.py`, `tests/test_gpt_whisperx_asr.py`, `tests/test_modal_video00_full_benchmark.py` → 72 passed.
- Importación local de `modal_video00_full_benchmark.py` con el mismo entorno (proveedor Medium+WhisperX, payload Yaskira/01) → OK.
- No había ninguna corrida de Actions en curso; el grupo `concurrency` del workflow impide solapamiento.
- Precedentes: el mismo workflow ejecutó la variante GPT+WhisperX en Modal L4 (36045181502, 24-sep) y `cutsell-video00-medium-whisperx.yml` ejecutó Medium+WhisperX en L4 (36100898847, 25-sep). Esta combinación exacta (workflow Modal RAW + Medium+WhisperX + clave Yaskira) no se había ejecutado antes.

## Resultado observado

| Paso | Resultado | Duración |
| --- | --- | --- |
| D-228 preflight S3 (`head-object Yaskira/01.mp4`) | success | 5 s |
| Pin Yaskira01 Medium WhisperX + Gemini | success | <1 s |
| Run Modal full benchmark (L4) | success (el wrapper local no lanzó excepción) | **21 s** |
| Print Modal run summary | **failure** en <1 s | — |
| Download Video00 Modal artifacts | success («no result_uri») | — |
| Print full canonical diagnostics | success («No result JSON downloaded») | — |

Interpretación de la mecánica del workflow (no del error, que no se pudo leer):

- El paso «Print Modal run summary» falla en <1 s solo por dos ramas: `modal-video00-result.json` existe pero `exit_code != 0`, o el JSON tiene `ok=false`. Si el archivo hubiera faltado, habría intentado recuperar el marcador S3 durante ~60 s. Por tanto **el wrapper sí escribió un resultado y ese resultado reporta fallo**.
- Duración total de 21 s y artefacto `cutsell-video00-modal-run-log` de 972 bytes (frente a 11 249 bytes en la corrida GPU real 36045181502): no hubo ejecución del motor. Causas compatibles: excepción del wrapper local (`terminal_state=local_wrapper_exception`, p. ej. autenticación Modal o fallo de construcción de imagen) o excepción temprana dentro del contenedor (p. ej. error de importación de `cutsell_worker.serverless_handler` en la imagen Modal tras los cambios de la rama; la última corrida Modal exitosa de esta línea de código es del 25–27 de septiembre). **Ninguna de estas hipótesis está confirmada.**
- El artefacto `cutsell-video00-modal-human-review` (58.8 MB) contiene solo el RAW `Yaskira/01.mp4` descargado por el paso «Download Human Gold reference video» (rama sibling: sin Gold/Cut.ai). No contiene JSON de motor ni MP4 renderizado.
- Costo: el paso Modal duró 21 s; la probabilidad de un cargo GPU significativo es baja pero no se consultó facturación.

## Qué no se pudo comprobar y por qué (límite exacto)

- El texto del error está en las primeras líneas del log del job (paso 15/17). La herramienta de logs de GitHub disponible en esta sesión devuelve solo las últimas ~5 000 líneas y este job tiene más (los pasos posteriores vuelcan sus scripts), así que el tramo con el error queda fuera.
- La descarga completa del log (`/actions/jobs/{id}/logs`) y de los artefactos (`cutsell-video00-modal-run-log`, `.../human-review`) redirige a `*.blob.core.windows.net` (almacenamiento de GitHub Actions), y la política de red del entorno rechaza ese host (`CONNECT 403`). No se rodeó el bloqueo.
- El marcador durable `s3://<bucket>/cutsell/benchmark-results/video00-modal-36587713782-1/...` no es accesible: las credenciales AWS del entorno son inválidas (`InvalidAccessKeyId`). No hay token Modal local.

## Estado

- **RAW COMPLETE: NO.** No existe MP4 ni JSON de selección para Yaskira/01 en esta corrida. No hay nada que comparar contra las decisiones recuperadas de la propietaria (DELETE solo 0:58–0:59; KEEP 0:29–0:32, 1:07–1:08, 1:27–1:30).
- **Siguiente acción técnica:** leer las primeras ~300 líneas del log del job `109472885639` (o el artefacto `cutsell-video00-modal-run-log`) para obtener `error` / `error_type` / `terminal_state` del resultado; según eso, corregir en la capa responsable y relanzar **una** corrida. No relanzar a ciegas.
- Este documento no altera Gold, canon ni estado de puertas V3.

---

## Actualización — causa confirmada y segunda corrida (36591488345)

**Causa raíz de 36587713782 (leída del log completo, aportado por la propietaria):** `ValueError: AV budget and conservative multimodal prices must be explicitly configured`. `cutsell_worker/whole_video_av.py` (`GeminiWholeVideoAVProvider.__post_init__`) exige presupuesto por edición y los dos precios por millón de tokens finitos y positivos; el paso «Pin Yaskira01…» solo fijaba `CUTSELL_WATCH_LISTEN_AV_MAX_EDIT_USD`. Clasificación: `evidence` (configuración de la percepción AV), no una decisión editorial.

**Corrección (`fec5193`):** el paso fija `CUTSELL_WATCH_LISTEN_AV_INPUT_USD_PER_MILLION=0.30` y `..._OUTPUT_USD_PER_MILLION=2.50` (las tarifas conservadoras que ya usan `cutsell-editorial-v2-focused-01-preflight.yml`, `cutsell-editorial-v2-ten-raws.yml`, `cutsell-editorial-v2-batch-01-10.yml`, `cutsell-v2-video08-block-diagnostic.yml` y `benchmarks/run_uploaded_asr_comparison.py` para `gemini-3.5-flash-lite`) más `CUTSELL_WATCH_LISTEN_AV_TIMEOUT_RETRY_ENABLED=1`, y valida las tres variables en el runner CPU antes de cualquier despacho GPU. Test `tests/test_cutsell_yaskira01_modal_av_preflight.py` (7 casos) ejecuta el script real del paso y pasa el entorno resultante por `build_av_provider`; falla contra la versión anterior del paso.

**Segunda corrida:** <https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36591488345> (job `109485435971`, SHA `fec5193`, mismos inputs).

| Paso | Resultado | Duración |
| --- | --- | --- |
| Pin Yaskira01 (con validación AV) | success | <1 s |
| Run Modal full benchmark (L4) | success | 5 min 09 s |
| Print Modal run summary | **success** (`ok=true`) | <1 s |
| Download Video00 Modal artifacts | success | 1 s |
| Print full canonical diagnostics | success | 1 s |
| Pasos Video00-específicos (D-235G/J, arquitectura Video00, Gold QA 18-check, Pacing V2 D-218R/221/225) | failure (esperado en clave sibling; no leen Yaskira) | — |

Artefactos: `cutsell-video00-modal-run-log` 3 166 B (972 B en la corrida fallida); `cutsell-video00-modal-human-review` **106.8 MB** frente a 58.8 MB en la corrida fallida (esa contenía solo el RAW), es decir, ahora incluye el JSON del motor y un render; `cutsell-video00-modal-validator-reports` 36 KB.

**Lo que aún NO está inspeccionado:** el registro editorial (KEEP/DISCARD con tiempos y texto, auditoría ASR Medium+WhisperX, `live_render_qc`/`delivery_status`) se imprime en los pasos 17 y 21, fuera de las últimas 5 000 líneas que la herramienta de logs de esta sesión puede leer; la cola visible (pasos 46+) solo trae diagnósticos legacy (P1/P2, ordering) sin intervalos de selección. Por tanto: **RAW técnicamente completo; resultado editorial NO evaluado; WhisperX NO confirmado en video; nada se compara todavía con las decisiones de la propietaria.**

**Medida para que esto no se repita:** nuevo paso «Tail-safe editorial summary (sibling-safe; never fails)», último de los pasos de impresión, que reproyecta desde `artifact/video00-modal.json` la identidad de fuente, la auditoría ASR compacta, KEEP/DISCARD con `start/end/text` y el estado de QC/entrega, y lo guarda en `artifact/editorial-summary-tail.json`. No recalcula decisiones ni consulta Gold. Test `tests/test_cutsell_modal_raw_tail_safe_editorial_summary.py` (4 casos).
