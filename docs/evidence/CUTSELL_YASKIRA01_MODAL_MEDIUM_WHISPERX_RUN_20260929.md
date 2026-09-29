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
