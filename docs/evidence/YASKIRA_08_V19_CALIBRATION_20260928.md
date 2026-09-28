# Yaskira08 V19 — primer ciclo del master de calibración

Fuente: `Yaskira/08.mp4`, 149.533 s. Human Gold redondeado: conservar
120–147 s. Run `36410612325`, commit `12730096`, artefacto `10964875544`.
El comparador de solo lectura `scripts/compare_selection_to_human_gold.py`
lee el JSON de resultados y `YASKIRA_08_GOLD_EVAL.json`; no participa en
decisiones de producción.

| Medida | Resultado V19 |
|---|---:|
| Selección (unión de intervalos) | 122.720 s |
| Gold retenido | 24.110 / 27.000 s |
| Gold perdido | 2.890 s |
| Material extra retenido | 98.610 s |
| Recall/precision aproximados | 89.30% / 19.65% |
| Estado de entrega | NOT_DELIVERABLE_NEEDS_HUMAN_REVIEW |

Los intervalos de Gold son redondeados. El motor seleccionó desde 19.25 s
y mantuvo casi todo lo grabado antes del Gold. La última selección termina
en 144.11 s y descarta el CTA 145.1–146.709. El modelo etiquetó las tomas
previas como `independent_story_coverage`, la toma final como
`failed_delivery` y el CTA como `recording_process_bts`. La protección
`av_audience_unique_content_overrides_failed_label` recuperó la toma final
parcialmente por novedad de tokens, sin resolver la comparación de intentos.

El render de diagnóstico existe en el artefacto. El QC reportó
`LINGERING_ACCIDENTAL_SILENCE` de 1.397 s en la salida 93.391–94.788,
mapeado a fuente 117.128–118.525; el trimmer lo rechazó. Esa etiqueta
automática requiere confirmación al escuchar el MP4; el bloqueo técnico es
real. El MP4 tampoco ha recibido aprobación editorial humana.

Hipótesis siguiente: la información agrupada sigue sin hacer competir la
toma final completa con la unión de fragmentos anteriores, y la etiqueta
`failed_delivery` de un candidato con final recuperable se resuelve mediante
proxy léxico. Investigar el contenido audiovisual exacto y el CTA antes de
modificar autoridad. Separadamente, reproducir el rechazo físico en 117–119.
No lanzar la regresión pagada 01–10 hasta que una corrección general supere
el caso diagnosticado y pruebas contrarias offline; luego correr los diez
con un mismo commit/configuración.

### Experimento V20 preparado

El modelo recibió `current_bucket` y votos Hybrid locales en cada candidato
y volvió a escoger casi exactamente los candidatos ya marcados SELECT; la
toma final estaba marcada DISCARD. Esto muestra correlación, no demuestra
por sí solo causalidad de anclaje. V20 elimina esas decisiones previas del
payload de V2, incluidos sus resúmenes de grupos, mientras conserva tiempos,
texto, ASR y evidencia audiovisual. El runtime legado y el diagnóstico
conservan los buckets. Hipótesis falsable: sin esa señal circular, el modelo
elegirá la toma posterior y examinará el CTA como contenido de audiencia.
La instrucción sobre CTA es la misma de V19 para aislar el cambio de entrada.
Si mantiene la misma selección, el bloqueo está en comprensión de intentos
o cobertura, y no se repetirá este experimento sin una nueva causa.

El silencio 117.128–118.525 está en la toma anterior 103.07–120.45;
`ffmpeg silencedetect` confirmó en el MP4 el mismo intervalo de salida
93.391–94.788. Se tratará la selección primero: al descartar esa toma,
el silencio también abandona la línea de tiempo. No se relaja QC para
permitir un render bloqueado.
