# CutSell — comparación tras corrección: Video00 y último MOV

Fecha: 25 septiembre 2026. Motor corregido 94df34a; SHA probado bf4c77ca6d92da205c049130b3dfca4ad4998ecd. Dos ejecuciones autorizadas, secuenciales, Medium + WhisperX + Watch & Listen. No cambios del motor entre ambas.

## Veredicto

La corrección NO queda validada. No hay mejora editorial consistente ni evidencia para promoverla a producción. Activación y 181 tests locales no predijeron éxito en estos videos.

| Resultado | Video00 anterior | Video00 nuevo | MOV anterior | MOV nuevo |
|---|---:|---:|---:|---:|
| Run | 36100898847 | 36106648051 | 36105019864 | 36106648113 |
| Palabras alineadas | 619 | 619 | 252 | 256 |
| Segmentos seleccionados | 24 | 25 | 5 | 5 |
| Duración MP4 | 150.367 s | 165.700 s | Sin MP4 | Sin MP4 |
| Criterios editoriales Video00 | 6/11 | 4/11 | No aplican | No aplican |
| Gold histórico Video00 | 16/18 | 16/18 | No aplica | No aplica |
| Entrega | Bloqueada por control perceptual | Pendiente revisión humana | Bloqueada antes de render | Bloqueada antes de render |

## Video00

WhisperX pasó: español, 619 palabras, 54 segmentos. Se generó MP4 con QC técnico PASS, hash verificado y decodificación completa sin errores. Aún requiere revisión humana audiovisual; no está editorialmente aprobado.
Siete de once criterios editoriales fallan. Regresa la toma repetida de espinillas; persisten la reformulación del porcentaje y el cierre repetido. El intento abandonado del estómago queda fuera en esta ejecución. La cobertura de ginecóloga sigue por debajo del umbral de tiempo del benchmark, lo que no demuestra por sí solo pérdida audible.
La transcripción del cierre cambió respecto al anterior: «hidrátate» pasó a «hídrate» y «aliméntate» a «Alimentate». El fallo textual del cierre no debe atribuirse sin más a un corte.
El clasificador devolvió 70 decisiones de rol audience. Cero pruebas de recording_only aceptadas. Una de ocho ventanas solicitadas quedó sin clasificación por el presupuesto; siete disponibles. Los límites de gasto no se aumentaron. El nuevo orden de análisis no garantiza cobertura completa dentro de ese límite.

## Último MOV

Fuente exacta verificada: SHA256 5ae3cffb9034cc9aebbe4f645c4f1f34538f9113f05990291215c86b7163b681, 68,730,266 bytes. WhisperX pasó: inglés, 256 palabras, 17 segmentos. El ASR anterior tenía 252 palabras en 18 segmentos, así que no es un replay controlado del mismo texto.
No se generó MP4: freeze_blocked_no_render y cero intentos de render. Dos bloqueos UNIQUE_FACT_LOST / REAL_CONTENT_LOSS, con omission_permit_denied_reason=missing_identity, sobre «too many people ready set these are the» y el intento mixto «already gonna mess it up…».
«how am I supposed to say» ya no figura en la selección, pero persisten «I just needed a pep talk», «you can get a flip it I'm», «of course I can't hit it that time…» y preparación unida a la toma final. Esto impide considerar buena la selección, aunque cambiaron algunos descartes.
Roles recibidos: 13 audience, 12 mixed, 5 recording_only. Cero pruebas aceptadas: los recording_only tienen confianza 0.90 (umbral 0.95) y algunos presentan roles diferentes entre ventanas. No hubo ventanas rechazadas por presupuesto en este MOV. La conexión nueva se ejecutó, pero no tuvo evidencia suficiente para ejercer la autoridad nueva.

## Qué queda demostrado y qué falta

La ruta Medium + WhisperX y las capas se ejecutan. La limpieza y la validación aún fallan para mezclas de discurso útil y preparación, además de clasificación variable entre ventanas. Rebajar el umbral o desactivar el validador solo para que este MOV renderice no probaría una solución general.
El siguiente trabajo debe reproducir estos casos guardados: clasificación de fragmentos mixtos, separación de sus partes con evidencia temporal y propagación de esa evidencia al control de contenido perdido. También debe resolver la cobertura de clasificación bajo el presupuesto vigente. Este turno no introduce otra corrección ni inicia más ejecuciones.
