# Video00 — Medium + WhisperX + Watch & Listen

Run 36100898847, SHA probado 456d0bc12d190d673b45c52a968266cd423b5370.
Una ejecución pagada autorizada. Fuente SHA verificada: b37059b1790cf3eb0447bb54595cc99f6668aadc5f65dafa9cdf6181621ef9f5.

## Resultado

WhisperX 3.8.6 ejecutado en L4: 619 palabras, 54 segmentos; estado passed.
Idioma detectado es. Sin interpolación ni proveedor alternativo. Una reutilización de la evidencia dentro del trabajo. Alineación 32.402 segundos; ASR más alineación 100.597 segundos.
Salida 150.367 segundos, 24 segmentos. Integridad SHA256 y decodificación completa del MP4 verificadas. Control técnico PASS, control perceptual FAIL: NOT_DELIVERABLE_WATCH_LISTEN_BLOCKED. El campo deliverable=true del resumen no representa aprobación editorial.

| Comprobación | Medium + Watch/Listen anterior | Medium + WhisperX + Watch/Listen |
|---|---:|---:|
| Criterios editoriales | 4/11 | 6/11 |
| Gold histórico | 15/18 | 16/18 |
| Duración | 160.289 s | 150.367 s |
| Segmentos seleccionados | 22 | 24 |

## Hallazgos

El texto seleccionado conserva «No quiero sonar a conspiración» y las instrucciones del cierre. Ya no mantiene la toma completa repetida de espinillas; sin embargo queda el fragmento «de personas con problemas hormonales», por lo que pasar ese criterio automático NO demuestra limpieza completa.
Reaparece «Tuve problemas de estómago, no» de un intento abandonado. Persisten la reformulación del porcentaje y dos apariciones de «cuídate».
La cobertura temporal de la toma de ginecóloga es 6.772 s, por debajo del umbral 7.5 s del benchmark; el texto completo está presente. Con nuevos tiempos de alineación, ese fallo no demuestra por sí solo pérdida audible: requiere revisión del audio.
El control perceptual bloquea un gesto de reinicio fuera de palabra, mapeado desde evidencia fuente cerca de 99.33 s del video. Es evidencia fuente mapeada, no un nuevo análisis visual del MP4.

BestTake V2 evaluó ocho casos y registró una dominancia; la autoridad de protección no cambió ganadores. Prosodia procesó tres candidatos en una familia, pero el árbitro declaró evidencia insuficiente. Contexto global preparó 12/29 regiones antes de clasificación; esto no prueba influencia causal de cada región.

## Veredicto

La integración Medium + WhisperX funciona técnicamente. El resultado editorial sigue sin aprobarse. El cambio de 4/11 a 6/11 describe estas dos ejecuciones; no permite atribuir causalmente toda diferencia a WhisperX, pues el clasificador se ejecutó otra vez y no hubo un replay controlado del mismo texto. No se ha realizado una revisión humana audiovisual completa. No se promovió a producción ni se inició otra ejecución.

Evidencia: uploaded-medium-whisperx-reports del run 36100898847; evaluaciones locales con los mismos manifests sin modificar. 50 tests locales pasaron antes del lanzamiento.
