# QA de Watch + Listen: bloqueo técnico y brechas editoriales

Fecha: 25 septiembre 2026. Auditoría sin llamadas pagadas ni renders nuevos.

## Veredicto

La integración audiovisual todavía NO está calificada. Las dos últimas pruebas no llegaron a editar. El problema inmediato es un rechazo de respuesta en el adaptador audiovisual, seguido de un aborto deliberado del flujo. Existe además una brecha entre reconocer una preparación y producir un descarte verificable dentro del motor. Un parche de duración no demuestra que esa brecha esté resuelta.

## Evidencia y versiones

- Versión ejecutada: `5ff922a75963df9c62bd53bf44d98120bb3fc7e8`, motor audiovisual `842f4c1`.
- MOV: run `36110633614`, 123.924 s, `ok=false`.
- Video00: run `36110633620`, 240.261 s, `ok=false`.
- Ambos errores: `Required audiovisual Watch + Listen failed: ValueError: AV region outside source`.
- Corrección posterior, NO ejecutada en esos runs: `6dddf39fa98beafa15b7c37fa276b1bff6fba1fe`.
- Baseline editorial anterior, diferente de estas pruebas: runs `36106648051` y `36106648113`.

## 1. Qué sí ocurrió y qué no

El flujo pasó la preparación/verificación de fuente, transcripción y percepción local; envió el video con audio a Gemini y llegó al análisis de su respuesta JSON. En el código ejecutado, ese error solo aparece después de countTokens, reserva de presupuesto, generateContent, respuesta terminada en STOP, parseo JSON y comprobación numérica básica de una región.

La respuesta no superó la validación de región. `safe_whole_video_analyze` convirtió la excepción en `provider_error`; `flow_b` vio que el proveedor audiovisual requerido no estaba disponible y levantó RuntimeError antes de construir las tomas y llamar a `build_flow_b_draft`.

Por tanto NO se alcanzaron la selección editorial posterior, Best Take, sus protecciones, prosodia de finalistas, limpieza mixta nueva ni render. Las banderas habilitadas no acreditan ejecución de esas capas en estos runs. No existe MP4 final que evaluar. El workflow verde significa que completó el proceso de conservar artefactos (`continue-on-error`), no que el motor aprobara.

## 2. Qué significa realmente el error

El código ejecutado usaba un solo mensaje para esta condición combinada:

`not 0 <= start < end <= source.duration_sec OR not 0 <= confidence <= 1`

La reproducción sin red, cargando exactamente ese módulo desde el SHA ejecutado, produjo el MISMO error en cinco situaciones:

| Caso controlado | Código ejecutado | Código corregido |
|---|---|---|
| Región válida | Acepta | Acepta |
| Final dentro del pequeño padding del encoder | Rechaza | Interseca con el final original |
| Confianza 90 en vez de 0.90 | Rechaza con mensaje temporal genérico | Rechaza mostrando confianza y límites |
| Intervalo invertido | Mismo mensaje genérico | Rechaza con coordenadas |
| Inicio negativo | Mismo mensaje genérico | Rechaza con coordenadas |
| Final muy posterior al video | Mismo mensaje genérico | Rechaza con coordenadas |

La duración del archivo convertido usada en el prompt y la duración original usada por el validador estaban desalineadas: es un defecto reproducible, corregido con una intersección limitada al padding ya verificado de <=0.3s. Pero los runs NO guardaron los valores rechazados ni la respuesta original. No se puede identificar retrospectivamente cuál de estas condiciones disparó cada fallo. Mi explicación anterior exclusivamente temporal fue demasiado concluyente.

## 3. Por qué activar las capas no asegura una edición buena

### Contexto audiovisual no equivale a prueba de descarte

`whole_video_av` produce `audiovisual_evidence`, un resumen con observaciones. Ese campo se consume directamente en la preparación del contexto del clasificador y en diagnósticos. No se convierte automáticamente en eventos físicos locales ni en pruebas de descarte por palabra. No hay que contarlo como una segunda evidencia independiente si el mismo modelo vuelve a citar su propia observación.

`recording_process_trim` exige, además de la clasificación mixta y palabras alineadas, acuerdo entre ventanas, confianza >=0.97, identidad exacta y corroboración local de cada fragmento. Una reproducción con observación audiovisual positiva y clasificación mixta a 0.99, pero sin eventos locales, conserva la toma completa y produce cero pruebas. Es una protección vigente, no un fallo demostrado del umbral; falta demostrar cómo la evidencia legítima atraviesa esa protección en casos reales.

### Cobertura limitada del resumen

El análisis pide hasta 12 regiones significativas, no una decisión exhaustiva para cada toma. El contexto enviado al clasificador se limita a 1,800 caracteres y elimina regiones completas cuando no caben. En un fixture de 12 observaciones con textos al máximo permitido, sobrevivió una y se omitieron once. Esto demuestra un riesgo reproducible de pérdida de cobertura, NO que esa omisión concreta ocurriera en los dos runs: ambos abortaron antes de construir ese payload. La ordenación por solapamiento actual prioriza regiones locales, pero no garantiza que cada toma reciba toda la evidencia necesaria.

### Recortes parciales con alcance limitado

La nueva limpieza de tomas mixtas solo recorta prefijos y sufijos. No resuelve preparación situada en el interior ni toda la agrupación de repeticiones. WhisperX alinea palabras; no decide qué intento es válido. Necesitamos conservar una realización coherente y los datos únicos, sin ensamblar frases inventadas de varias tomas.

### Prosodia y protecciones no necesariamente cambian ganadores

En el baseline anterior, no en los dos runs abortados:
- Video00: prosodia evaluó tres candidatos en una familia; hubo evidencia insuficiente para el arbitraje final y cero dominancias prosódicas.
- MOV: prosodia informó `no_eligible_families`.
- Protecciones Watch + Listen: siete evaluaciones en Video00 y tres en MOV; cero cambios de ganador en ambos.

Eso no prueba que las protecciones sean erróneas. Prueba que decir «activas» no demuestra un efecto editorial. Hay que auditar elegibilidad de familias, evidencia disponible y decisión resultante.

## 4. El problema editorial ya existía antes del fallo audiovisual

En el baseline Video00, 70 decisiones fueron audience y la evaluación editorial cayó de 6/11 a 4/11, con repeticiones persistentes. En el MOV hubo 12 decisiones mixed y cinco recording_only, pero cero pruebas aceptadas; la selección provisional mantuvo preparación y el control de contenido perdido bloqueó el render con dos hallazgos.

No corresponde atribuir esos defectos antiguos al nuevo error del adaptador: son etapas y ejecuciones distintas. Tampoco corresponde afirmar que recibir audio y video corregirá por sí solo agrupación, selección, continuidad o preservación de información.

## 5. Prioridades de corrección y criterios de aceptación

1. **Respuesta audiovisual observable y validable.** Guardar valores rechazados, motivo específico, duración original/convertida, estado de proveedor, consumo y evidencia estructurada suficiente para reproducir el fallo sin otra llamada. La corrección actual añade coordenadas y uso, pero no un archivo completo de respuesta para replay. Aplicar contrato de salida estructurado; `responseMimeType` JSON por sí solo no impone todos los límites del contrato.
2. **Mapa de evidencia por toma.** Relacionar observación de fuente/tiempo con intento y palabras alineadas. Mostrar qué evidencia llegó, cuál se omitió, qué regla decidió y por qué. Conservar independencia de señales y controles de contenido válido.
3. **Cobertura y casos mixtos.** Demostrar preparación al principio, al final y en medio; repeticiones completas/parciales; tropiezos; humor intencional; hechos únicos y negaciones, en español e inglés. Resolver cuándo resegmentar dentro de una toma sin introducir un montaje artificial.
4. **Elegibilidad real de Best Take/prosodia.** Verificar que tomas del mismo intento compitan, con señales suficientes, y explicar por qué una capa decide no intervenir. No convertir «cero cambios» en fallo automático ni «enabled» en éxito.
5. **Calificación final.** Repetir fuentes solo con autorización de gasto; revisar decisiones y MP4, comprobar qué preparación/repetición desapareció y qué contenido válido se conservó. Exigir trazabilidad desde evidencia hasta corte. No usar tests unitarios ni workflow verde como prueba de buena edición.

No se modificó el motor en esta auditoría. No hubo otra ejecución pagada. Estado: integración no aprobada; root cause exacta de los dos rechazos no recuperable con los artefactos existentes; defecto de padding reproducido y parcheado, pendiente de calificación real.
