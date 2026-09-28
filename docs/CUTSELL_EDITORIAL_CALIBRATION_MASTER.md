# CutSell — master de calibración editorial

Estado: activo, 2026-09-28. Aplica al motor experimental V2 y a sus
sucesores. Este documento rige el trabajo editorial cuando llega un nuevo
video, una corrección del Human Gold o un error del motor. Referencias de
seguridad, despliegue y autoridad de producción siguen en `AGENTS.md` y
`CUTSELL_COMMERCIAL_ENGINEERING_OPERATING_MODEL.md`.

## Punto de control actual (2026-09-28)

V25 logró en una corrida de 08 una selección de 25.199/27 s Gold sin material
externo, pendiente de revisión humana del render. La repetición de 08 y 09
con esa misma versión, run `36435114060`, **falló**: 08 recuperó además un
fragmento 97.21–101.09 s ajeno al Gold y descartó una continuación hablada
142.93–144.77 s; 09 recibió HTTP 503 en Watch + Listen antes de Selection.
No hay afirmación de estabilidad. El RAW 09 tiene acción útil
durante un silencio medido 98.4–106.8 s que V25 no ofrecía como
candidato visual independiente. Los fotogramas 90–108 s están registrados en
`evidence/YASKIRA_08_V25_AND_09_VISUAL_GAP_20260928.md`.

La corrección en investigación exige observar límites de acción con audio y
video locales, presentar un intervalo mudo explícito al mismo selector antes
de Freeze, proteger su identidad y sus bordes durante Boundary, y reservar
solo esa porción visual seleccionada como silencio intencional en QC. Una
región AV amplia, un silencio ASR o una pausa entre dos frases no bastan por
sí solos para conservarla. El contrato de Freeze ya protege intervalos mudos
que se seleccionen explícitamente; V26 ya creó y calificó ese candidato en
una corrida real. El primer probe focal `36435404880` falló por respuesta del
proveedor en forma de lista en vez de objeto; el contrato JSON se corrigió
y `36435957367` observó acción visual 96–106 s. Su afirmación de voz en
96–106 contradice el silencio del audio fuente 98.4–106.8; solo se permite
usar la intersección visual/audio medida, no la afirmación de voz del modelo.
El renderer ya respeta el final de escenas visuales
explícitas y QC reserva solo su ventana tras verificar el Freeze (35 pruebas
focales aprobadas). Una prueba de QA detectó y permitió corregir Freeze vacío
y fusión de acciones contiguas. La regla tentativa basada en una palabra
«para/to» y la normalización léxica global fueron descartadas tras QA:
ambas habrían permitido errores fuera del caso. La continuación hablada
142.93–144.77 requiere una prueba semántica o audiovisual adicional antes de
reclasificarla. Ningún estado de este bloque
equivale a aceptación editorial o autorización de despliegue.

**V26 ya medido**: run `36438018959` en los RAW 08 y 09. El 08 volvió a
25.199/27 s Gold, 0 s DELETE, QC PASS. El 09 retuvo solo 35.647/73 s Gold,
añadió 7.722 s DELETE, QC PASS: `EDITORIAL_FAIL`. El análisis focal halló
la acción 98.435–104.435 s, el selector la eligió y una comparación
posterior la eliminó porque consideró equivalente una explicación de voz
posterior. Ver `evidence/YASKIRA_08_09_V26_REGRESSION_20260928.md`.
V27 protege esa acción y cantidades/condiciones ausentes en equivalencias;
no implica que las demás pérdidas de 09 estén corregidas. Es obligatorio
repetir 08 y 09 y revisar el MP4 antes de lanzar 01–10.

## Objetivo y autoridad

El producto debe editar un RAW nuevo sin recibir timestamps de la usuaria en
producción. Los Human Gold son etiquetas de evaluación y calibración, nunca
entradas del selector, prompts de producción o excepciones por video.
El criterio final es el MP4 renderizado y revisado escuchando y mirando;
un workflow exitoso, una duración similar o una métrica agregada no aprueban
una edición.

## Datos y versiones

Cada caso conserva identidad de fuente (clave + SHA), RAW, Human Gold con
intervalos y tolerancia de redondeo, decisiones KEEP/DELETE, render, plan,
proveedor/modelo, commit, costos, estado técnico y revisión humana. Si cambia
el Gold, se crea una revisión con explicación; no se reescribe el histórico.

El conjunto inicial es `Yaskira/01`–`10`, con 07=Video00; sus decisiones están
en `CUTSELL_YASKIRA_01_10_HUMAN_GOLD.md`. El Video00 requiere además alinear
el MP4 Human Gold a coordenadas RAW antes de puntuar sus microcortes. Los
intervalos redondeados de 01–10 producen métricas aproximadas: ninguna cifra
por sí sola es aceptación de borde exacto.

Los diez casos son el **conjunto de desarrollo y regresión**. La primera
certificación comercial requerirá RAW posteriores de creadores, idiomas,
encuadres y estructuras no usados para elegir reglas; mantener juntos todos
los fragmentos de un mismo RAW en un solo conjunto. Se podrá introducir una
colección nueva de forma escalonada: casos de diagnóstico, regresión y prueba
ciega. No afirmar generalización por resolver los mismos diez repetidamente.

## Ciclo obligatorio de una corrección

1. **Congelar evidencia.** Registrar versión, fuente, Gold, selección,
   descartes, MP4, gate físico y decisiones del proveedor. Marcar claramente
   `TECHNICAL_PASS`, `EDITORIAL_FAIL`, `BLOCKED` o `ACCEPTED`; nunca convertir
   éxito de CI en aprobación humana.
2. **Localizar una causa.** Señalar capa responsable: transcripción,
   reconstrucción de toma, comparación de reintentos, selección global,
   protección, Freeze, Boundary, render o entrega. Formular una hipótesis
   falsable y ejemplos contrarios antes de editar código.
3. **Corregir la regla general.** Prohibidos IDs, frases o timestamps de un
   RAW como atajos del runtime. Los cambios de autoridad editorial deben
   pasar por la auditoría de reglas `CUTSELL_V2_CANONICAL_RULE_AUDIT_20260928.md`.
4. **Probar offline.** Replay del error, casos opuestos (composite legítimo,
   información única, CTA, números/negaciones, silencios), contrato Freeze,
   pruebas focales y QA independiente. Sin nuevo cómputo pagado para una
   hipótesis que ya falló de la misma manera.
5. **Calificar el caso diagnosticado.** Una corrida de RAW con commit y costo
   registrado cuando la hipótesis offline esté lista. Comparar intervalos y
   MP4. Si falla, volver a paso 1 con evidencia nueva; no hay límite fijo de
   ciclos de investigación. Cada corrida pagada requiere una hipótesis nueva
   **o** un objetivo de diagnóstico, reproducibilidad o recuperación técnica
   documentado y un presupuesto autorizado. Nunca repetir ciegamente el
   mismo fallo sin nueva información esperada.
6. **Regresión de conjunto.** Cuando la corrección resuelve el caso, ejecutar
   **los diez con la misma versión y configuración**. Informar por video
   pérdida de KEEP, retención de DELETE, duplicaciones, cortes físicos,
   legibilidad, estado QC y resultado humano. Conservar resultados fallidos.
7. **Decisión de promoción.** Ningún P0/P1, ninguna pérdida de afirmación
   crítica, ningún render bloqueado ni degradación material silenciosa. Las
   tolerancias editoriales se fijan antes de comparar la versión candidata.
   Una mejora del promedio no tapa un video roto. La usuaria aprueba la
   calidad editorial cuando se requiere aceptación humana; QA certifica
   pruebas y la autoridad de release gobierna despliegues.

## Cómo cambia el trabajo al madurar

La dinámica **no cambia de principio**: diagnóstico individual y regresión
conjunta por versión. Cambia la frecuencia: con un motor más estable se
agrupan defectos por clase, se ejecutan replays y regresiones automáticas
en cada commit, se reserva el lote pagado para cambios de selección y se
valida periódicamente en RAW nuevos sin ajustar el motor a ellos. Una
incidencia nueva entra en el registro, se reproduce, recibe severidad y se
convierte en prueba de regresión tras corregirse. No se pide a la usuaria
que etiquete de nuevo escenas ya declaradas; se consulta solo por una
ambigüedad editorial real o un cambio de producto.

## Registro inicial de casos y prioridades

| Caso | Gold | Línea base conocida | Próxima acción |
|---|---|---|---|
| 01–03 | Registrado | Pérdida de KEEP 12.06, 4.80, 1.72 s | Regresión completa |
| 04–06 | Registrado | 05 cercano; 06 cercano; 04 pierde 4.66 s | Proteger tomas completas y bordes |
| 07 / Video00 | Render Gold y referencias RAW parciales | Cierre duplicado retenido | Alinear mapa completo y medir |
| 08 | Solo 120–147 | V19 retuvo 122.72 s, perdió CTA 145.1–146.709, QC físico bloqueó | Clasificación de intentos y silencio |
| 09 | Cinco bloques KEEP | Pérdida inicial 33.65 s; existen corridas posteriores | Regresión de composite y demo |
| 10 | Cuatro bloques KEEP | 18.47 s extra iniciales | Regresión de reintentos |

Las cifras iniciales son de `CUTSELL_YASKIRA_BASELINE_VS_GOLD_20260927.md`.
La versión V19 es run `36410612325`, SHA `12730096`, artefacto
`10964875544`; el workflow pasó, pero la edición quedó bloqueada por silencio
físico de 1.397 s y falla editorial. Su siguiente hipótesis debe explicar
por qué se clasificó la toma final como `failed_delivery`, las anteriores
como `independent_story_coverage` y el CTA final como BTS.

## Plantilla para cada ciclo

Registrar en `docs/evidence/`:

```
case_id / RAW key / RAW SHA / Human Gold revision
baseline commit + run + artifacts
observed failure (times, speech, visual/audio, QC)
responsible layer + falsifiable hypothesis + counterexamples
change commit + offline tests + independent QA
qualification commit + run + cost + selected/discarded intervals
per-video Gold errors + rendered Watch/Listen verdict
regression 01–10 on identical commit/config
decision: rejected / diagnostic continuation / accepted for next gate
```

## Pendientes inmediatos

- Reproducir offline las decisiones V19 de Video08, incluidos los tres
  fragmentos de toma final y su CTA; separar clasificación semántica de
  restauración conservadora basada en tokens.
- Localizar en los datos de silencio 117.128–118.525 por qué Boundary dejó
  silencio detectado por QC y bloqueó la entrega; confirmar perceptualmente
  ese intervalo en el MP4.

El antiguo límite de dos ciclos por clase del documento
`CUTSELL_YASKIRA_BASELINE_VS_GOLD_20260927.md` queda sustituido por la
autorización posterior de continuar la investigación sin límite fijo, con
trazabilidad y objetivo para cada corrida pagada.
- Instrumentar el paso de grupos provisionales a intentos completos, y
  comparar con verdaderas familias de reintentos y escenas independientes.
- Corregir y validar contra 05, 06, 09 y 10 antes de lanzar el lote 01–10.

## 2026-09-28: dependencia audiovisual entre fragmentos adyacentes

La calificación V26 de RAW09 descartó 73.782–77.070 («... en tu»)
como `retry_alternate`, mientras seleccionó 77.070–86.526 («primera
semana ...») como pieza independiente. El probe focal de fuente 71–92
(run 36441381173, artifact 10979480433) confirmó `linked=true`,
`restart_observed=false`, incertidumbre baja y continuidad de prosodia,
postura y gestos. El artefacto es evidencia de diagnóstico, nunca una
excepción incorporada al código.

En V28 se inspeccionan pares adyacentes contradictorios con una ventana
limitada del audio/video original. La dependencia se acepta únicamente si
la fuente, palabras alineadas, orden, ausencia de reinicio y observación
multimodal coinciden; el testigo de continuidad se evalúa antes de Freeze.
La comprobación consume a lo sumo una llamada adicional por fuente,
reservada en el mismo presupuesto. Un fallo o falta de presupuesto conserva
la decisión previa y queda sujeto a la evaluación humana. Regresiones
negativas: cambio de fuente, hueco entre fragmentos y ausencia de palabras.
Pendiente: ejecución real en los RAW 08/09, inspección de render y repetición
01–10 en configuración idéntica antes de considerar la versión lista.

La corrida V27 `36440885494` (SHA `35163926`, artifact `10978832574`)
confirmó QC PASS de ambos RAW: 08 retuvo 25.199/27 s Gold **pero añadió
22.760 s del intervalo DELETE 0–120 (regresión frente a V26)**; 09 retuvo
58.516/73 s Gold, con 14.484 s todavía ausentes. En 09 el probe sí
observó «adds powder from a spoon» durante 97.935–104.935, pero la
enumeración de verbos de la acción silenciosa no reconoció «adds powder».
Se agregó reconocimiento general de verbo de manipulación más producto,
con negativo para mera exhibición del envase. Se revalida en V28.

## V28/V29: resultados y reliability gate (2026-09-28)

V28 run 36443271393 (SHA 5e0ebbb6) falló en 08: el selector
recibió dos respuestas sin candidatos y rechazó el plan. En 09 completó
el render y QC PASS, pero intersectó solo 58.860/73 s Gold y conservó
11.480 s fuera de los tramos KEEP. Su Watch+Listen global etiquetó
68–113.5 como mixed por un fumble de tapa, ocultando la acción 97–104;
no nominó probe focal. También propuso «en tu» después de «primera
semana», por lo que el verificador estricto omitió el enlace.

V29 run 36444903026 (SHA c4181717) corrigió la regresión de 08:
seleccionó 120.39–141.388, 142.709–145.1 y 145.1–146.91, con
25.199/27 s Gold y cero segundos DELETE conservados, QC PASS. El 09
falló antes de Selection por respuesta AV temporal inválida 113–20 s
sobre una fuente de 123.81 s; el pipeline la rechazó sin inventar cortes.

V30 prepara una nominación focal desde regiones mixtas con objeto/producto
más silencio medido, y corrige inversión de dos fragmentos únicamente si
la fuente audiovisual confirma una frase continua. Fuentes >90 s usarán
ventanas locales de 45 s para evitar depender de una única descripción
larga. QA y corrida 08/09 pendientes; son hipótesis, no resultados.

## V30/V31: unión de voz y demostración muda (2026-09-28)

V30 run `36452891320` recuperó en 09 la operación visual observada de
98.435–104.435 s y seleccionó 67.51/73 s Gold, pero el render quedó
`NEEDS_HUMAN_REVIEW`: silencio de salida 54.785–60.997 s abarcó el
remanente mudo de una frase 96.65–99.59 s y la escena visual seleccionada.
La auditoría fuente muestra silencio medido 98.421–106.983 s. V30 no permite
concluir que el producto está listo: 09 pierde aún el antecedente 73–77 s y
conserva material fuera de Gold; 08 no llegó a selección porque el límite de
sesión híbrida siguió en 0.02 USD pese a fijarse en 0.05 USD la suboperación.

V31 SHA `e0b07f234fc9ffc3dcd01768d838a7622197d1d8` ajusta el límite de
sesión a 0.05 USD sin quitar topes y reconcilia el solapamiento voz/acción
antes de Freeze únicamente si una fuente idéntica, silencio físico medido,
última palabra completa y cobertura total por el visual seleccionado lo
permiten. El QC perceptual excluye solo los fotogramas mudos autorizados por
el Freeze, conservando los defectos de silencio del resto. QA independiente
detectó y permitió cerrar el contraejemplo donde un visual más corto hubiera
eliminado contenido posterior; 48 pruebas locales pasaron. Run V31
`36456010160` está en calificación 08/09; registrar a continuación selección,
QC técnico, Watch+Listen y errores Gold por separado. El rechazo de una
acción visual sin evidencia y la falta de garantía de cobertura nunca deben
convertirse en exenciones de QC.

El enlace entre 73.782–77.070 y 77.070–87.37 permanece **disputado**: una
observación focal 71–92 marcó continuidad y otra 67.09–89.09 marcó reinicio.
No imponer unión ni descarte por una respuesta contradictoria; inspeccionar
la misma unión física y la evidencia de palabras antes de decidir. El lote
01–10 sigue condicionado a 08/09 sin regresiones materiales, y los renders
siguen requiriendo evaluación mirando y escuchando antes de promocionar.

La calificación V31 run `36456010160` terminó **sin aprobar**: 224 tests
pasaron, pero 08 recibió dos respuestas de selección Gemini sin candidatos
y falló cerrado. En 09 el selector eligió 58.431/73 s KEEP, 5.962 s fuera de
Gold; descartó 19.218–28.14 como `failed_delivery` aunque el Gold conserva
la afirmación «más fuerte en 30 días». A diferencia de V30, sí conservó
73.35–77.07 y 77.07–87.37; por tanto la pérdida editorial de 09 varía entre
corridas y no se certifica como estable. El revisor perceptual dejó de
marcar la demo muda como defecto; QC físico aún bloqueó un silencio
43.636–49.804 que cruza un resto mudo hablado inferior a 1.2 s y la acción
visual congelada. V32 divide ese intervalo en la parte protegida y la parte
no protegida, conservando FAIL si esta última alcanza 1.2 s. QA independiente
pasó la regla y 116 pruebas locales la cubren; V32 registra además el motivo
`blockReason` de respuestas vacías sin imprimir texto del RAW. Sigue pendiente
calificar 08/09 y observar el conjunto completo sin usar Gold en el motor.

V32 run `36458204884` pasó pruebas y ambos renders dieron QC PASS, pero
la edición **sigue en EDITORIAL_FAIL**. 08 intersectó 25.199/27 s Gold y
añadió 19.914 s DELETE 62.503–82.417. 09 intersectó 57.367/73 s Gold,
añadió 6.102 s fuera de Gold y no generó ningún candidato para la operación
98.435–104.435, que V31 sí había observado. En 09 V32 conservó
19–28.14 gracias a `material_retry_claim_preserved`; V31 lo clasificó
`failed_delivery` y perdió la afirmación. Esto prueba variación entre
percepción, grupos de reintentos y selección global en el mismo RAW;
corregir solo QC no basta. El batch 01–10 de V32 se lanza para medir clases
de error por fuente con la misma configuración, sin promocionar ningún
render a calidad comercial. Investigar primero la pérdida intermitente de
demostración silenciosa y el rescate erróneo de 08 62–82 con auditoría AV,
decisiones y metraje. La prueba Gold nunca entra al prompt de producción.
