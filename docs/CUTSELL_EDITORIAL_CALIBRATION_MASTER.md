# CutSell — master de calibración editorial

Estado: activo, 2026-09-28. Aplica al motor experimental V2 y a sus
sucesores. Este documento rige el trabajo editorial cuando llega un nuevo
video, una corrección del Human Gold o un error del motor. Referencias de
seguridad, despliegue y autoridad de producción siguen en `AGENTS.md` y
`CUTSELL_COMMERCIAL_ENGINEERING_OPERATING_MODEL.md`.

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
