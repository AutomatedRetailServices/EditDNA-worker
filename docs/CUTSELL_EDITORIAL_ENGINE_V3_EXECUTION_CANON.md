# CutSell — canon de ejecución del nuevo selector editorial

**Estado:** dirección de implementación solicitada por la Product Owner en la conversación del 2026-09-29. **Naturaleza:** plan; este documento no implementa V3 ni certifica resultados. **Alcance de calibración:** los diez RAW `Yaskira/01–10`, con `docs/CUTSELL_YASKIRA_01_10_HUMAN_GOLD.md` como referencia de evaluación. No se amplía todavía a 150 videos.

## 1. Resultado que debemos entregar

CutSell recibe un RAW y entrega automáticamente un primer corte terminado que elimina pausas inútiles, tropiezos y retomas inferiores, elige la mejor entrega de cada idea y conserva la historia, hechos, cifras, condiciones, negaciones, humor, personalidad, demostraciones y continuidad. El editor permite ajustes posteriores, pero **no sustituye la obligación del primer corte automático**. Ningún éxito de CI, FFmpeg o QC físico equivale por sí solo a calidad editorial.

Este es un objetivo de comportamiento observable comparable con lo que CutAI publica; no hay evidencia pública que autorice afirmar que CutAI use nuestra arquitectura interna.

## 2. Autoridad y límites

1. Respetar `AGENTS.md`, `docs/CUTSELL_BRAIN_DOCTRINE.md`, `docs/CUTSELL_EDITORIAL_RESOLUTION_AND_HUMAN_ESCALATION_CONTRACT.md` y Selection Freeze. Ante contradicciones materiales entre intentos, seguir el bloqueo y la jerarquía del contrato existente. Antes de juzgar calidad de entrega, proteger cobertura semántica crítica. La revisión humana es excepcional, no una vía normal para resolver incertidumbre del motor.
2. Gold sirve para evaluar y diagnosticar fuera del motor. Nunca entra al prompt de producción ni se codifican nombres, frases, tiempos o IDs de un RAW como lógica de selección.
3. No fabricar habla ni unir retomas incompatibles para simular una frase que nunca se dijo. Toda palabra y fotograma elegido debe tener procedencia temporal de fuente.
4. Mantener V2 disponible como referencia y reversión durante la implementación. No agregar más reparaciones editoriales por video a V2. Eliminar sus módulos de autoridad y reparaciones solo después de que V3 pase las puertas definidas abajo; no borrar RAW, Gold, evidencia, historial ni funciones de importación, ASR, AV, editor, Boundary y render que sigan vigentes.
5. No crear infraestructura pagada, aumentar gasto recurrente, desplegar, fusionar PR ni actuar sobre producción por este canon. Las pruebas offline con evidencia guardada anteceden cualquier nueva corrida pagada. El alcance de autorizaciones concretas se toma de la sesión vigente y de `AGENTS.md`; este documento no otorga una aprobación general nueva.

## 3. Decisión arquitectónica

V3 reemplaza la **autoridad editorial** `SELECT/SWAP/DISCARD` sobre candidatos provisionales. Las piezas que V2 genera pueden conservarse como evidencia, pero sus límites y buckets no obligan a V3. V3 trabaja con una línea temporal de origen, reconstruye intentos completos, compara entregas de la misma idea y emite un plan de tramos de fuente verificables. Boundary y FFmpeg ejecutan ese plan tras Selection Freeze; no corrigen pertenencia semántica.

Contratos versionados mínimos:

| Contrato | Contenido y garantía |
| --- | --- |
| `SourceEvidence` | Hash del RAW, versiones de ASR/AV/modelos, palabras con IDs y tiempos completos, pausas y observaciones con procedencia. Una repetición offline recibe exactamente el mismo objeto. |
| `AttemptMap` | Intentos y continuaciones con tramos de palabras/imagen, idea propuesta, señales de reinicio y huecos. Permite varias hipótesis y cruzar límites de candidatos V2. No decide KEEP/DELETE. |
| `EditorialDecision` | Idea, alternativas comparadas, cobertura de afirmaciones, contradicciones, elección, descartes, partes complementarias, evidencia y estado de resolución. No requiere seleccionar un candidato entero. |
| `CanonicalEditPlan` | Tramos ordenados de fuente, palabras/acciones incluidas, alternativas y motivos de exclusión. Valida procedencia, cobertura, solapamientos, uniones y correspondencia con el MP4 antes y después de Freeze. |

La implementación puede ubicar módulos nuevos junto a `cutsell_worker/editorial_engine_v2.py`, `attempt_reconstruction.py` y `unified_selection_google.py`, pero debe mantener contratos separados de percepción, reconstrucción, decisión, Boundary y render. Elegir nombres de archivos tras inventariar interfaces existentes; no duplicar servicios por comodidad.

## 4. Secuencia obligatoria de ejecución

### Puerta 0 — Inventario reproducible

**Hacer:** fijar revisión V2 de referencia y reunir para cada 01–10 identidad del RAW, Gold, ASR por palabra, AV, candidatos, decisiones, intervalos finales y artefactos existentes. Registrar ausencias y diferencias de versiones. Alinear mecánicamente el Human Gold render de 07 al RAW antes de usar una comparación completa; sus ejemplos parciales no son un mapa microtemporal completo. Reproducir offline las decisiones solo donde se disponga de todas las entradas necesarias; un ASR guardado por sí solo no prueba replay completo.

**Salida:** manifiesto de diez fuentes y tabla de errores con etapa de origen (`evidence`, `attempt_map`, `decision`, `boundary`, `render`) y enlace a evidencia. **Bloqueo:** una fuente sin datos necesarios queda `EVIDENCE_INCOMPLETE`; no se inventa el dato, no se declara reproducibilidad y no se lanza automáticamente un RAW para rellenarlo.

### Puerta 1 — Representación de intentos

**Hacer:** definir contratos y construir `AttemptMap` desde el flujo de palabras, pausas y evidencia audiovisual. Proponer límites, reinicios, continuaciones y alternativas sin forzar candidatos V2 como unidades completas. Mantener acciones visuales significativas sin habla. Una hipótesis alternativa permanece disponible hasta comparar contenido y performance.

**Pruebas:** frase repartida entre candidatos; comienzo fallido y entrega completa; ideas contiguas diferentes; retoma con un hecho nuevo; contradicción de cifra/negación; silencio intencional; acción visual útil. Controles negativos impiden agrupar por tema amplio, unir habla fabricada o borrar una historia KEEP-all.

**Salida:** para cada Gold, una consulta de *representabilidad* demuestra que la entrega aprobada se puede expresar en el contrato sin insertar el Gold en producción. Esto prueba capacidad de representación, **no** que V3 la elija. Si falla, corregir evidencia o mapa antes de trabajar en la comparación.

### Puerta 2 — Comparación editorial

**Hacer:** comparar intentos de la misma idea en orden de autoridad existente: contradicciones; cobertura de ideas; completo frente a abandonado; contenido crítico y equivalencia; error humano de performance; ajuste audiovisual, historia y personalidad. Pedir a un modelo evidencia acotada y estructurada cuando el juicio requiera comprensión; validar que todo índice y tramo devuelto exista. La decisión debe explicar por qué cada parte del corte conserva información útil o descarta una repetición.

**Salida:** `EditorialDecision` trazable por idea, sin etiquetas autónomas `independent`/`redundant` que por sí solas autoricen incluir o borrar grandes tramos. Resolver automáticamente con evidencia suficiente; aplicar el default doctrinal y la escalación excepcional exactamente según el contrato canónico. Fallo del proveedor o respuesta incoherente es observable y no se convierte en un falso PASS.

### Puerta 3 — Plan de corte y compatibilidad

**Hacer:** construir `CanonicalEditPlan` usando IDs de palabra/tramos fuente. Validar límites, orden, procedencia y afirmaciones protegidas; prohibir duplicaciones, cortes a mitad de una palabra y composiciones incompatibles. Adaptar el plan validado a la interfaz actual de Selection Freeze, Boundary, render y editor detrás de una activación V3 aislada. V2 sigue siendo el control, no el fallback silencioso de una decisión V3 fallida.

**Salida:** plan congelado y render que no cambian selección ni orden. Pruebas de integración verifican que los tramos de origen del MP4 coinciden con el plan; Boundary solo pule tiempos físicos.

### Puerta 4 — Calibración offline de diez RAW

**Hacer:** ejecutar la misma revisión V3 contra evidencia versionada. Medir por fuente `KEEP` perdido, `DELETE` retenido, toma incorrecta, orden, hechos críticos y continuidad. Comparar con V2 y con la revisión V3 anterior; incluir 01–03 KEEP-all, 06 como control, 08–10 como errores conocidos y 07 alineado. Las pruebas de control negativo y de protección de personalidad no se omiten por mejorar promedios.

**Salida:** matriz 01–10 de antes/después, errores por etapa y artefactos de replay. Una reparación local que causa pérdida material en otra fuente se rechaza. No promediar un fallo grave hasta esconderlo. Si la fuente cambió de evidencia, se registra como nueva entrada, no como mejora de selección.

### Puerta 5 — RAW real, inspección y generalización

**Hacer:** congelar una revisión que pasó la puerta 4, ejecutar la prueba de RAW dentro de autorización y control de gastos vigentes, comparar intervalos, mirar y escuchar los MP4, y registrar la diferencia entre QC técnico y aceptación editorial. Probar después videos no vistos sin calibrar con su Gold. No limitar arbitrariamente 08 a cero pruebas ni imponer dos pruebas máximas; toda nueva corrida debe responder una hipótesis y guardar sus entradas y salidas.

**Salida:** primer corte automático juzgado contra la referencia por video y revisión audiovisual. Pasar los diez no demuestra por sí solo capacidad comercial: se requiere prueba ciega y análisis de fallos. Solo entonces se considera promoción o retirada de V2.

### Puerta 6 — Retiro controlado

**Hacer:** identificar el código V2 obsoleto mediante referencias y pruebas, retirar su autoridad y reparaciones una vez que V3 sea apto, conservar historial y procedimiento de reversión, actualizar documentación canónica y pruebas de integración. No borrar evidencia ni contratos todavía usados por app/editor. Eliminar en cambios pequeños verificables, nunca antes de la puerta 5.

## 5. Disciplina contra el loop

- Una diferencia del Gold se clasifica primero por etapa; se reproduce con entradas fijadas; se explica una causa general y su contraejemplo; se implementa en la etapa responsable; se evalúa el conjunto. No abrir un parche por `Video08`, `Video09`, frase, clip ID o timestamp.
- No lanzar una corrida pagada después de cada cambio. Agrupar cambios coherentes, pasar pruebas offline y congelar una revisión antes de procesar RAW. Registrar costo, modelo, respuesta y estatus, sin reintentar un prompt bloqueado como si fuera un error transitorio.
- La Product Owner ya declaró el Gold de 01–10. No pedirle que repita los KEEP/DELETE. Solo la aceptación editorial del artefacto final o una ambigüedad genuina según doctrina necesita su criterio.
- El estado de cada puerta es `NOT_STARTED`, `IN_PROGRESS`, `PASS`, `BLOCKED` o `FAIL`, con evidencia identificable. Ninguna puerta se marca `PASS` por número de tests, workflow `success` o MP4 renderizado solamente.

## 6. Registro de progreso inicial

| Puerta | Estado al crear este canon | Evidencia pendiente |
| --- | --- | --- |
| 0 Inventario | `NOT_STARTED` | Inventario completo 01–10 y alineación total de 07 |
| 1 Intentos | `NOT_STARTED` | Contrato V3 y pruebas de representabilidad |
| 2 Comparador | `NOT_STARTED` | Decisiones trazables por idea |
| 3 Plan | `NOT_STARTED` | Adaptador a Freeze/Boundary/render |
| 4 Offline | `NOT_STARTED` | Matriz 01–10 misma revisión |
| 5 RAW y prueba ciega | `NOT_STARTED` | MP4 inspeccionados y videos no vistos |
| 6 Retiro V2 | `NOT_STARTED` | Evidencia de sustitución y reversión |

**Próxima acción al iniciar implementación:** puerta 0. Este documento define dirección y criterio de avance; no afirma que V3 exista ni fija una fecha sin inventario verificable.
