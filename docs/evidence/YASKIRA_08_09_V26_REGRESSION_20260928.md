# V26 — calificación conjunta Yaskira 08/09

Commit `3faa14fc70de93ce29b0462a10fa24981d70cfa0`, run `36438018959`,
artefacto `10977600550`. El workflow y QC de ambos MP4 pasaron. El estado de
entrega de ambos sigue `DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN`.

| RAW | Gold aprobado | Aprobado conservado | Aprobado perdido | DELETE retenido | QC |
|---|---:|---:|---:|---:|---|
| 08 | 27 s | 25.199 s | 1.801 s | 0 s | PASS |
| 09 | 73 s | 35.647 s | 37.353 s | 7.722 s | PASS |

08 seleccionó 120.39–141.388, 142.709–145.1 y 145.1–146.91: replica el
resultado bueno anterior, aunque la historia V23/V25 prueba que una sola
repetición buena no certifica consistencia. El déficit de ~1.8 s está
mayoritariamente en una pausa medida en fuente, pendiente de revisión humana.

09 seleccionó 50.33–70.89, 77.07–86.53, 106.66–118.85 y 120.618–121.78.
Su Gold es 5–14, 19–28, 52–66, 73–91 y 97–120. El selector propuso SELECT
para acción muda 98.435–104.435 (confianza .95; visual focal: la creadora
agrega otro scoop al recipiente). El intervalo resulta de intersectar
observación visual 97.935–104.435 con silencio medido 98.435–106.764. La
comparación de tomas posterior la cambió a DISCARD, diciendo que la
explicación hablada 106.66–121.78 «resume la preparación». Una explicación
posterior no contiene los fotogramas de aquella acción; la guarda V27 debe
conservar una acción SELECT salvo cuando el ganador cubra físicamente su
intervalo en la misma fuente.

La misma comparación de tomas descartó las entregas 5.404–16.9 y
19.218–28.1 a favor de 50.33–70.89, aunque la primera contiene una condición
de audiencia («si estás usando gelpe») ausente de la ganadora y la segunda
un resultado medido en 30 días, también ausente. V27 prueba una guarda
general de condiciones y cantidades con unidad ante equivalencias falsas;
la guarda solo impide la eliminación por competición y no reescribe una
decisión DISCARD previa si ninguna protección la contradice.

Quedan dos defectos independientes: 73.8–77.1, parte inicial de una
afirmación que continúa en la selección 77.07–86.53, está descartada como
reintento; el modelo debe verificar la relación de continuación y la
ausencia de reinicio antes de rescatarla. Además, 120.618–121.78 retiene
«ya se acabó ese», fuera del Gold. No se ha aprobado ninguna regla de
palabras finales ni frase literal específica del video para encubrirlos.

Decisión: **EDITORIAL_FAIL** para 09; V26 no se promueve al lote 01–10.
