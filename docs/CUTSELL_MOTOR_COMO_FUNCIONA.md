# CutSell — cómo funciona hoy el motor de corte (1.2.1)

Estado al 17 de octubre de 2026. En vivo en el worker de Modal desde el commit `0ef2c50`
(rama `feat/simple-engine-v1`). Este documento explica qué hace el motor paso a paso, cuánto cuesta,
cómo se mide y dónde se podría usar una IA más barata. No contiene claves ni datos de clientes.

---

## 1. Qué hace, en una frase

Recibe la grabación cruda de una creadora de TikTok Shop (con errores, repeticiones, intentos
abandonados y comentarios fuera de cámara) y devuelve un primer corte limpio: qué partes se ven,
qué partes se ocultan y por qué, más los subtítulos palabra por palabra y los momentos para el zoom.
El corte es **no destructivo**: nada se borra; lo oculto se puede volver a mostrar en el editor de la app.

## 2. Flujo completo

```
App iPhone ──(sube el video a R2, por partes)──> API en Render ──(cola Redis)──> Worker en Modal (CPU)
                                                                                    │
     1. ffmpeg: audio mono 16 kHz                                                   │
     2. Deepgram Nova-3: palabras con tiempos                                       │
     3. Claude: paso de "pasadas"                (1 llamada)                         │
     4. Claude: paso de "decisión"               (1 llamada, a veces 2)              │
     5. Claude: revisión de hauls                (1 llamada; 3 si propone algo)      │
     6. Claude: revisión de venta (nuevo 1.2.1)  (1 llamada; 3 si propone algo)      │
     7. Código: tiempos finos (silencios, cortes de cámara, pausa natural)           │
     8. Claude: corrige palabras mal oídas + marca momentos para zoom (1 llamada)    │
     9. Guarda el "draft" (Redis) ──> la app lo muestra en el editor                 ┘
```

### Paso 1–2. Transcripción (Deepgram Nova-3)
- `cutsell_worker/simple_engine/asr.py`
- Audio mono 16 kHz. Puntuación activada, muletillas activadas ("eh", "um"), sin formato inteligente.
- Idioma: el que manda la app (es/en) o detección automática.
- "Keyterms" para que escriba bien marcas comunes: Shein, TikTok Shop, Temu, Amazon, leggings, jumpsuit…
- Resultado: lista de palabras `{palabra, inicio, fin}`.

### Paso 3. Pasadas (Claude) — `prompts/pass_v1.txt`
- La transcripción se parte en **líneas** (por punto final o pausa ≥ 0,5 s) y se numera L1, L2…
- Pregunta: ¿la creadora dijo el guion completo varias veces seguidas (varias "pasadas")?
  Si sí, ¿cuál es la mejor pasada completa (la "columna vertebral")?
- Respuesta JSON: `{"passes":[[La,Lb,completa?],...],"backbone":[La,Lb]}`.

### Paso 4. Decisión (Claude) — `prompts/decision_v1.txt`
El paso central. Reglas editoriales (calibradas contra el Gold de la dueña del producto):
- **Pasadas:** si hay varias, se queda la mejor pasada completa y se oculta el resto ("otra pasada"),
  salvo una frase entera con información que la pasada elegida no tiene.
- **Retomas:** cuando repite la misma frase justo después de equivocarse, gana la versión más completa
  (luego la más limpia, luego la más fluida; empate → la más tardía).
- **Frases abandonadas, ruido, comentarios fuera de cámara** ("otra vez", "¿quedó bien?") → se ocultan.
- **Tropiezos dentro de una frase** (palabra repetida, falso arranque de 1–4 palabras, "eh") → se recortan.
- **Secciones reiniciadas** (presenta un producto, lo explica, y vuelve a presentarlo) → se queda la mejor.
- **Ante la duda, se conserva:** ocultar contenido real es peor que dejar una repetición.
- Respuesta JSON: `{"hide":[[La,Lb,"código",Lw]], "trim":[[L,"palabras exactas"]]}`
  con códigos `r` retoma, `a` abandonada, `n` ruido, `p` otra pasada.
- Red de seguridad: si la respuesta oculta casi todo, se pregunta otra vez y gana la respuesta menos destructiva.

### Paso 5. Revisión de hauls (Claude) — `prompts/review_v1.txt`
- Solo para hauls / try-ons (varios productos distintos). Para todo lo demás responde "other" y no toca nada (1 llamada).
- En un haul arregla: el mismo artículo anunciado dos veces, o tomas mezcladas de una misma parte.
- Un cambio se aplica solo si **2 de 3** lecturas independientes coinciden.

### Paso 6. Revisión de venta (Claude) — `prompts/sale_review_v1.txt` — **nuevo en 1.2.1**
- Un "editor senior" lee el corte de los pasos 3–5 (cada línea marcada [KEEP]/[CUT]).
- Primero decide si es **venta** o no (historia, opinión, vlog, haul). Si no es venta, no toca nada.
- En una venta solo puede arreglar dos problemas (activados porque se midieron y nunca empeoraron un video):
  - **"restart" — venta reiniciada:** la creadora abandona la venta ("ay, qué payasa, vamos otra vez") y la
    empieza de nuevo; el corte había dejado parte del intento abandonado + el intento completo. Se oculta el
    intento abandonado, excepto el gancho y lo que solo dice ahí.
  - **"mixed" — frase mezclada:** media frase de un intento abandonado pegada a otro intento.
- Medidos pero **apagados**: "repeat" (la misma idea dicha dos veces; empeoró 2 videos del Gold) y
  "missing" (falta gancho/precio/llamada a comprar; nunca se activó).
- Protecciones:
  - Si la primera lectura no propone nada → 1 sola llamada y fin.
  - Si propone algo → 2 lecturas más; se aplica solo lo que proponen **al menos 2 de 3**.
  - Nunca puede desaparecer un número o precio del corte.
  - Nunca quita más del 25 % de las palabras que se ven (40 % si es una venta reiniciada).
  - Una palabra oculta nunca queda dentro de una parte visible (ajuste de bordes).
- Interruptor: `CUTSELL_SIMPLE_ENGINE_MODE=rules` en el secreto de Modal vuelve a la 1.2 sin redesplegar.

### Paso 7. Tiempos finos (solo código, sin IA)
- `splits_from_labels`: convierte las decisiones por palabra en tramos visibles/ocultos; las pausas
  largas dentro de una frase se acortan.
- `orphans`: un pedazo visible de 0–1 palabras y menos de 2 s se oculta ("pedazo suelto").
- `refine`: ajusta cada borde al silencio real midiendo la energía del audio (para no cortar sílabas).
- `keep_spoken_words`: un tramo oculto como "silencio" nunca puede contener una palabra dicha.
- `snap`: acerca los bordes a los cambios de toma de la cámara (ffmpeg scene detection).
- `breathe`: deja una pausa natural corta antes y después de cada frase, solo sobre silencio.

### Paso 8. Palabras y zoom (Claude) — `prompts/fixwords_v2.txt`
- Corrige solo palabras claramente mal oídas en lo visible (para los subtítulos; no cambia el corte).
- En la misma llamada marca hasta 6 momentos clave para Auto Zoom AI: gancho, producto, precio, llamada a comprar.

### Paso 9. Resultado (`cutsell_worker/simple_engine_adapter.py`)
- `selected`: tramos visibles (clips). `alternates`: tramos ocultos que tienen voz (se pueden restaurar).
- `diagnostics.simple_engine`: modelo, subtítulos palabra por palabra, grupos de subtítulos, cada tramo con
  su **motivo** (retoma, frase abandonada, ruido, otra pasada, tropiezo, pedazo suelto, silencio,
  revisión: venta reiniciada…), uso de tokens por llamada y el registro de la revisión de venta.

## 3. Modelos y proveedores que usa hoy

| Paso | Proveedor | Modelo |
|---|---|---|
| Transcripción | Deepgram | Nova-3 |
| Pasos 3, 4, 5, 6, 8 | Anthropic | `claude-sonnet-5-5` (variable `CUTSELL_SIMPLE_ENGINE_MODEL`) |
| Tiempos, silencios, cortes de cámara | — | Código propio + ffmpeg |
| Servidor | Modal | CPU (2 núcleos), sin GPU |

## 4. Costo real medido (video de venta de 106 s, 17 oct 2026)

| | Motor 1.2 | Motor 1.2.1 (con revisión que arregló el video) |
|---|---|---|
| Llamadas a Claude | 4 | 7 |
| Tokens entrada / salida | 7.092 / 1.646 | 12.413 / 3.558 |
| Claude (Sonnet 5.5 a $2 / $10 por millón*) | ~$0,031 | ~$0,060 |
| Deepgram + Modal | ~$0,013 | ~$0,013 |
| **Total por video** | **~$0,044** | **~$0,073** |

\*Precio de una fuente externa; confirmar en la consola de Anthropic.
Cuando la revisión de venta no encuentra nada (lo normal), hace 1 sola lectura y el extra es ~1 centavo.
La mayor parte del costo de Claude es **texto de salida** (las respuestas JSON y sus explicaciones).

## 5. Cómo se mide la calidad (el "Gold")

- `benchmarks/simple_engine_gold/gold_v1.json`: 26 videos reales con las decisiones de la dueña del producto
  ("esto se queda / esto se corta") por segmento. Solo se usa para medir; nunca se le muestra al motor.
- `scripts/simple_engine_eval.py`: compara el corte con el Gold (un segmento coincide si el motor deja
  ≥ 50 % de un segmento "se queda" o < 50 % de uno "se corta").
- Resultados en los 21 videos de **venta** del Gold (2 corridas):

| | Corrida 1 | Corrida 2 |
|---|---|---|
| 1.2 | 90,4 % | 96,0 % |
| 1.2 + "venta reiniciada" | +2 decisiones | +1 |
| 1.2 + "frase mezclada" | +1 | 0 |

  Ningún video empeoró con esos dos arreglos. Ojo: la 1.2 sola varía de 90 % a 96 % entre corridas
  (el modelo no responde igual cada vez); por eso se exige "ningún video peor", no solo el promedio.
- Lo que se probó y **no** se activó (y por qué):
  - 1.3 "repetido" como regla dura en la decisión: cortaba de más (92,9 % vs 94,6 %).
  - 1.4 "comprensión primero" (Claude escribe el guion de la venta y el corte sale de ahí): quedó muy bien en un
    video, pero en los videos de venta del Gold bajó a 90,4 % (escogía otra pasada, cortaba prendas en hauls).

## 6. Dónde se podría usar una IA más barata (para evaluar)

Cada paso es independiente y tiene una entrada/salida JSON fija, así que se puede cambiar de modelo paso por paso
y **medir con el Gold** antes de activarlo.

| Paso | Dificultad | Tokens típicos (entrada / salida) | Candidato a modelo más barato |
|---|---|---|---|
| 3. Pasadas | Media (entender estructura) | ~1.000 / ~300 | Sí, probable |
| 4. Decisión | **Alta** (es el corazón editorial) | ~3.000 / ~800–1.300 | Riesgoso; medir con cuidado |
| 5. Revisión de hauls | Baja la mayoría (responde "other") | ~1.600 / ~50 | Sí |
| 6. Revisión de venta | Media | ~1.900 / ~600 por lectura | Sí, y pedir respuestas más cortas |
| 8. Palabras + zoom | Baja–media | ~1.000 / ~70–400 | Sí |

Ideas de ahorro que no cambian el resultado:
- Pedir respuestas más cortas (sin explicaciones largas) en los pasos 5, 6 y 8: la salida es lo más caro.
- Caché de prompts de Anthropic: las 3 lecturas de una revisión mandan el mismo texto.
- Lotes (Batch API) si no hace falta resultado inmediato (no aplica a la app, que espera el corte).

Opciones de modelo a comparar (precios a confirmar en cada proveedor): Claude Haiku (Anthropic),
GPT mini/nano (OpenAI), Gemini Flash (Google), DeepSeek / Qwen (más baratos, revisar calidad en español y
privacidad de datos). El cambio se hace con una variable (`CUTSELL_SIMPLE_ENGINE_MODEL`) para Anthropic; para
otro proveedor habría que agregar un adaptador en `cutsell_worker/simple_engine/llm.py` (una función que recibe
el prompt y devuelve texto + uso de tokens).

Requisitos que debe cumplir cualquier modelo alternativo:
1. Español e inglés coloquial (spanglish, muletillas, transcripciones con errores).
2. Responder JSON válido de forma consistente.
3. Seguir reglas largas y precisas (el prompt de decisión tiene ~1.600 tokens de reglas).
4. Pasar el Gold: no bajar de la 1.2 en los videos de venta, en 2 corridas o más.

## 7. Archivos

| Archivo | Qué contiene |
|---|---|
| `cutsell_worker/simple_engine/engine.py` | Todo el motor (pasos 3–8) |
| `cutsell_worker/simple_engine/prompts/*.txt` | Los prompts exactos (pass, decision, review, sale_review, fixwords) |
| `cutsell_worker/simple_engine/asr.py` | Deepgram |
| `cutsell_worker/simple_engine/llm.py` | Llamada a Anthropic (modelo, reintentos) |
| `cutsell_worker/simple_engine_adapter.py` | Convierte el resultado al formato del editor de la app |
| `modal_cutsell_worker.py` | El worker en Modal |
| `benchmarks/simple_engine_gold/gold_v1.json` | El Gold (solo para medir) |
| `scripts/simple_engine_eval.py` | La medición contra el Gold |
| `docs/CUTSELL_SIMPLE_ENGINE_V1.md` | Historial técnico de versiones y mediciones |
