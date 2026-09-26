# Yaskira 05 — referencia editorial humana

Fecha: 25 septiembre 2026. Fuente cruda completa: `Yaskira/05.MP4`
(139.965 s, 464x832, 14,847,667 bytes).

Esta referencia registra la decisión explícita de la Product Owner. Los
tiempos son de navegación; los límites técnicos deben ajustarse a evidencia
por palabra y audiovisual antes del render.

| Tiempo visible | Evidencia | Decisión humana | Estado |
| --- | --- | --- | --- |
| `0:00–0:05` | Preparación y entrada | REMOVE | Confirmado |
| `0:05–0:14` | Primer comienzo abandonado | REMOVE | Confirmado; cubierto por el retake final |
| `0:14–0:29` | Interrupción con “Achi…” y continuación fallida | REMOVE | Confirmado |
| `0:29–0:51` | Primer intento con trabadas y frase abandonada | REMOVE | Confirmado; cubierto por el retake final |
| `0:51–0:53` | Pausa entre intentos | REMOVE | Confirmado |
| `0:53–1:07` | Segundo intento incompleto | REMOVE | Confirmado; reemplazado por el retake final |
| `1:07–1:27` | Conversación fuera del contenido y proceso de grabación: “estoy haciendo video”, “ok, vamos otra vez” | REMOVE | Confirmado; BTS/reinicio explícito |
| `1:27–1:38` | Preparación antes del retake final | REMOVE | Confirmado |
| `1:38–2:17` | Toma final coherente: problema, solución, uso y CTA | KEEP completo | Confirmado |
| `2:17–2:20` | Cola final | REMOVE | Confirmado |

## Resultado humano final

Conservar únicamente `1:38–2:17`.

## Consecuencia para la calibración

- El motor debe agrupar varios intentos separados, interrupciones y BTS con
  el retake completo posterior de la misma comunicación.
- La cobertura debe comprobarse a nivel de afirmaciones: la toma final
  conserva el problema, la recomendación, el uso del producto y el CTA.
- “Ok, vamos otra vez” aporta autoridad explícita de reinicio; no es contenido
  de audiencia.
- El intervalo eliminado no es un único silencio: mezcla preparación,
  delivery parcial, interrupciones, fumbles y conversación fuera del video.
  Por tanto, no puede resolverse únicamente con ASR o VAD.
