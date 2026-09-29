# Yaskira 01–10 — recuperación de decisiones de la propietaria

**Fecha de recuperación:** 2026-09-29. Fuente: búsqueda de mensajes de la conversación «Branch · Comparar OpenAI Deepgram» (mensajes UTC). Los tiempos de edición son tiempos del RAW indicados en la revisión; verificar identidad del RAW antes de utilizarlos. Esta tabla separa una orden textual de una aceptación «sí» de una propuesta del asistente. No sustituye alineación de palabras ni la revisión de un render.

| RAW | Decisión recuperada | Mensaje de la propietaria (UTC) | Tipo de respaldo / pendiente |
| --- | --- | --- | --- |
| `Yaskira/01.mp4` | KEEP 0:29–0:32 («I was like, oh, I…»); DELETE **solo** 0:58–0:59 («because…»); KEEP 1:07–1:08 («He is…»); KEEP completo 1:27–1:30. | 2026-09-25 21:00:41, 21:15:36 | Órdenes explícitas recuperadas. La propuesta de quitar 0:55–0:59 del 29-sep **no fue aprobada** y no modifica estas decisiones. No interpretar estos puntos como permiso para borrar 0:31–0:37. |
| `Yaskira/02.mp4` | DELETE 0:49–0:50 (leyendo/ruptura visual) y 1:08–1:10 (frase abandonada); KEEP el resto. | 2026-09-25 21:26:07 | Instrucción explícita recuperada. |
| `Yaskira/03.mp4` | KEEP todo, incluido el corte visual de 0:06. | 2026-09-25 21:31:09 | Respuesta «sí» a la propuesta de conservar el RAW completo; requiere preservar la propuesta como contexto. |
| `Yaskira/04.MP4` | DELETE 0:00–0:05; DELETE 0:27–0:28 («Voy a tener que editarlo, pero bueno»); DELETE después de 0:44. KEEP 0:05–0:27 y 0:28–0:44. | 2026-09-25 21:40:39, reafirmación 21:42:17 | Correcciones explícitas recuperadas. |
| `Yaskira/05.MP4` | DELETE 0:00–1:38; KEEP 1:38–2:17; DELETE 2:17–2:20. | 2026-09-25 22:03:36 | «Sí, correcto, confirmado» a una propuesta con esos tres bloques. Ver también `YASKIRA_05_HUMAN_EDITORIAL_REFERENCE_20260925.md`. |
| `Yaskira/06.MP4` | KEEP solamente 1:25–1:47; DELETE el resto. | 2026-09-25 22:11:21 | Orden explícita recuperada. |
| `Yaskira/07.mp4` | **SIN GOLD APROBADO.** La propietaria rechazó `Video00_Human_Gold.mp4` por estar mal editado. | 2026-09-29 12:00:14 | Rechazo explícito. Los ejemplos históricos de Video00 no son un mapa completo aprobado de `Yaskira/07.mp4`; verificar identidad y volver a revisar el RAW. |
| `Yaskira/08.mp4` | KEEP solamente 2:00–2:27; DELETE el resto. | 2026-09-25 22:27:30 | Orden explícita recuperada. No hay lista de palabras individuales aprobadas para el resto. |
| `Yaskira/09.mp4` | DELETE 1:06–1:13; KEEP propuesto: 0:05–0:14, 0:19–0:28, 0:52–1:06, 1:13–1:31, 1:37–2:00; DELETE el resto. | 2026-09-25 22:38:39 («eliminar 1:06–1:13»), 22:39:38 («sí») | La eliminación es orden explícita; cinco bloques se aceptaron mediante «sí» a la propuesta inmediatamente anterior. Conservar esa relación, no atribuirle a la usuaria el texto de los cinco bloques como cita literal. |
| `Yaskira/10.MOV` | KEEP 0:16–0:25, 0:55–1:04, 1:15–1:19, 1:23–1:37; DELETE el resto. | 2026-09-25 23:14:07; confirmaciones «sí» 23:15:11, 23:39:59 | Aceptación recuperada de propuesta con cuatro bloques; la propietaria no escribió necesariamente cada bloque en el «sí». |

## Uso y límites

- Los sellos UTC identifican mensajes recuperados; no son enlaces a mensajes ni sustituyen el texto original completo de cada intercambio. Los pasajes entre comillas anteriores solo se marcan literales cuando la búsqueda devolvió esas palabras; el resto es una reconstrucción de decisiones y contexto.
- Verificar hash/duración por fuente. Para 01, el archivo local `01-VIDEO-2026-07-30-09-21-35(4).mp4` coincide por SHA-256 (`750e989c105e1bbf516ca180dccb01d4f688e08215ccf5d5075b3c3d2cecc52f`) con la identidad `Yaskira/01.mp4` del resultado histórico 36321269066. La identidad de los otros nueve no quedó comprobada mediante esta recuperación conversacional.
- Ningún «sí» aislado autoriza una etiqueta sin la propuesta anterior. No convertir el render rechazado de Video07 en Gold ni sumar la propuesta nueva de Video01 al dataset.
