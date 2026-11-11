# Assessment de diseño · Build 79 · Pantallas cambiadas

- **Rama:** `feature/broll-after-stop` · **Commit de la app:** `5ad00288`
- **Base:** build 78 (ver [build-78](../build-78/parte5_resumen.md)). Solo se listan las pantallas que cambiaron.
- **Figma:** `rXfUuJQgwZIZJTuONvU1Z7`, página "📱 App build 78 · 01 Inicio, Record y Upload", fila E.

| Pantalla | Archivo (SwiftUI) | Figma | Estado |
|---|---|---|---|
| Cámara de Grabar, B-roll apagado | `CreateCameraView.swift` + `BrollAfterSwitch` (`BrollAfterStop.swift`) | E1 `3023:324` | ✅ |
| Cámara de Grabar, B-roll encendido | igual | E1b `3024:331` | ✅ |
| Cámara de B-roll | `CreateCameraView.swift` (fase B-roll) + `BrollCameraLabel`, `BrollShotPill`, `BrollShotsStrip`, `BrollDoneButton` | E3 `3023:387` | ✅ con notas |
| Ready to edit | `BrollSummaryView` (`BrollAfterStop.swift`) | E4 `3023:439` | ✅ con notas |
| Choose your style | `StyleSheetView.swift` (sin cambios) | E5 `3023:468` | sin cambios |

## Cámara de Grabar (E1 / E1b)
- Se quitaron los chips "Talk" / "B-roll" del build 78.
- **Interruptor apagado (E1):**
  - Botón "+ B-roll", Plus Jakarta Sans Bold 15, blanco.
  - Cápsula de 120×34, negro al 45 %, borde blanco al 35 % de 1 pt.
  - Va 12 pt arriba de las duraciones.
- **Interruptor encendido (E1b):**
  - Botón "✓ B-roll after", cápsula de 150×34 con degradado #2F7BFF → #0A55F0 y sombra azul (0, 4, 10, #1769FF al 40 %).
  - Encima, separado 8 pt, el aviso "After STOP you'll add product shots": Medium 12 blanco, cápsula de 250×30, negro al 55 %.
- Mientras se graba, el interruptor se esconde.
- Todo lo demás de la cámara queda igual que en el build 78: Flip, Flash, Grid, Mirror, Auto-pause, Clon, Script, 10m / 60s / 3m, Upload y el botón de grabar.

## Cámara de B-roll (E3)
- **Arriba:**
  - etiqueta "B-roll", cápsula de 110×34 #1FA971, Bold 16 blanco, al centro;
  - mientras graba, la pastilla "● Shot N · 0:03": Bold 13 blanco sobre #E5484D, alto 30, ancho mínimo 150, arriba a la izquierda.
- **Columna de la derecha:** solo Flip, Flash, Grid y Mirror. No están Auto-pause, Clon, Script ni las duraciones.
- **Abajo, de arriba hacia abajo:**
  - "Auto-pause off · records everything", Medium 12 blanco al 85 %;
  - las miniaturas de las tomas:
    - cada una de 54×72, radio 10, borde blanco de 2 pt, con "×" en un círculo de 18 #0B1020;
    - al final un "+" punteado de 54×72, con borde blanco de 2 pt.
- **Botón de grabar:**
  - mismo aro;
  - mientras graba, el centro es un círculo rojo #E5484D de 64.
- "Upload" queda en el mismo lugar que en Grabar, pero abre la galería del iPhone para agregar clips como B-roll.
- "Done": azul 3D de 92×46 (el mismo estilo de "App · Botón 3D / Azul"), a la derecha, a la altura del botón de grabar.
- **Notas:**
  - "‹" vuelve a la cámara de Grabar y descarta lo grabado. Figma no define este caso.
  - El "+" punteado graba una toma nueva.

## Ready to edit (E4)
- **Fondo y título:**
  - fondo "App · Fondo (HomeBackdrop)";
  - botón atrás: círculo de 40 en 3D blanco con "‹" Bold 22 #0B1020;
  - "Ready to edit", ExtraBold 29, #0B1020.
- **Tarjetas** (degradado #FFFFFF → #F7F9FF, radio 24, base #D3DDF5 de 3 pt y sombra suave):
  - **"Your video"** (Bold 13 #6B7385):
    - miniatura de 64×94, radio 14;
    - "Talking · 0:24", Bold 16 #0B1020;
    - "Silences removed", Medium 13 #6B7385.
  - **"B-roll · N shots"** (Bold 13 #6B7385):
    - miniaturas de 64×94 con "×" en un círculo de 20 #0B1020;
    - "+" en una caja de 64×94 #F3F5F9 con borde #E6E9EF;
    - "We'll place them over your voice at the right moments.", Medium 13 #6B7385.
- **Botón:** "Continue", azul 3D de 52 de alto ("App · Botón 3D / Azul") → "Choose your style".
- **Notas:**
  - Con una sola toma dice "B-roll · 1 shot". Figma solo muestra el plural.
  - "Silences removed" solo aparece si Auto-pause estaba encendido al grabar la parte hablada.
  - El "+" vuelve a la cámara de B-roll para grabar o subir más tomas.
  - Las posiciones salen de la distribución de la pantalla (márgenes de 20–24 pt, igual que Figma), no de coordenadas fijas. Así sirve para cualquier tamaño de iPhone.

## Textos en español (nuevos)
| Inglés | Español |
|---|---|
| "+ B-roll" | "+ B-roll" |
| "✓ B-roll after" | "✓ B-roll después" |
| "After STOP you'll add product shots" | "Después de STOP agregas tomas del producto" |
| "Shot N" | "Toma N" |
| "Auto-pause off · records everything" | "Auto-pausa apagada · graba todo" |
| "Ready to edit" | "Listo para editar" |
| "Your video" | "Tu video" |
| "Talking · 0:24" | "Hablando · 0:24" |
| "Silences removed" | "Silencios quitados" |
| "B-roll · N shots" | "B-roll · N tomas" |
| "We'll place them over your voice at the right moments." | "Las pondremos sobre tu voz en los momentos justos." |
