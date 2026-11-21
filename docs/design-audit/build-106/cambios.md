# Assessment de diseño · Línea de tiempo: fotos y videos encima tal como son (rama, sin build todavía)

- **Rama:** `feature/shop-board` · PO 20 nov 2026: "todo lo que se overlay en foto o video debe presentarse tal como es".
- **Figma** (página "📱 App build 78 · 03 Edits, Editor y Export"):

| Pantalla | Figma | Estado |
|---|---|---|
| Editor · fila de overlays con la imagen real + etiqueta pequeña | 14c `3090:733` (copia de 14 `3015:8052`, que queda igual como referencia) | ✅ igual a Figma (captura del simulador `editor_v4_video_overlay`) |

## Cambios
- Antes: un video o foto encima con nombre ("Video 1", "Photo 1") se dibujaba como barra azul sólida con letra blanca
  (diseño "White Indigo", Figma 04_Editor 9). Tapaba los cuadros.
- Ahora: se ven los cuadros del video (o la foto) en todo el bloque; el nombre va en una etiqueta oscura (60 %) de
  9 pt arriba a la izquierda. Borde azul de 1; seleccionado, de 2.
- Los textos encima (por ejemplo "50% OFF") no cambian: siguen azules (no tienen imagen).
