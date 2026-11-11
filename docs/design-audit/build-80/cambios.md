# Assessment de diseño · Build 80 · Pantallas cambiadas

- **Rama:** `feature/broll-after-stop` · **Commit de la app:** `262c9378`
- **Base:** build 79 (ver [build-79](../build-79/cambios.md)). Solo se listan las pantallas que cambiaron.
- **Figma:** `rXfUuJQgwZIZJTuONvU1Z7`, página "📱 App build 78 · 01 Inicio, Record y Upload", filas E y F.

| Pantalla | Archivo (SwiftUI) | Figma | Estado |
|---|---|---|---|
| Cámara, B-roll encendido → Upload | `CreateCameraView.swift` | F1 `3026:328` | ✅ (sin cambio visual; Upload abre la galería de 2 pasos) |
| Your talking video (1 of 2) | `BrollGalleryFlow` (`BrollGallery.swift`) | F2 `3026:366` | ✅ con notas |
| B-roll shots (2 of 2) | `BrollGalleryFlow` + `BrollGalleryTile` | F3 `3026:416` | ✅ con notas |
| Ready to edit, desde galería | `BrollSummaryView` (`BrollAfterStop.swift`) | F4 `3026:471` | ✅ |
| Ready to edit, tomas grabadas | `BrollSummaryView` | E4 `3023:439` | ✅ |
| Choose your style | `StyleSheetView.swift` (sin cambios) | F5 `3026:521` | sin cambios |
| Editor: fotos de B-roll | `EditorV4View.swift` (`MediaOverlayOnVideo`), `DeviceExport.swift` | no está en Figma | zoom lento, ver abajo |

## Galería de 2 pasos (F2 / F3)
- **Fondo y encabezado:**
  - fondo #FFFFFF;
  - "✕", SemiBold 18 #0B1020, a la izquierda;
  - título centrado, SemiBold 22 #0B1020: "Your talking video" / "B-roll shots";
  - chip "1 of 2" / "2 of 2" a la derecha: 64×24, #EEF3FF, Bold 12 #1769FF;
  - línea #E6E9EF debajo del encabezado.
- **Cuadrícula:**
  - 3 columnas, separación de 6;
  - cuadros con proporción 122×158.6 y radio 12;
  - duración abajo a la derecha, SemiBold 12 blanco; en las fotos dice "PHOTO".
- **Seleccionado:**
  - borde #1769FF de 2 pt;
  - número arriba a la derecha en un círculo de 24 #1769FF con borde blanco de 2 pt, Bold 12 blanco. Se usa el estilo nuevo en todos; Figma mezclaba el estilo viejo #0D9EFF/cyan y la PO eligió el nuevo.
- **Paso 2:** cada toma elegida lleva "✓ B-roll" abajo a la izquierda (70×24, #1769FF, Bold 11 blanco).
- **Panel de abajo** (blanco, 204 de alto, sombra hacia arriba):
  - "N videos selected" / "N shots selected", Regular 14 #6B7385;
  - el texto de ayuda, Medium 13 #6B7385;
  - en el paso 2, "Record shots instead", Bold 13 #1769FF;
  - botón azul 3D de 52: "Next: add B-roll" / "Add N shots".
- **Notas:**
  - La "✕" en el paso 2 vuelve al paso 1 y conserva lo elegido (decisión de la PO, 11 nov).
  - El botón se ve al 45 % mientras no hay nada elegido.
  - Mientras se copian los archivos, el botón muestra el aro de carga de la app (no está en Figma).
  - El paso 1 no tiene Single / Bulk / Combine ni el chip B-roll.

## Ready to edit (E4 / F4)
- **"Your video":**
  - grabado: "Talking · 0:24" + "Silences removed" (solo si Auto-pause estaba encendido);
  - desde la galería: "Talking · N clips · 1:20" + "Joined in your order".
- **"B-roll · N shots":**
  - cada miniatura (64×94, radio 14) lleva su número abajo a la izquierda: círculo de 22 #1769FF con borde blanco de 2 pt, Bold 12 blanco;
  - la "×" sigue arriba a la derecha;
  - mantener presionado y arrastrar cambia el orden, y los números se actualizan;
  - "Hold and drag to change the order.", Medium 13 #6B7385.
- **Fila "Keep this order":**
  - fondo #F3F5F9, radio 14, alto 50;
  - "Keep this order", Bold 15 #0B1020;
  - "Off = AI picks the best moment for each", Medium 11 #6B7385;
  - interruptor de 48×28 #1769FF con círculo blanco; encendido por defecto.
- **"+":**
  - si vino de la cámara de B-roll, vuelve a esa cámara;
  - si vino de la galería, abre el selector de fotos y videos del iPhone para agregar más tomas.
- **"‹" desde la galería:** vuelve a "B-roll shots" con lo elegido.

## Fotos de B-roll en el editor
- El motor las pone como superposición tipo "photo" a pantalla completa (recortada a 9:16), sin sonido.
- El zoom va en línea recta de 1.0 a 1.08 durante su tiempo, centrado.
- El editor y la exportación del iPhone hacen el mismo zoom que el servidor.

## Textos en español (nuevos)
| Inglés | Español |
|---|---|
| "Your talking video" | "Tu video hablando" |
| "B-roll shots" | "Tomas de B-roll" |
| "1 of 2" / "2 of 2" | "1 de 2" / "2 de 2" |
| "Pick one or more. We'll join them in this order." | "Elige uno o varios. Los unimos en este orden." |
| "Next: add B-roll" | "Siguiente: agregar B-roll" |
| "N videos selected" / "N shots selected" | "N videos elegidos" / "N tomas elegidas" |
| "Tap them in the order they'll appear: 1, 2, 3." | "Tócalas en el orden en que van a salir: 1, 2, 3." |
| "Record shots instead" | "Mejor grabar tomas" |
| "Add N shots" | "Agregar N tomas" |
| "✓ B-roll" / "PHOTO" | "✓ B-roll" / "FOTO" |
| "Talking · N clips · 1:20" | "Hablando · N clips · 1:20" |
| "Joined in your order" | "Unidos en tu orden" |
| "Hold and drag to change the order." | "Mantén presionado y arrastra para cambiar el orden." |
| "Keep this order" | "Mantener este orden" |
| "Off = AI picks the best moment for each" | "Apagado = la IA elige el mejor momento para cada una" |
