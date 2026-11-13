# Assessment de diseño · Build 85 · Shop board: mover productos a mano

- **Rama:** `feature/shop-board`
- **Base:** build 84 (ver [build-83](../build-83/cambios.md)). Solo se listan las pantallas que cambiaron.
- **Figma:** `rXfUuJQgwZIZJTuONvU1Z7`, página "💡 10 · Shop board (propuesta)". Cambiado el 12 nov a pedido de la PO.
  La versión anterior de S4 quedó en "🗄 Antes · S4 · Your board (build 84)" (`3053:170`).

| Pantalla | Figma | Estado |
|---|---|---|
| Your board (línea de ayuda nueva) | S4 `3032:237` | ✅ |
| Your board · moviendo un producto (nuevo estado) | S4e `3053:196` | ✅ |

## Cómo funciona
- **Arrastrar** un producto lo mueve. Al tocarlo o arrastrarlo pasa al frente y lleva un borde #1769FF de 2 pt
  (radio 10, 4 pt alrededor).
- Con un producto elegido: **pellizcar** cambia el tamaño (de 0.3× a 3×) y **girar con dos dedos** lo voltea.
  Funciona en cualquier parte del board.
- **Tocar el board** (fuera de un producto) termina.
- **Shuffle** y elegir otra plantilla borran lo movido a mano.
- Lo movido se guarda en medidas del board, así la imagen de 1080×1920 sale igual que en pantalla. El borde azul no
  sale en la imagen.
- Línea de ayuda bajo las tarjetas (Medium 12, centrada):
  - sin producto elegido, #6B7385: "Drag a product to move it · pinch to resize · twist to turn";
  - con producto elegido, #1769FF: "Pinch to resize · twist to turn · tap outside to finish".

## Shop boards guardados (Edits › Shop boards)
- **Figma:** página "📱 App build 78 · 03 Edits, Editor y Export", pantalla nueva `3055:516` "3e · Edits · Shop boards".
- Cada board **se guarda solo en el iPhone** mientras se arma (1 s después de cada cambio y al cerrar). Guarda:
  productos sin fondo (PNG), lo movido a mano, plantilla, letras, fondo, link, caption y una miniatura de 270×480.
  No sale del teléfono.
- **Edits:** segmento nuevo "Shop boards" (con número) después de Posted. Subtítulo: "Your Shop boards, saved as you
  make them."
- **Cuadrícula** de 3 columnas:
  - primera tarjeta "+ New Shop board": borde punteado #1769FF sobre #EEF3FF, radio 16;
  - después, los boards: miniatura 9:16 con radio 16, "•••" blanco arriba a la derecha, título Bold 12 #0B1020 y
    fecha Medium 11 #6B7385 ("Today", "Yesterday" o "Nov 10").
- Tocar un board lo abre en **"Your board"** tal como quedó.
- "••• → Delete" pregunta antes: "Delete this Shop board?" / "It is only removed from CutSell. Pictures you saved to
  Photos stay."
