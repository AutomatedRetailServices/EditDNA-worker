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
