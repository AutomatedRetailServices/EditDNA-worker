# Assessment de diseño · Build 89 · Your board más grande, tarjetas en una línea, título que se mueve

- **Rama:** `feature/shop-board`
- **Base:** build 88 (ver [build-86](../build-86/cambios.md) y [build-85](../build-85/cambios.md)). Solo se listan las
  pantallas que cambiaron.
- **Pedido de la PO (13 nov, con captura del build 88):** "la pantalla de editar es muy chiquita", "los botones
  Shuffle etc. están mal, palabras muy largas, desalineadas", "el título no se puede mover".

| Pantalla | Figma | Estado |
|---|---|---|
| Your board | S4 `3032:237` / S4e `3053:196` | 🟠 la app cambió; **Figma todavía no** (no se pudo conectar a Figma) |

## Cambios
- **Board más grande:** ocupa todo el espacio libre entre "Your board" y las tarjetas (antes medía 218 pt de ancho
  como en Figma y quedaba un hueco abajo).
- **Tarjetas Title / Shuffle / Background:** el detalle va en una sola línea, cortado con "…" si no cabe.
  - Title muestra solo el título grande ("HOLIDAY").
  - Background muestra "✨ Match" cuando el fondo es automático.
- **El título se mueve** como un producto: arrastrar, pellizcar y girar con dos dedos. Lleva el mismo borde azul al
  tocarlo. Shuffle lo regresa a su lugar. Se guarda con el board.
- Línea de ayuda: "Drag a product or the title · pinch to resize · twist to turn".

## Pendiente
- Actualizar S4 y S4e en Figma con el board grande y las tarjetas en una línea.
