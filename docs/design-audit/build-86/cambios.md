# Assessment de diseño · Build 86 · Pestaña "Shop board"

- **Rama:** `feature/shop-board`
- **Base:** build 85 (ver [build-85](../build-85/cambios.md)). Solo se listan las pantallas que cambiaron.
- **Decisión de la PO (12 nov):** los Shop boards salen de Edits y tienen su propia pestaña, entre Edits y Plan, para
  que no se mezclen con los videos. Edits vuelve a estar como antes del build 85.

| Pantalla | Figma | Estado |
|---|---|---|
| Barra de abajo con 5 pestañas | componente "App · Barra de abajo" `3012:31` | ✅ |
| Pestaña Shop board (boards guardados) | `3055:516` "3e · Shop board (pestaña nueva)" (página 03 Edits) | ✅ |
| Edits | igual que el build 84 (sin el segmento "Shop boards") | ✅ |

## Barra de abajo
- Create · Edits · **Shop board** · Plan · Profile. Las 5 pestañas miden 68 de ancho.
- Ícono nuevo de Shop board: un cuadro con tres cuadritos adentro (un collage), línea de 1.8 pt, como los otros
  íconos. Archivo: `Assets.xcassets/WhiteIndigo/WITabShopBoard.imageset`.
- El cambio se hizo en el componente, así que todas las pantallas de Figma ya tienen la barra de 5 pestañas.

## Pestaña Shop board
- Arriba, el logo (sin buscador ni menú). Título ExtraBold 32 "Shop boards" y subtítulo Medium 15 "Your boards, saved
  as you make them.".
- Abajo, la misma cuadrícula del build 85: "+ New Shop board" y los boards guardados (miniatura 9:16, "•••" → Delete,
  título y fecha).
- La tarjeta "Shop board" en Inicio sigue igual.
