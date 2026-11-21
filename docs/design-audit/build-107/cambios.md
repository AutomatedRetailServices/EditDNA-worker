# Assessment de diseño · Línea de reproducción que se mueve y se arrastra (rama, sin build todavía)

- **Rama:** `feature/shop-board` · PO 21 nov 2026: "la línea negra… debe moverse independiente… con el dedo se pueda desplazar".
- **Figma** (página "📱 App build 78 · 03 Edits, Editor y Export"):

| Pantalla | Figma | Estado |
|---|---|---|
| Editor · línea de reproducción con agarradera, en cualquier punto de las pistas | 14d `3092:648` | ✅ igual a Figma (captura `editor_v4_video_overlay`); en la app la regla está una fila más abajo que en Figma 14, así que la agarradera va sobre la regla |

## Cambios
- La línea ya no está fija en el medio: avanza en play y las pistas pasan de página al llegar al borde derecho.
- Agarradera de 12 × 16 pt, color tinta, sombra suave; crece un poco mientras el dedo la mueve.
- Dedo en la línea o en la regla = mover la línea. Dedo en otra parte = correr las pistas sin cambiar el tiempo.
