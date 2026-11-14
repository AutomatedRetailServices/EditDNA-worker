# Assessment de diseño · Build 93 · Wear it, quitar productos, "YOUR BRAND" editable, sonido del B-roll

- **Rama:** `feature/shop-board`
- **Base:** build 92. Solo se listan las pantallas que cambiaron. Todas las pantallas nuevas se diseñaron primero en
  Figma (14 nov 2026) y la app las sigue.

| Pantalla | Figma | Estado |
|---|---|---|
| Choose a style (4ª plantilla Wear it, sin "+ More soon") | S3w `3064:206` | ✅ igual a Figma |
| Board Wear it (foto de ella, productos a la izquierda, título blanco abajo) | `3064:197` | ✅ igual a Figma |
| Your board · × roja para quitar el producto elegido | S4x `3061:190` | ✅ igual a Figma |
| Title · campo "Bottom line" (YOUR BRAND) | S4d2 `3062:193` | ✅ igual a Figma |
| B-roll · Ready to edit · bocina por toma | E4s `3060:354` | ✅ igual a Figma |
| Clon · botón cambiar cámara (build 92) | — | 🟠 en la app; **falta en Figma** |

## Cambios
- **Wear it (PO 14 nov):** 1–4 productos con lugares propios a la izquierda; más de 4 en dos columnas chicas. Al
  tocar Create board se abre la galería para su foto; si cancela, fondo café de muestra hasta que la elija en
  Background. Sombra suave abajo para que el texto blanco se lea.
- **× roja:** en la esquina del marco azul del producto elegido, gira con él; no aparece si solo queda 1 producto.
  Los demás productos se quedan donde estaban.
- **Bottom line:** texto libre; se muestra en mayúsculas; vacío = sin línea. Los boards guardados antes siguen con
  "YOUR BRAND".
- **Bocina del B-roll:** apagada por defecto (B-roll en silencio); encendida = su sonido debajo de la voz (30 %).
  Las fotos no tienen bocina.
