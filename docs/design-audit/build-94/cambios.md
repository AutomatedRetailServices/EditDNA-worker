# Assessment de diseño · Build 94 · Wear it con video de fondo

- **Rama:** `feature/shop-board`
- **Base:** build 93. Solo pantallas cambiadas; diseñadas primero en Figma (14 nov 2026, PO: "sí a todo", "todo junto", "hasta 12 seg").

| Pantalla | Figma | Estado |
|---|---|---|
| Background › "Photo / video" (Wear it) | W1 `3065:201` | ✅ igual a Figma |
| Your board · Wear it con video (chip ▶ 0:12 · 🔊, tarjeta "Your video") | W2 `3065:301` | ✅ igual a Figma |
| Share · video (Post as TikTok video, Save to Photos video 9:16) | W4 `3065:383` | ✅ igual a Figma |
| Clon · botón cambiar cámara (build 92) | — | 🟠 en la app; falta en Figma |

## Cambios
- Solo Wear it: después de Create board la galería deja elegir **foto o video**. El video (hasta 12 s; más largo = primeros
  12 s) corre de fondo; productos, título y marca están encima **desde el primer cuadro** (sin animación, PO 14 nov).
- En Your board el video se ve en bucle y en silencio; el video que se comparte lleva su sonido.
- Compartir crea un MP4 1080 × 1920; TikTok / Instagram / Fotos usan ese video.
- El video se guarda con el board (`background.mov`); al cambiar a otra plantilla se quita.
