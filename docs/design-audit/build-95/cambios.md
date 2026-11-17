# Assessment de diseño · Build 95 · Sin Shape it + zoom inteligente

- **Rama:** `feature/shop-board`
- **Base:** build 94. Ninguna pantalla nueva: se quita un botón; el zoom cambia cómo se mueve el video, no la pantalla.

| Pantalla | Figma | Estado |
|---|---|---|
| Your board · botón "✦ Shape it" y su texto de ayuda | (se quita) | ✅ escondido (PO 16 nov: "trabajaremos sin shape it ahora mismo") |
| Editor · rombos (keyframes) del Auto Zoom AI | sin cambio de diseño | ✅ igual; ahora con la velocidad y el punto del zoom del motor |
| Clon · botón cambiar cámara (build 92) | — | 🟠 en la app; falta en Figma |

## Cambios
- Shop board: el botón Shape it ya no aparece (cuesta dinero por foto). El código queda guardado y apagado
  (`ShopShapeFeature.isOn = false`).
- Zoom inteligente: los zooms del motor llegan al editor con su velocidad de entrada / salida (`ramp_in` /
  `ramp_out`) y con su punto (`x` / `y`: la cara, el producto, el brazo...). Zooms viejos sin esos datos: igual que antes.
