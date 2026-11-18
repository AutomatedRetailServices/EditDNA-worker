# Assessment de diseño · Build 99 · Choose a style en carrusel + My idea

- **Rama:** `feature/shop-board`
- **Base:** build 98. Pantallas diseñadas primero en Figma (18 nov 2026, página "💡 10 · Shop board (propuesta)").

| Pantalla | Figma | Estado |
|---|---|---|
| Choose a style · carrusel (Collage, Wear it, sus estilos, ＋ My idea) | S3p-B `3078:319` (elegido por la PO: "me gusta la B") | ✅ igual a Figma |
| My idea · leyendo su captura | S3i-C `3078:418` | ✅ igual a Figma |
| My idea · resultado + "Use this style" | S3i-D `3078:499` | ✅ igual a Figma |
| Clon · botón cambiar cámara (build 92) | — | 🟠 en la app; falta en Figma |

## Cambios
- Choose a style va después de agregar los productos. Solo dos estilos fijos: Collage y Wear it (PO: "solo collage,
  wear it and the upload the idea"); los templates inventados se quitaron.
- Carrusel: se desliza de lado, una página por estilo, con el nombre en una píldora y puntos abajo.
- ＋ My idea: ella sube la captura de un board; el servidor (`POST /v1/shop/idea`) devuelve el mapa (fondo, textos,
  lugares de productos por tipo) y la app pone SUS productos en esos lugares, por tipo. Se guarda como estilo en el
  iPhone (no en el servidor). Mantener presionado: Rename / Delete.
- Prueba real (18 nov, US$0.0135): fondo y textos bien; cada producto cae 98-100 % sobre su lugar real (pants,
  shirt, shoes, sweater, por tipo como la app). Diferencias: la IA leyó los productos derechos (en la captura tenían
  4-6° de giro) y el título sale un poco más chico. La primera hoja salió mal por mi dibujo de prueba (fotos sin
  recortar y en otro orden), no por la IA.
