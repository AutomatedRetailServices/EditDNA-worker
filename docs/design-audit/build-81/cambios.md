# Assessment de diseño · Build 81 · Shop board (pantallas nuevas)

- **Rama:** `feature/shop-board` · **Commit de la app:** `03fc369f`
- **Base:** build 80 (ver [build-80](../build-80/cambios.md)). Solo se listan las pantallas nuevas o cambiadas.
- **Figma:** `rXfUuJQgwZIZJTuONvU1Z7`, página "💡 10 · Shop board (propuesta)" (workflow `3039:240`).
- **Código:** `ShopBoardViews.swift` (pantallas), `ShopBoardModel.swift` (estado, dibujo del board, exportación, Shape it, compartir), `ShopBoardLogic.swift` (reglas probadas), `ShopBoardImaging.swift` (fuentes, quitar fondo, clasificar, colores, texturas). Inicio: `HomeHubView.swift`. Créditos de fuentes: `YouView.swift`.

| # | Pantalla | Figma | Estado |
|---|---|---|---|
| 1 | Inicio con tarjeta "Shop board" · "NEW" | `3032:3` | ✅ (separación de 24 pt bajo los atajos; en Figma la tarjeta se monta sobre Stats) |
| 2 | Add products | `3032:73` | ✅ con notas |
| 2b | Shape it | `3033:77` | ✅ (llama al servidor; sin deploy muestra "Retry") |
| 3 | Choose a style | `3032:143` | ✅ (vistas previas con tus productos) |
| 4 | Your board | `3032:237` | ✅ con notas |
| 4a | Title & fonts | `3039:152` | ✅ |
| 4b | Background · Colors | `3035:71` | ✅ con notas |
| 4c | Background · Textures | `3035:146` | ✅ con notas |
| 5 | Share | `3032:286` | ✅ con notas |
| — | Font credits (Profile → Help) | no está en Figma | 🔴 estilo sencillo de la app |
| — | Pestaña "My photo" | no está en Figma | 🔴 solo abre las fotos |
| — | Lista de "From my products", cuadro de "Paste a link", mensajes de error de Shape it, avisos al compartir | no están en Figma | 🔴 estilo sencillo de la app |

## Decisiones provisionales (Figma no las define; aprobadas por la PO para probar en el build 81)
1. Las 3 plantillas usan el mismo acomodo; solo cambian colores y textos. "Shuffle" da el otro acomodo (intercambia lugares y, cada dos veces, refleja el board).
2. Figma muestra 6 lugares. Los productos 7 y 8 van en dos espacios libres (izquierda al medio y abajo a la izquierda).
3. Los boards de Figma miden 390×844; la app usa 9:16 (exporta 1080×1920), como pide la especificación.
4. Los nombres de las plantillas salen cortados en Figma y así quedaron.
5. "YOUR BRAND" abajo del board igual que Figma, sin campo para editarlo.
6. En "Your board", Background muestra "✨ Match my products" (Figma dice "Cream", pero Match viene encendido).
7. Estados sin diseño: Match apagado, color de producto elegido, color de título a mano ("Your pick") y producto sin foto ("+ Add photo").
8. Los 3 círculos de "From my products / From photos / Paste a link" no tienen ícono en Figma: azules sin ícono.
9. Se resalta la última fuente usada en Add products y la fila de TikTok en Share.
10. "✦ Uses AI · 1 credit per item" igual que Figma, aunque no hay sistema de créditos.
11. En Textures, Match tiñe la textura con el color de los productos.
12. "Post as TikTok photo" dice "Carousel" (Figma), aunque el board es una sola imagen.
13. El caption por defecto mantiene "for fall 🍂".
14. "From my products": los productos no tienen foto propia (solo la portada de un video); después de elegir uno, se pide su foto desde Fotos.

## Fuentes (todas SIL Open Font License 1.1, confirmado en OFL.txt y METADATA.pb de google/fonts)
| Pareja | Título | Línea script |
|---|---|---|
| Classic | Bodoni Moda (Owen Earl) | Great Vibes (Robert Leuschke) |
| Editorial | Playfair Display (Claus Eggers Sørensen) | Parisienne (Astigmatic) |
| Bold | Bebas Neue (Ryoichi Tsunekawa) | Dancing Script (Impallari Type) |
| Elegant | Cinzel (Natanael Gama) | Pinyon Script (Nicole Fally) |
| Soft | Cormorant Garamond (Christian Thalmann) | Allura (Robert Leuschke) |
| Modern | Italiana (Santiago Orozco) | Sacramento (Astigmatic) |

Archivos: `mobile/ios/CutSell/Fonts/ShopBoard/<Familia>/ShopBoard-*.ttf` con `OFL-<Familia>.txt` al lado. Fuente original: `https://raw.githubusercontent.com/google/fonts/main/ofl/<familia>/`.

## Texturas
Linen, Paper, Marble, Wood, Knit, Satin, Concrete y Velvet se dibujan con código en el iPhone (ruido con semilla, vetas, grano y puntadas sobre los colores de las muestras de Figma). No hay archivos de imagen ni licencias.

## Compartir
- **TikTok:** el flujo existente solo subía videos. Se agregó foto: guarda en Fotos, copia el caption y abre TikTok.
- **Instagram Stories:** falta un Facebook App ID. Por ahora guarda el board, copia el link y abre la cámara de historias; el sticker de link lo pone el usuario.
