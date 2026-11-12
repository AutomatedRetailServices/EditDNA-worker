# Assessment de diseño · Build 83 · Shop board: links según dónde se publica

- **Rama:** `feature/shop-board`
- **Base:** build 82 (mismo diseño que el [build 81](../build-81/cambios.md)). Solo se listan las pantallas que cambiaron.
- **Figma:** `rXfUuJQgwZIZJTuONvU1Z7`, página "💡 10 · Shop board (propuesta)". Cambiado el 12 nov a pedido de la PO.
  Las versiones anteriores no se borraron: están a la derecha, en "🗄 Reemplazadas (así estaban en el build 81)"
  (`3051:161` Add products y `3051:223` Share).

| Pantalla | Figma | Estado |
|---|---|---|
| Add products (sin "Paste a link") | S2 `3032:73` | ✅ |
| Share (link de Amazon / afiliado opcional, textos por destino) | S5 `3032:286` | ✅ |
| Aviso de TikTok Shop (nuevo, se abre al tocar "Post as TikTok photo") | S5b `3051:264` | ✅ |

## Por qué
- TikTok no deja tocar links en el caption, y los productos de TikTok Shop solo se ligan dentro de TikTok
  ("Add link" → "Products").
- El link de Amazon / afiliado sirve para la bio, el sticker de la historia de Instagram y para copiar.
- Por eso el link ya no se pide al agregar productos; se pide en Share según dónde se publica.

## Add products (S2)
- Quedan "From my products" y "From photos". "Your items" sube 74 pt.

## Share (S5)
- Campo: "Amazon / affiliate link", texto de ejemplo "amazon.com/…  (optional)".
- "Post as TikTok photo": "Tag your TikTok Shop product in TikTok" → abre S5b.
- "Instagram story": "Your link goes on the link sticker".
- "Save to Photos": "9:16 image · link & caption copied" (ahora también copia link y caption).
- Nota de abajo: "TikTok Shop products are tagged inside TikTok. Your Amazon / affiliate link goes in your bio, your
  story sticker or wherever you paste it. We add "#ad" so your post follows the rules for affiliate links."

## Aviso de TikTok Shop (S5b)
- Hoja blanca de 437 pt, radio 28, agarradera 40×5 #E6E9EF.
- "Tag your TikTok Shop product" ExtraBold 22 #0B1020; texto Medium 13 #6B7385.
- 3 pasos en cajas #F3F5F9 (radio 14) con número en círculo #1769FF de 24 pt, Bold 14:
  "We open TikTok with your board", "Tap "Add link" → "Products"", "Pick it from your Showcase and post".
- Botón azul 3D "Open TikTok" (abre TikTok con el board y el caption copiado).
- Pie Medium 12 centrado: "Using Amazon instead? Put your link in your bio — the caption already says "link in bio"."

## Workflow
- El diagrama "WORKFLOW · Shop board" (`3039:240`) se actualizó igual (sin "Paste a link", textos nuevos de Share).
