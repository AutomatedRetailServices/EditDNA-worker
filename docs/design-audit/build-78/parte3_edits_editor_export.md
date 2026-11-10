# Parte 3 — Edits · Editor · Export (build 78, commit 71ae403)

Auditoría solo de lectura. Código: `/home/claude/buildclon/mobile/ios/CutSell`. Figma `rXfUuJQgwZIZJTuONvU1Z7`, páginas 3003:6 "02 · Edits", 3003:28 "06 · Editor", 3003:25 "05 · Export". Comparé con get_screenshot los frames 1213:2, 1177:2, 1166:4, 1213:191, 1213:293, 1218:2, 1220:2, 1220:142, 1220:408, 1220:533, 1220:916, 1222:69, 1222:379, 1222:476, 1222:669, 1222:1080, 1222:1859, 1222:2781, 1222:3179, 1222:3563 y 1259:3101. De 1220:275, 1222:1463 y 1222:2321 solo leí los metadatos, sin captura. La página de tokens 3003:5863 no la abrí: usé los tokens del brief y los de `Theme.swift`.

## Notas para leer los colores

- La app siempre está en modo claro: `AppPrefs.colorScheme = .light`, aplicado con `.appColorScheme()` en MainTabView.
- `Color(hex:)` es un color dinámico. En claro, `LightPalette` lo cambia por su par claro:
  - 0x070910, 0x050814, 0x03050B, 0x060B18 → #FFFFFF
  - 0x040812, 0x0C1730, 0x03060F → #F8F9FB
  - 0x0A1326, 0x081226 → #F3F5F9
  - 0x19345F, 0x14336B, 0x1F4794 → #E6E9EF
  - 0xFFFFFF, 0xF7FAFF (`Color.ink`, `CS.textPrimaryF`) → #0B1020
  - 0xAFC4E8, 0xA3B8E0, 0x6F86AE, 0x8FA9D6 (`CS.textSecondaryF`, `CS.textSecondary`, `CS.textMutedF`, `CS.textMuted`) → #6B7385
  - `CS.cyan` / `CS.selection` → #1769FF
  - Los colores vivos no cambian: 0x7C3AED, 0xFF21D7, 0xFF4D67, 0xFF2E5E y 0x39FF78 se quedan igual.
- `Color(fixed:)` y `.white` no cambian nunca.
- Tokens WI que usa el código:
  - `CS.accent` = #1769FF y `CS.accentSoft` = #EEF3FF
  - `CS.wiInk` = #0B1020, `CS.wiSecondary` = #6B7385 y `CS.wiBorder` = #E6E9EF
  - `CS.wiPage` = #FFFFFF, `CS.wiCard` = #F8F9FB y `CS.wiSurface2` = #F3F5F9
  - `CS.wiError` = #E5484D, `CS.wiSuccess` = #1FA971, `CS.wiReady` = #33D98C, `CS.wiSuccessBg` = #EAF8F1 y `CS.wiErrorBg` = #FFF7F7
- Fuente: `CS.inter(...)` es en realidad **Plus Jakarta Sans** (Regular, Medium, SemiBold, Bold y ExtraBold). `.system(...)` es SF Pro.
- `softPuff(.white)` es un relleno 3D: degradado #FFFFFF→#EEF3FF, base #D3DDF5 y sombra #1769FF al 16 %. `softPuff(.blue)` es degradado #2F7BFF→#0A55F0, base #0636B8 y sombra #0A55F0 al 35 %.
- `wiCard()` por defecto (relleno #F8F9FB) usa ese `softPuff(.white)`. Si la tarjeta tiene otro relleno, usa ese relleno, borde de 1 pt y radio 20.

## Qué editor se abre de verdad

Edits → tocar una tile → `ProjectDetailView`, que decide según el estado del corte:
- estado "created": `UploadNotFinishedView`;
- procesando: `CutProgressView`;
- resto: `DraftEditorView`.

`DraftEditorView.body`, cuando el modelo carga, muestra siempre **`EditorV4View`** (EditorV4View.swift). Las piezas que usa:
- `TimelineV4` (EditorV4Timeline.swift);
- `ToolbarV4`;
- los paneles v5 (EditorV5Panels.swift);
- `CaptionStylePanel`, `StylePanel` y `OverlayGallerySheet` (EditorV4Panels.swift);
- `TextStyleSheet`;
- `ExportSettingsView`;
- `ExportSheet` (EditorV2Sheets.swift).

`EditorV5Model`, `EditorB17Model`, `TextLookModel` y `EditorMedia` son solo lógica que extiende `DraftEditorViewModel`.

La función `DraftEditorView.editor(_:)` (TimelineEditorView, CaptionsPanelView, VisualTimelineView, ReorderList, EditorExtrasView, botón "Export 1080p") **nunca se llama**: es código muerto.

---

## A. Pestaña Edits

### 1. Edits · Ready (grid) — `ProjectsView.swift` (`ProjectsView`)
- **Cómo se llega:** barra inferior → "Edits" ("Mis videos"). Abre en el filtro Ready.
- **Fondo:** claro. `ScreenBackground` = `HomeBackdrop`: degradado #E7EEFF→#F5F8FF→#FFFFFF más dos brillos (#1769FF al 16 % y #4A5BFF al 12 %, con blur 80). La franja de la barra de estado es #FFFFFF.
- **Elementos de arriba hacia abajo:**
  - Header (`WIHeader`): `BrandLogo` horizontal de 172 pt a la izquierda.
  - A la derecha, dos botones redondos de 40 pt (`softPuff` blanco) con iconos de 20 pt en #0B1020: asset `WISearch` y asset `WIMore`.
  - El menú de `WIMore` tiene "New cut" ("Nuevo corte") y "Refresh" ("Actualizar"). Es un menú del sistema.
  - Título "Edits" ("Mis videos"): #0B1020, Jakarta ExtraBold 32, tracking −0.64.
  - Subtítulo "Your saved cuts, ready to review or share." ("Tus cortes guardados, listos para revisar o compartir."): #6B7385, Medium 15.
  - Campo de búsqueda (solo después de tocar la lupa): `WISearch` de 18 pt #6B7385, placeholder "Search edits" ("Buscar cortes"), Medium 14, texto #0B1020, alto 44, `wiCard`.
  - Segmentado (`WISegmented`): alto 40, radio 20, `softPuff` blanco.
    - Segmentos: "Processing" (solo si hay algo procesándose) · "Ready" · "Products" · "Drafts" · "Posted" ("Procesando · Listos · Productos · Borradores · Publicado").
    - Texto SemiBold 13 en #0B1020. El contador va en Bold 11, #1769FF.
    - Segmento activo: cápsula `softPuff(.blue)` con texto #FFFFFF.
  - Enlace a la derecha "Select" / "Cancel" ("Seleccionar" / "Cancelar"): SemiBold 14, #1769FF.
  - Grid de 3 columnas (espacio 10/8). Cada tile (`EditTile`):
    - rectángulo 110:150, radio 16, fondo #F3F5F9 con la miniatura; borde 1 pt #E6E9EF solo cuando no hay miniatura;
    - punto de 8 pt arriba a la izquierda: #33D98C si está listo, #9AA3B5 si no, con borde blanco de 1,5;
    - botón "•••" de 26 pt arriba a la derecha: círculo #FFFFFF al 92 %, SF `ellipsis` 12 bold #0B1020;
    - si está procesando, `ProgressView` en el centro;
    - sin miniatura, el nombre del corte abajo: SemiBold 11, #0B1020.
  - Barra inferior (`CSTabBar`): cápsula #FFFFFF de 68 pt, borde #E6E9EF y sombra #0D1433 al 10 %.
    - Iconos `WITabCreate`, `WITabEdits`, `WITabPlan` y `WITabProfile` de 24 pt.
    - Pestaña activa: icono y texto #1769FF (Bold 11) sobre una burbuja #EEF3FF.
    - Pestañas inactivas: icono #0B1020, texto #6B7385 (Medium 11).
    - Textos: "Create · Edits · Plan · Profile".
- **Estados:**
  - Vacío: tarjeta `wiCard` con una placa #EEF3FF de 72×120 y SF `play.fill` en #1769FF.
    - Título (Bold 15, #0B1020) según el filtro: "No edits yet" ("Todavía no hay ediciones"), "No ready edits" ("No hay ediciones listas"), "No drafts" ("No hay borradores"), "Nothing posted yet" ("Aún no hay nada publicado") o "No matches" ("Sin resultados").
    - Texto (Medium 13, #6B7385): "Your finished and draft edits will appear here." ("Aquí aparecerán tus ediciones terminadas y tus borradores.") o "Try another name." ("Prueba otro nombre.").
  - Cargando: no hay pantalla de carga; se actualiza tirando hacia abajo (pull-to-refresh del sistema).
  - Error: alerta del sistema con el título "CutSell", el texto del error y "OK".
  - Subida interrumpida: fila arriba del grid (`wiCard`) con SF `arrow.clockwise.icloud` #1769FF.
    - Título (Bold 14): "Interrupted Cut" ("Corte interrumpido") o el nombre del corte.
    - Texto: "N clips • Resume upload" en Regular 12, #6B7385.
    - Chevron #6B7385.
- **Hojas/popups que abre:**
  - menú "•••" (ver nº 7);
  - `confirmationDialog` "Delete project?" con "Delete Project" (rojo) y "Cancel";
  - hoja `NewCutView` para reanudar una subida;
  - `AssignProductSheet` (ver nº 11).
- **Figma:** 🟠 distinta de "Edits · Ready" (1213:2):
  - la app tiene un 5.º segmento, "Posted", que Figma no tiene;
  - la app tiene el enlace "Select" encima del grid, que Figma no tiene;
  - las tiles de la app no muestran la duración "0:05" abajo a la derecha que sí está en Figma;
  - en la app la pestaña activa lleva una burbuja #EEF3FF; en Figma es solo icono azul más un punto debajo;
  - en la app el segmento activo es 3D; en Figma es plano #1769FF.

### 2. Edits · Processing — `ProjectsView.swift` (`processingList` y `QueueRowView` en ProcessingQueue.swift)
- **Cómo se llega:** Edits → segmento "Processing". Solo aparece si hay algo en curso. También se llega con "Go to Edits" desde Create o Bulk.
- **Fondo:** claro, `HomeBackdrop`.
- **Elementos:**
  - Título "Edits" y subtítulo "Cuts being edited now. You can leave the app." ("Cortes en edición ahora. Puedes salir de la app.").
  - Segmentado (igual que en nº 1).
  - Fila de sección: "In order" ("En orden") en Bold 17 #0B1020, y a la derecha "Each one moves to Ready when it’s done" ("Cada uno pasa a Listos al terminar") en Medium 12 #6B7385.
  - Si se llegó desde Plan con un grupo enfocado: "Showing this Bulk" / "Showing this Sales Mix" ("Mostrando estos videos" / "Mostrando este Sales Mix") en SemiBold 12 #6B7385, y "Show all" ("Ver todo") en #1769FF.
  - Si hay un Bulk o un Mix: `WINote` sin título con el texto "We'll notify you when your videos are ready. You can keep using CutSell." (SemiBold 13, #0B1020).
  - Cabecera del Bulk (`BulkGroupHeader`): "Bulk · N videos" (Bold 15) y "X of N ready" (SemiBold 12, #1769FF). Debajo, barra de 6 pt: fondo #E6EAF2 y relleno #1769FF.
  - Fila (`QueueRowView`), alto mínimo 84:
    - número "01" en Bold 11 #1769FF dentro de un círculo #EEF3FF de 28 pt;
    - miniatura de 46×60, radio 10;
    - nombre en Bold 14 #0B1020, en una línea, cortado con "…";
    - debajo, "0:33 · Processing 68%" (duración en Medium 12 #6B7385, estado en SemiBold 12 con el color del estado);
    - barra de progreso de 180×6: fondo #E6EAF2, relleno #1769FF.
  - A la derecha de la fila, según el estado:
    - procesando: anillo `ProgressRing` de 28 pt (#E6EAF2 / #1769FF);
    - en cola: asset `WIClock` de 24 pt en #6B7385;
    - listo: SF `checkmark.circle.fill` en #33D98C;
    - fallido: botón "Retry" ("Reintentar") con asset `WIRetry`, Bold 12, #E5484D, cápsula blanca de 32 pt con borde de 1,5.
  - Fila fallida: fondo #FFF7F7, borde #F6C9CB, y el número en #E5484D sobre #FDECEC.
  - Fila sin voz: "No voice found — use it as a product clip" ("No encontramos voz — úsalo como clip de producto") y debajo el botón con borde "Add to Sales Mix" ("Agregar a Sales Mix") en #1769FF.
  - Nota (`WINote`): asset `WINoteIcon` en una placa blanca de 36 pt. Título "You can close the app" ("Puedes cerrar la app") en Bold 14; texto "Editing keeps going on our servers. We’ll notify you as each cut is ready." en Medium 12 #6B7385.
- **Estados:**
  - textos de estado: "Queued", "Done", "Uploading N%", "Processing N%", "Processing" (en grupo), "Failed", "No voice", "Upload paused";
  - texto de espera larga: "Taking a little longer, we'll notify you" ("Está tardando un poco más, te avisamos");
  - error al reintentar: alerta "Couldn’t retry this cut. Please upload it again.".
- **Hojas:** ninguna. Tocar una fila abre `ProjectDetailView`.
- **Figma:** 🟠 casi igual a "Edits · Processing" (1177:2). Diferencias:
  - el segmento "Posted";
  - la cabecera del Bulk y la nota del grupo no están en Figma;
  - la burbuja de la barra inferior.

### 3. Edits · Drafts — `ProjectsView.swift`
- **Cómo se llega:** segmento "Drafts".
- **Fondo:** claro.
- **Elementos:** el mismo grid que en nº 1, filtrado. Las tiles llevan el punto gris #9AA3B5.
- **Estados:** vacío "No drafts".
- **Figma:** 🔴 no existe ningún frame de Drafts.

### 4. Edits · Posted — `ProjectsView.swift` (`postedList`)
- **Cómo se llega:** segmento "Posted".
- **Fondo:** claro.
- **Elementos:**
  - una sección por mes: nombre del mes en Bold 17 y "N posted" en Medium 12 #6B7385;
  - el mismo grid de tiles;
  - "•••" de 28 pt en un círculo #FFFFFF, con el menú "Not posted · back to Ready" ("No publicado · volver a Ready"; la traducción mezcla español con "Ready" en inglés) y SF `arrow.uturn.backward`.
- **Estados:** vacío "Nothing posted yet".
- **Figma:** 🔴 no existe.

### 5. Modo selección + "Add to Plan" — `ProjectsView.swift` (`selectableTile`, `addToPlanBar`)
- **Cómo se llega:** Ready → "Select".
- **Fondo:** claro.
- **Elementos:**
  - tiles listas: círculo de 24 pt (#FFFFFF al 90 %, borde #E6E9EF);
  - tile elegida: borde de 3 pt #1769FF y número blanco en Bold 12 dentro de un círculo #1769FF;
  - tiles que no están listas: opacidad 45 %, no se pueden tocar;
  - botón inferior "Add to Plan · N videos": SemiBold 16, #FFFFFF, cápsula de 50 pt `softPuff(.blue)`. Su traducción no está confirmada (la cadena se interpola).
- **Lleva a:** Plan › Plan my week, fuera de mi alcance.
- **Figma:** 🔴 no existe.

### 6. Barra de búsqueda de Edits
- Está descrita en nº 1.
- **Figma:** 🔴 en Figma solo existe la búsqueda de Products ("Search brand or product").

### 7. Menú "•••" de la tile + borrar — `ProjectsView.swift` (`EditTileMenu`)
- **Cómo se llega:** tocar "•••" en una tile.
- **Elementos (menú del sistema):**
  - "Personal · don't plan" o "Use in Plan": solo en cortes listos. **No se traducen**: el botón recibe un `String` y no pasa por `L10n.tr`.
  - "Delete" ("Eliminar"), en rojo.
  - Al borrar, diálogo "Delete project?" ("¿Eliminar proyecto?") con el texto "This permanently removes the project, its uploaded media, exports, overlays, timeline assets and feedback." y los botones "Delete Project" y "Cancel".
- **Figma:** 🔴 en Figma solo se dibuja el botón "•••", no el menú.

### 8. Toast "X is ready · View" — `ProcessingQueue.swift` (`ReadyToast`)
- **Cómo se llega:** aparece solo cuando un corte termina, en cualquier pestaña, a 84 pt del borde inferior. Dura 6 s.
- **Fondo:** tarjeta #FFFFFF de 46 pt, radio 14, borde #E6E9EF.
- **Elementos:**
  - SF `checkmark` 11 bold en #1FA971 dentro de un círculo #EAF8F1 de 22 pt;
  - "cutsell…849 is ready" en SemiBold 14 #0B1020;
  - "View" ("Ver") en Bold 14 #1769FF.
- **Figma:** ✅ igual a "Toast/NewReady" (1213:185) dentro de 1213:2.

### 9. Toast de grupo listo — `ProcessingQueue.swift` (`GroupReadyToast`)
- **Cómo se llega:** cuando termina un Bulk. Dura 8 s.
- **Elementos:** el mismo diseño que nº 8, con el texto "Your 10 videos are ready ✨" (o similar) y "View".
- **Figma:** 🔴 no existe.

### 10. Banner superior "Your cut is ready" — `MainTabView.swift` (`ReadyBanner`)
- **Cómo se llega:** un corte termina en segundo plano. También aparece cuando termina una exportación, con el texto "Open the cut in Edits to save or share it".
- **Fondo:** tarjeta #F8F9FB (`CS.surfaceCard` en claro), radio 18, borde #1769FF al 50 % y sombra #1769FF al 25 %.
- **Elementos:**
  - asset `ProcessingLogo` de 34 pt con `.blendMode(.screen)`: **probablemente casi invisible sobre fondo claro (no confirmado)**;
  - "Your cut is ready" ("Tu corte está listo") en SemiBold 15 #0B1020;
  - "Tap to review it" ("Toca para revisarlo") en Regular 12 #6B7385;
  - SF `xmark` 13 en #6B7385.
- **Abre:** `CutProgressView` a pantalla completa (Ready).
- **Figma:** 🔴 no existe en mis páginas.

### 11. Edits · Products / Detalle de producto / Asignar producto — `ProductsView.swift` (`ProductsList`, `ProductDetailView`, `AssignProductSheet`)
- **Cómo se llega:** segmento "Products" → tocar un producto. "Assign ›" abre la hoja de asignar.
- **Fondo:** claro (`HomeBackdrop`). La hoja es #FFFFFF.
- **Elementos de la lista:**
  - búsqueda "Search brand or product" (`wiCard`);
  - tarjeta "N videos need a product" + "We couldn’t tell from the video" + "Assign ›" (SemiBold 13, #FFB840);
  - nombre de la marca en Bold 17 y "N products · N videos" en #6B7385;
  - fila de producto (`wiCard`): portada, nombre en Bold 15, "12 videos", estados "7 ready" (#33D98C) · "3 drafts" (#6B7385) · "2 processing" (#1769FF), y el chevron `WIChevronSmall`.
- **Elementos del detalle:**
  - "‹ CeraVe" en SemiBold 16 #1769FF;
  - menú con "Brand & product" y "Select videos" / "Done selecting" (estos dos últimos sin traducir);
  - nombre del producto en ExtraBold 26;
  - "✎ Rename" en #1769FF;
  - subtítulo;
  - segmentado All/Ready/Drafts/Processing;
  - nota "✦  Grouped automatically from what you said in each video.";
  - grid de videos;
  - botones "Select" (blanco 3D), "Edit N drafts in bulk" (azul 3D) y "Add to Plan · N".
- **Elementos de la hoja de asignar:**
  - "Which product is this?" en ExtraBold 22;
  - "N videos selected";
  - tarjeta "✦ SUGGESTED FROM THE VIDEO" con el botón "Confirm" (azul);
  - "Or choose one of your products" (o "You have no products yet");
  - la lista de productos;
  - "+  New brand or product" en #1769FF;
  - "Move N videos" (azul, 3D);
  - alerta "New brand or product" con los campos "Brand (e.g. CeraVe)" y "Product (e.g. Hydrating Cleanser)" y los botones "Move here" y "Cancel".
- **Estados de la lista:**
  - "No products yet. New cuts are grouped by the brand and product you talk about.";
  - "Your products will appear here as your cuts are organized.".
- **Figma:**
  - lista: 🟠 casi igual a "Edits · Products" (1166:4); solo cambia el segmento "Posted";
  - detalle: 🟠 casi igual a "Edits · Product detail" (1213:191). En la app las acciones también están en un menú "•••", y el estilo exacto de las tiles no está confirmado;
  - hoja: ✅ igual a "Edits · Assign product" (1213:293).

### 12. Subida sin terminar — `ProjectsView.swift` (`UploadNotFinishedView`)
- **Cómo se llega:** tocar un corte en estado "created".
- **Fondo:** claro (`HomeBackdrop`).
- **Elementos:**
  - "!" en Bold 46, color `CS.alert` = #FF2E5E. Es el rojo viejo, no el #E5484D de WI;
  - "This video didn’t finish uploading" ("Este video no terminó de subir") en Bold 22 #0B1020;
  - un texto según el caso, en Regular 15 #6B7385: "Your video is saved on this iPhone. Resume the upload and CutSell will make the cut." o "The video isn’t on this iPhone anymore. Create a new cut from Create.";
  - botón "Resume upload" ("Reanudar subida"): `PrimaryButton` de 54 pt y radio 16.
- **Abre:** hoja `NewCutView`.
- **Figma:** 🔴 no existe. La pantalla es clara pero usa el rojo viejo #FF2E5E.

### 13. Abriendo el corte / error al abrir — `DraftEditorView.swift` (`DraftEditorView`)
- **Cómo se llega:** tocar un corte listo.
- **Fondo:** claro, `EditorBackdrop`: `HomeBackdrop` más brillos #FFB3D9, #B9C8FF y #D9C2FF.
- **Elementos:**
  - botón atrás `WIHeaderIcon("WIBack")` arriba a la izquierda;
  - cargando: `BrandRing` de 96 pt y "Opening your cut…" ("Abriendo tu corte…") en SemiBold 15 #6B7385;
  - error: "Couldn’t open this cut" ("No se pudo abrir este corte") en SemiBold 18 #0B1020, el motivo en Regular 13 #6B7385, y `PrimaryButton` "Try again" de 48 pt y 200 de ancho;
  - pastilla "Restoring video… 40%": ver los problemas de nº 31.
- **Figma:** 🔴 no existe.

---

## B. Editor (`EditorV4View`)

### 14. Editor · Base — `EditorV4View.swift` (`EditorV4View`)
- **Cómo se llega:** Edits → corte listo, o Ready → "Review & edit".
- **Fondo:** claro (`EditorBackdrop` con brillos rosa, lila y azul). El marco del video es oscuro por naturaleza.
- **Elementos de arriba hacia abajo:**
  - **Header** (48 pt):
    - atrás (`EditorBackButton`): SF `chevron.left` 12 bold #1769FF en un círculo 3D blanco de 32 pt;
    - nombre "Cut 13 oct ✎" en SemiBold 14 #0B1020, ancho máximo 124. Al tocarlo abre "Rename cut";
    - `ExportQualityMenu`: "1080p HD" o "4K Ultra HD" en Bold 13 #1769FF con el asset `Ed4Chevron`, cápsula 3D blanca de 108×34;
    - "Export" ("Exportar") en SemiBold 14 #FFFFFF, cápsula azul 3D de 78×34.
  - **Video:** tarjeta de 266×472 como máximo, radio 20, borde de 2 pt con degradado #1769FF→#7C3AED→#FF21D7.
    - Fondo de la tarjeta `Color(hex:0x070B16)`, que en claro sale casi blanco (≈#F4F5F8, calculado).
    - Mientras carga se ve la primera imagen guardada y un `BrandRing` de 64 pt.
    - Si hay un mensaje del reproductor, se muestra en Regular 12 #6B7385.
  - **Fila de controles** (24 pt):
    - `Ed4CtlExpand` (pantalla completa);
    - `Ed4CtlPlay` en el centro (en pausa, dos barras #1769FF);
    - `Ed4CtlKeyframe` ◇ (o `ZoomDiamond` relleno);
    - flechas ‹ › (SF `chevron.left` / `chevron.right`, 15 semibold, #1769FF) solo si hay keyframes;
    - `Ed4CtlUndo` y `Ed4CtlRedo`.
  - **Timeline** (`TimelineV4`), alto 162:
    - fondo #E9ECF1, bandas #DDE1E8, línea superior #E6E9EF;
    - hora "00:03" en SemiBold 12 #4A5263 y "/ 00:54" en Medium 12 #6B7385;
    - regla "00:00 • 00:02" en Medium 10 #6B7385;
    - iconos de fila: cuadrados blancos de 32 pt, radio 8, borde #C9CFD9, con los assets `Ed4RowSpeaker`, `Ed4RowMic` y `Ed4RowLayers` y SF `music.note` en #0B1020. Si el sonido del video está apagado, el altavoz lleva una raya roja #FF4D67;
    - clips: miniaturas, radio 7, borde blanco de 1 pt (seleccionado: #1769FF de 3 pt con asas #C7CFDB). Encima, el texto del subtítulo en SF 9 semibold blanco sobre negro al 45 %; abajo, barras de audio #1769FF;
    - etiqueta de velocidad: SF 9 bold blanco sobre #1769FF;
    - voz en off y overlays: bloques #1769FF con texto blanco en SemiBold 10 (seleccionado: bloque blanco con texto azul);
    - "+" en un cuadrado blanco de 24 pt con borde #C9CFD9;
    - cursor fijo de 2 pt en #0B1020.
  - **Barra de herramientas** (`ToolbarV4`): fondo #F3F5F9 al 85 % con línea #E6E9EF.
    - Fila principal: "Edit" (SF `scissors`) · "Audio" (SF `music.note`) · "Text" (SF `textformat`) · "Captions" (`Ed4Captions`) · "Overlay" (`Ed4Overlay`) · "Style" (`Ed4Style`). En español: "Editar · Audio · Texto · Subtítulos · Superponer · Estilo".
    - Cada icono va en una burbuja 3D de 38 pt y radio 13. El icono es #0B1020; la herramienta activa es azul con el icono blanco.
    - Etiquetas en Medium 10,5, #6B7385.
  - Fondo de la zona inferior: #E9ECF1 al 72 % con el timeline, #FFFFFF al 70 % con un panel.
- **Estados:**
  - aviso "Almost no voice in this video. Check the cut." ("Casi no se oye voz en este video. Revisa el corte.");
  - error: alerta "CutSell" + "OK".
- **Hojas:** Rename (nº 16), menú de calidad (nº 17), pantalla completa (nº 18), Export (nº 35), galería (nº 27), panel de voz (nº 29–30), `TextStyleSheet` (nº 21–22), paneles (nº 23–34).
- **Figma:** 🟠 distinta de "1 Base" (1218:2):
  - **Barra de herramientas:** Figma tiene la barra plana de v4 "Split · Delete · Mirror · Rotate | Overlay · Voice Over · Audio · V…", sin burbujas. La app tiene una fila principal nueva (Edit · Audio · Text · Captions · Overlay · Style) con burbujas 3D y subfilas.
  - **Botón atrás:** en Figma es un cuadrado redondeado gris claro; en la app es un círculo 3D blanco.
  - **Menú de calidad:** en Figma es una cápsula con relleno azul claro; en la app es una cápsula 3D blanca.
  - **Fondo:** en Figma es blanco; en la app lleva brillos rosa y lila.
  - El timeline y los controles coinciden.

### 15. Subfilas "Edit" y "Audio" de la barra — `EditorV4View.swift` (`ToolbarV4`, `ToolMenu`)
- **Cómo se llega:** "Edit", o seleccionar un clip de video. "Audio" abre la otra.
- **Elementos:**
  - botón atrás: SF `chevron.left` 17 bold blanco en un círculo azul 3D de 44 pt;
  - Edit: "Split" (`Ed4Split`, **sin traducción al español**) · "Delete" (`Ed4Delete`) · "Duplicate" (SF `plus.square.on.square`) · "Mirror" (`Ed4Mirror`) · "Rotate" (`Ed4Rotate`);
  - Audio: "Voice Over" (`Ed4Voice`) · "Music" (`Ed5Audio`) · "Volume" (`Ed5Volume`) · "Speed" (`Ed5Speed`).
- **Estados (toasts):**
  - "Clip deleted", "Clip duplicated", "Clip rotated 90°", "Video mirrored", "Mirror off";
  - "Move the cursor inside a clip to split it.", "This clip can’t be split here. Move the cursor between two words.";
  - "Video split at 3.4", "Select a clip first.", "Select a clip, voice-over, overlay or caption first.".
- **Figma:** 🔴 no existen las subfilas, el botón atrás azul ni "Duplicate".

### 16. Alerta "Rename cut" — `EditorV4View.swift`
- **Elementos:** alerta del sistema "Rename cut" ("Cambiar nombre del corte") con el campo "Name" y los botones "Save" y "Cancel". Si falla: "Couldn’t rename the cut. Please try again.".
- **Figma:** 🔴 no existe.

### 17. Menú de calidad — `EditorV4View.swift` (`ExportQualityMenu`)
- **Elementos:** menú del sistema con "4K Ultra HD" (desactivado si el original no es 4K), el texto "Your video was recorded in 1080p" y "1080p HD". La opción elegida lleva un checkmark. "4K Ultra HD" y "1080p HD" no tienen traducción.
- **Figma:** 🔴 solo existe el botón cerrado.

### 18. Pantalla completa — `EditorV4View.swift` (`fullScreenPlayer`)
- **Fondo:** negro, el `VideoPlayer` del sistema. Es oscuro por naturaleza.
- **Elementos:** SF `xmark` 17 bold blanco en un círculo negro al 40 % de 44 pt.
- **Figma:** 🔴 no existe.

### 19. Keyframes ◇, barra "Curve" y etiqueta de zoom — `EditorV4View.swift`
- **Elementos:**
  - barra de curva: "Curve" ("Curva") en SF 11 semibold #6B7385 y opciones en cápsulas de 24 pt (activa #1769FF con texto blanco; inactiva blanco al 8 % con texto #0B1020). Fondo: cápsula #F3F5F9 al 92 %;
  - etiqueta "◇ Zoom 4%": SF 10 blanco sobre `Color(hex:0x040812, .85)`, que en claro es **#F8F9FB: texto blanco sobre casi blanco, probablemente ilegible**.
- **Toasts:** "No keyframe after this point" / "No keyframe before this point", "Zoom deleted".
- **Figma:** 🔴 en Figma solo está el icono ◇ de la base.

### 20. Escribir el subtítulo sobre el video + panel de subtítulos 2b — `EditorV4View.swift` (`captionEditor`) y `EditorV4Panels.swift` (`CaptionStylePanel`)
- **Cómo se llega:** tocar el subtítulo en el video. Es el **único** camino a este panel; el botón "Captions" abre otra cosa (nº 21).
- **Fondo:** claro (`CS.wiPage` / zona #FFFFFF al 70 %).
- **Elementos sobre el video:**
  - pastilla "Type here · drag · pinch" ("Escribe aquí · arrastra · pellizca"): SemiBold 11, blanca, sobre `Color(hex:0x050814,.92)` con borde 0x2A4A80. En claro el fondo sale **≈#FFFFFF y el texto blanco: probablemente ilegible**; en Figma es oscura;
  - caja punteada blanca de 1,5 pt sobre negro al 55 %, con cuatro asas blancas de 10 pt;
  - el texto va en la fuente del subtítulo a 21, y la palabra resaltada en #39FF78.
- **Elementos del panel:**
  - "Captions" en SemiBold 15 #0B1020;
  - "On" ("Activado") en Regular 12 #6B7385 con un interruptor #1769FF;
  - "Done" ("Listo") en SemiBold 14 #0B1020: cápsula #EEF3FF de 62×28 con borde #1769FF;
  - cuatro tarjetas de estilo de 54 pt (#F3F5F9, radio 12; la elegida con borde #1769FF de 2 pt):
    - "Aa" "Classic" en #0B1020;
    - "Highlight" en #39FF78;
    - "Box": negro sobre blanco;
    - "Yellow" en #FFD60A;
    - las etiquetas van en Regular 10 #6B7385;
  - fila deslizable de fuentes: cada nombre en su propia fuente a 14, en chips #F3F5F9 de 34 pt;
  - con Highlight: "Word color" ("Color de las palabras") y círculos #39FF78, #FF453A y #1AA6FF;
  - con Box: "Black box, white words" y "White box, black words";
  - "Drag to move, pinch to resize, tap to fix words." en Regular 10 #6B7385.
- **Figma:** 🟠 distinta de "2 Caption escribir" (1220:2), "3 Captions Highlight" (1220:142), "4 Captions Box" (1220:275) y "5 Captions Fuentes" (1220:408):
  - el panel coincide;
  - la pastilla de ayuda es oscura en Figma y en la app probablemente sale blanca;
  - la barra de herramientas es la vieja en Figma (ver nº 14).

### 21. "Captions" → hoja Fuentes / Estilo — `TextStyleSheet.swift` (`TextStyleSheet`, target `.captions`)
- **Cómo se llega:** barra → "Captions".
- **Fondo:** hoja #FFFFFF, radio 28 arriba, sombra #0B1020 al 12 %. Alto 452 o 74 % de la pantalla; se puede arrastrar.
- **Elementos:**
  - asa #E6E9EF de 40×5;
  - "Captions" en ExtraBold 18;
  - "On" en SemiBold 13 #6B7385 con interruptor;
  - ✓: SF `checkmark` 18 bold blanco en un círculo azul 3D de 48 pt;
  - pestañas "Fonts" / "Style" ("Fuentes" / "Estilo"): Bold 16 #0B1020 con subrayado de 3 pt #1769FF; la inactiva en SemiBold #6B7385;
  - Fonts: grid de 3 columnas con chips 3D de 52 pt; cada nombre en su fuente a 19 (el elegido en azul con texto blanco);
  - Style:
    - chips "Aa" de 58 pt (degradado #8E97AA→#5F687C; el elegido con borde #1769FF de 3 pt);
    - pestañas "Text · Highlight · Stroke · Background · Shadow · Spacing";
    - fila de colores: "ninguno" (raya #FF3B30), `ColorPicker` y muestras de 38 pt;
    - sliders `LookSlider` (título en SemiBold 14, valor en SemiBold 13 #6B7385, tint #1769FF): Size, Opacity, Width, Corners, Blur, Distance, Letters, Lines;
    - alineación (SF `text.alignleft` / `text.aligncenter` / `text.alignright`) y "AA" para mayúsculas;
    - con Highlight: "The word being said changes colour.".
- **Figma:** 🔴 no existe. En Figma, Captions es el panel 2b de nº 20.

### 22. "Text" → texto nuevo o editar texto — `TextStyleSheet.swift` (target `.newText` / `.text(id)`) y `TextOverlayOnVideo`
- **Cómo se llega:** barra → "Text", o tocar dos veces un texto en el video.
- **Elementos:**
  - campo "Enter text" ("Escribe el texto") en SemiBold 16, cápsula 3D blanca de 50 pt;
  - ✓ azul;
  - con el teclado abierto la hoja se encoge a 92 pt;
  - el resto es igual que nº 21, sin la pestaña "Highlight".
  - En el video, el texto seleccionado lleva una caja punteada (detalle no confirmado).
- **Figma:** 🔴 no existe.

### 23. "Style" → Auto Zoom AI — `EditorV4Panels.swift` (`StylePanel`)
- **Cómo se llega:** barra → "Style".
- **Fondo:** claro.
- **Elementos:**
  - "Style" ("Estilo") en SemiBold 15;
  - "Done" (cápsula #EEF3FF con borde azul);
  - fila (`StyleToggleRow`, 66 pt, `wiCard` radio 16): "Auto Zoom AI" ("Auto Zoom IA") en Bold 15, "Zooms in on key moments automatically." en Medium 12 #6B7385, e interruptor `CSSwitch`.
- **Figma:** 🔴 no existe en 06 · Editor.

### 24. Panel "Add overlay" — `EditorV5Panels.swift` (`OverlayAddPanelV5`, dentro de `PanelCardV5`)
- **Cómo se llega:** barra → "Overlay", o el icono de capas del timeline.
- **Fondo:** tarjeta #FFFFFF, radio 20, borde #E6E9EF (0x19345F en claro).
- **Elementos:**
  - "Add overlay" ("Añadir superposición") en **SF** 17 semibold #0B1020;
  - "From your Photos" ("De tus Fotos") en SF 14 #6B7385;
  - X con el asset `Ed5CloseX` en #0B1020;
  - dos botones de 52 pt, "Photo" (`Ed5Photo`) y "Video" (`Ed5Video`), en SF 16 semibold sobre #F3F5F9 con radio 14.
- **Figma:** 🟠 distinta de "12 Overlay Add" (1222:1080):
  - en Figma los botones no tienen caja gris (solo icono y texto azules);
  - la app usa SF en lugar de Jakarta.

### 25. Galería (foto o video) — `EditorV4Panels.swift` (`OverlayGallerySheet` + `SystemMediaPicker`)
- **Cómo se llega:** Photo o Video en nº 24. El "+" del video abre la misma galería con el título interno "Add a video".
- **Fondo:** #FFFFFF.
- **Elementos:**
  - el **selector de fotos del sistema** (PHPicker). Los títulos "Photos" / "Videos" / "Add a video" solo se ven en el modo de capturas;
  - tarjeta de progreso: "Getting it ready…" o "Getting it from iCloud… 40%" en Medium 14, con una barra #1769FF;
  - tarjeta de problema: "This video's format isn't supported. Please pick another one." o "It couldn't be copied from your library. Check your connection (iCloud) and try again.", más "OK" en #1769FF.
- **Figma:** 🟠 distinta de "6 Overlay Galería" (1220:533): Figma dibuja una hoja propia ("Cancel" · "Photos & Videos" · grid gris) y la app usa el selector de Apple.

### 26. Overlay colocado / seleccionado — `EditorV5Panels.swift` (`OverlayPanelV5`) y `MediaOverlayOnVideo`
- **Cómo se llega:** elegir una foto o un video, o tocar un overlay.
- **Fondo:** tarjeta #FFFFFF, radio 22, borde #E6E9EF.
- **Elementos:**
  - asset `Ed5Film` #1769FF en una placa #EEF3FF de 30 pt;
  - nombre "Video 2" en SF 16 semibold;
  - "Overlay · 0:04–0:12" en SF 11,5 #6B7385;
  - botones redondos de 36 pt sobre círculo #F3F5F9 (activo con anillo #1769FF): "Sound"/"Muted" (`Ed5Muted` / SF `speaker.wave.2.fill`), "Fill" (`Ed5Fill`) y "Front" (`Ed5Front`); etiquetas en SF 10,5;
  - ✓ con `Ed5CloseCheck` en #1769FF;
  - línea divisoria;
  - "Layout" ("Diseño") y tarjetas de mini-teléfono (gris #D5DAE3 y azul #1769FF): Float · Side by side · Stacked · Left · Right · Top · Bottom · Full;
  - "Edge" ("Borde") con un segmentado "Sharp · Soft · Extra soft" (activo en cápsula #1769FF, fondo #F3F5F9);
  - sobre el video: pastilla "Drag · pinch · turn with two fingers" (mismo problema de contraste que nº 20).
- **Toasts:** "X added", "X added on a new layer · tap ↶ to undo", "Overlay deleted", "Overlay split at…", "Wait until it finishes uploading, then split it.".
- **Figma:** 🟠 distinta de "7 Overlay colocado" (1220:916), "14 Overlay Float" (1222:1859), "15 Overlay Stacked" (1222:2321) y "13 Overlay filas" (1222:1463):
  - en Figma los iconos Muted / Fill / Front no tienen círculo gris;
  - la pastilla de Figma dice "Drag to move · pinch to resize" en oscuro;
  - el resto (layouts, Edge) coincide.

### 27. Panel Volume — `EditorV5Panels.swift` (`VolumePanelV5`, `CleanUpRowV5`)
- **Cómo se llega:** Audio → "Volume".
- **Fondo:** tarjeta #FFFFFF.
- **Elementos:**
  - "Volume" ("Volumen") en SF 17;
  - subtítulo "Whole video" ("Todo el video") o "Part 0:04–0:09";
  - pastilla "Mute" / "Unmute" (SF `speaker.wave.2` / `speaker.slash.fill`), alta 36, #F3F5F9 (activa en azul);
  - ✓;
  - slider (pista #D5DAE3, relleno #1769FF, perilla blanca de 24 pt con borde azul) de 0 a 200 %;
  - porcentaje "100%" en SF 15 semibold #0B1020 ("Muted" si está silenciado);
  - ayuda en SF 12 #6B7385: "Tap a part of the video on the timeline to change only that part." o "Only this part changes. Tap it again to change the whole video.";
  - "Clean up voice" ("Limpiar voz") como `Toggle` del sistema (tint #1769FF) con "Removes background noise from the voice";
  - según el estado: "Cleaning… 40%", "Playing the clean voice" o "Couldn't clean the voice. Tap to try again.".
- **Figma:** 🟠 distinta de "16 Volumen" (1222:2781):
  - Figma tiene una placa con icono azul junto al título, "Mute" como icono con etiqueta, "Clean up" como botón redondo, la etiqueta "Level", las marcas 0/100/200 y la nota "Clean up removes background noise · applies on export";
  - la app usa una pastilla, un interruptor abajo y otro texto de ayuda.

### 28. Panel Speed — `EditorV5Panels.swift` (`SpeedPanelV5`)
- **Cómo se llega:** Audio → "Speed".
- **Elementos:**
  - "Speed" ("Velocidad") en SF 17;
  - "Selected clip" o "Select a clip";
  - cinco botones de 42 pt: "0.5×", "1×", "1.5×", "2×", "3×" en SF 15 (activo #1769FF con texto blanco; resto #F3F5F9 con radio 12).
- **Figma:** 🟠 distinta de "17 Velocidad" (1222:3179):
  - Figma tiene la placa con icono y el subtítulo "Selected clip · 0:08→0:04";
  - en Figma las opciones son solo texto y la elegida va en una cápsula azul; en la app son cajas grises.

### 29. Panel "CutSell sounds" (Music) — `EditorV5Panels.swift` (`AudioPanelV5`)
- **Cómo se llega:** Audio → "Music".
- **Elementos:**
  - "CutSell sounds" en SF 17;
  - "Tap one to hear it";
  - ✓;
  - grid de 3 columnas con "Tap", "Good", "Wrong", "Glass" y "Typing" en SF 16 sobre #F3F5F9 (radio 14, borde #1769FF si está elegido, spinner si se está descargando);
  - "+ Add at playhead" ("Añadir en la línea") en SF 15 blanco sobre #1769FF con radio 12; mientras descarga, "Getting sound…";
  - "Music from Files" ("Música de Archivos") con SF `music.note` sobre #F3F5F9;
  - error: "Sounds couldn't download. Tap to try again.".
- **Figma:** 🟠 distinta de "18 Audio" (1222:3563):
  - Figma tiene una placa con icono, un icono por sonido y "My music" como icono de carpeta;
  - en Figma el botón es de ancho completo: "Add Typing at playhead".

### 30. Aviso "Music from Files" + selector de archivos — `EditorV4View.swift`
- **Elementos:** alerta "Music from Files" con "Use only music you own or have rights to." ("Usa solo música tuya o con derechos.") y los botones "OK" y "Cancel"; luego el `fileImporter` del sistema.
- **Estados:** "Music added · tap ↶ to undo" o "That song couldn't be read. Please pick another file.".
- **Figma:** 🔴 no existe.

### 31. Panel de sonido seleccionado — `EditorV5Panels.swift` (`SoundClipPanelV5`)
- **Cómo se llega:** tocar un sonido o una música en el timeline.
- **Elementos:**
  - nombre, "Music" o "CutSell sound";
  - slider de 0 a 200 %;
  - pastillas "Lower video sound" (solo música), "Mute"/"Unmute" y "Delete" (#FF4D67).
- **Figma:** 🔴 no existe.

### 32. Permiso de micrófono — `EditorV4View.swift` (`micPermissionCard`)
- **Cómo se llega:** "Voice Over" la primera vez.
- **Fondo:** cortina negra al 55 % y tarjeta #FFFFFF con radio 24 y borde #E6E9EF.
- **Elementos:**
  - SF `mic` 40 en #1769FF;
  - "Allow microphone access" ("Permitir acceso al micrófono") en Bold 21 #0B1020;
  - "CutSell.ai needs microphone access to record your voiceover." en Regular 14 #6B7385;
  - `PrimaryButton` "Allow microphone" de 52 pt y radio 14;
  - "You can change this later in iOS Settings." en Regular 12 #6B7385.
- **Estados:** si se niega, toast "Allow the microphone in iOS Settings to record.".
- **Figma:** 🟠 distinta de "8 Permiso micrófono" (1222:69):
  - en Figma el micrófono es rosa/magenta en degradado; en la app es SF azul;
  - en Figma la cortina es blanca translúcida sobre el editor; en la app es negra al 55 %.

### 33. Voz en off · grabar (con 3·2·1) — `EditorV5Panels.swift` (`VoiceOverSheetV5` paso record, `CountdownOverlayV5`)
- **Cómo se llega:** Audio → "Voice Over", o el icono del micrófono en el timeline.
- **Fondo:** hoja #FFFFFF, radio 26, borde #E6E9EF.
- **Elementos:**
  - asa #D5D9E2;
  - SF `mic` #1769FF y "Recording voice-over" ("Grabando voz en off") en Bold 18;
  - "Add your voice to this video" o "● REC · 0:07" / "Paused · 0:07" en SemiBold 13 #6B7385;
  - tarjeta #F8F9FB: "Enhance voice" ("Mejorar voz") en SemiBold 15, "Removes background noise and sounds, so only your voice is heard." en Medium 12 #6B7385, e interruptor;
  - onda en vivo (`LiveWave`);
  - pausa: círculo #0B1020 de 48 pt con barras blancas;
  - micrófono grande: asset `Ed4VoiceMicBig` de 84 pt con anillo tornasol (#19E6FF→#1769FF→#7C3AED→#FF21D7) y, grabando, un anillo #FF3861;
  - terminar: círculo #0B1020 con un cuadrado #FF3861;
  - etiquetas "Tap to pause", "Tap / hold to record" y "Tap to finish" en Medium 10 #6B7385;
  - sobre el video: cuenta atrás con negro al 35 %, anillo tornasol de 128 pt, el número en SF 64 bold blanco, "Get ready…" ("Prepárate…") y "Recording starts in 3 seconds" en #6B7385.
- **Estados:** "Allow the microphone in Settings to record." en #FF4D67.
- **Popups:** alerta "Discard recording?" con "Your voice-over hasn’t been saved." y los botones "Discard" y "Keep recording".
- **Figma:** ✅ igual a "9 Voice-over grabar" (1222:379).

### 34. Voz en off · escuchar y guardar — `EditorV5Panels.swift` (`VoiceOverSheetV5` paso review)
- **Elementos:**
  - "Voice-over" en Bold 18;
  - "Preview your recording before saving." en Medium 13 #6B7385;
  - onda de 42 picos (oído #1769FF, por oír #C9CFD9) en una caja #F8F9FB con borde #E6E9EF;
  - tiempos "00:00" y "00:08";
  - "Play"/"Pause" y "Retake" ("Repetir"): blancos con borde #0B1020 de 1,5 pt y radio 14;
  - "Save" / "Saving…" ("Guardar"): azul 3D;
  - al guardar: toast "Voiceover 1 added · tap ↶ to undo" o "Voiceover recorded again · tap ↶ to undo".
- **Figma:** ✅ igual a "10 Voice-over guardar" (1222:476).

### 35. Voz en off seleccionada — `EditorV5Panels.swift` (`VoiceClipPanelV5`)
- **Cómo se llega:** tocar un bloque de voz.
- **Fondo:** #FFFFFF, radio 22, borde #E6E9EF.
- **Elementos:**
  - placa #EEF3FF con SF `mic` #1769FF;
  - "Voiceover 1" en SF 16;
  - "0:03 – 0:11" en SF 11,5 #6B7385;
  - botones "Lower video" (`VoPanelDuck`; "Lower"/"video" sin traducción), "Clean up" (`VoPanelSparkle`) y "Retake" (`VoPanelRetake`), con anillo #1769FF cuando están activos;
  - ✓ con `VoPanelCheck`;
  - "Voice" con slider de 0/100/200 y la perilla `VoPanelKnob`;
  - "100%" en una caja #F3F5F9;
  - "Delete voiceover" ("Eliminar voz en off") en #FF4D67 sobre #FF4D67 al 14 %.
- **Figma:** ✅ igual a "11 Voice-over seleccionado" (1222:669).

### 36. Toast con "Undo" y pastillas sobre el video — `EditorV4View.swift` (`ToastPillV4`, `soundDownloadPill`, `brollPill`) y `DraftEditorView.swift` (`RestoreBanner`)
- **Elementos:**
  - toast: Medium 13 **blanco** con "Undo" en #1769FF, fondo `Color(hex:0x040812,.97)` (= #F8F9FB en claro) y borde #7C3AED;
  - "Getting sounds…" / "A sound didn't download" + "Try again";
  - "Adding video… 45%" / "The video wasn't added" + "Try again";
  - "Restoring video… 40%";
  - todas usan texto blanco sobre `Color(hex:0x040812)`, que en claro es casi blanco.
- **Problema:** **texto blanco sobre #F8F9FB, probablemente ilegible.** No lo confirmé en el dispositivo. En Figma, nº 11 (1222:669), el toast es una pastilla blanca con texto oscuro.
- **Figma:** 🟠 en Figma el toast tiene texto oscuro y no lleva el borde violeta.

### 37. Arrastrar el borde de un clip (recorte) — `TrimEdges.swift` (`TrimFrameOverlay`)
- **Elementos:** encima del video se ve el cuadro exacto del original y "+0,4 s". Detalle visual no confirmado (no leí la vista entera).
- **Figma:** 🔴 no existe.

---

## C. Exportar, guardar y publicar

### 38. Ajustes de exportación — `ExportSettingsView.swift` (`ExportSettingsView`)
- **Cómo se llega:** editor → "Export". Se abre a pantalla completa.
- **Fondo:** claro, #E9ECF1. La hoja es #FFFFFF con radio 28 arriba.
- **Elementos:**
  - atrás `WIHeaderIcon("WIBack")`;
  - miniatura de 118×210, radio 18, borde blanco de 3 pt y sombra #1769FF al 28 %;
  - asa #D5D9E2;
  - "Export" ("Exportar") en Bold 28 #0B1020;
  - "9:16 · Final cut 1:26" en Regular 14 #6B7385;
  - tarjeta #EEF3FF: "✦ Original quality detected" / "✦ 4K export selected" / "✦ 1080p copy (original not found)" en Bold 15 #1769FF, y "1080 × 1920 · 30 fps · H.264" en Regular 12 #6B7385;
  - "Resolution": segmentado de 44 pt (`softPuff` blanco) con "Match · 720p · 1080p · 1440p · 4K". El elegido va en #1769FF con texto blanco; los no disponibles en #B5BBC7;
  - ayuda en Regular 11 #6B7385: "Match keeps the detected source resolution." / "Your video was recorded in 1080p." / "4K from your original, with its own detail." / "Your original isn't on this iPhone: exporting from the 1080p copy.";
  - "Frame rate": "Match · 24 · 30 · 60";
  - "Quality": "Auto · High · Maximum";
  - "Estimated file size" y "≈ 45 MB" en Bold 17;
  - "WHERE TO?" ("¿A DÓNDE?") en Bold 12 #1769FF;
  - "Save to Photos": fila blanca de 72 pt con borde #E6E9EF, SF `square.and.arrow.down` #1769FF sobre #EEF3FF y el texto "Export and keep it on this iPhone";
  - "Add to Plan to post": fila azul #1769FF con SF `calendar` blanco y "TikTok · Instagram · Facebook";
  - "Cancel" en Medium 15 #6B7385.
- **Estados:**
  - sin espacio: "Not enough space on your iPhone. Free up 1.2 GB" en #FF8A8A (rojo viejo), con "Open Settings" y "Export on our server (1080p)";
  - error dentro de la hoja: texto en #FF8A8A y "Export on our server (1080p)".
- **Figma:** 🟠 distinta de "Export · elegir destino" (1259:3101):
  - en Figma la hoja sale sobre la pantalla Ready atenuada; en la app sale sobre un gris #E9ECF1 con miniatura y botón atrás;
  - la app tiene la fila **"Frame rate"**, que Figma no tiene;
  - la resolución de la app tiene **"1440p"** (5 opciones; Figma tiene 4);
  - la app tiene la línea de ayuda bajo la resolución.

### 39. Exportando (progreso) — `ExportSettingsView.swift` (`progress`)
- **Elementos:**
  - `BrandRing` de 72 pt;
  - "Getting your original from iCloud…" / "Exporting…" / "Exporting 4K…" / "Saving to Photos…" en Regular 14 #6B7385;
  - "45%" en SemiBold 18 #0B1020;
  - barra de 8 pt (#F3F5F9 / #1769FF);
  - "Keep CutSell open until it finishes.";
  - "Cancel".
- **Figma:** 🔴 no existe en mis páginas.

### 40. "Saved to Photos" — `ExportSettingsView.swift` (`done`)
- **Elementos:**
  - SF `checkmark` 40 en #1FA971 dentro de un círculo #EAF8F1 de 96 pt con borde verde;
  - "Saved to Photos" ("Guardado en Fotos") en Bold 26;
  - "1080 × 1920 · 1:26";
  - "Your original wasn't on this iPhone, so this export used the 1080p copy.";
  - "Share" ("Compartir"): azul 3D de 52 pt;
  - "Done": blanco con borde azul.
- **Abre:** la hoja de compartir del sistema.
- **Figma:** 🔴 no existe en mis páginas. El código cita 1241:3561, que no está en mis páginas.

### 41. "Add to Plan to post" — `ExportSettingsView.swift`
- **Elementos:** "Opening Week Plan…" ("Abriendo Plan de la semana…") en Medium 14 #6B7385; después pasa a Plan › Plan my week (fuera de mi alcance).
- **Figma:** 🔴 no existe.

### 42. Exportación fallida — `ExportSettingsView.swift` (`ExportFailedView`)
- **Fondo:** claro (`HomeBackdrop`).
- **Elementos:**
  - `WIBackBar` con "Export failed" ("La exportación falló");
  - `BrandRing` quieto de 110 pt con "!" blanco sobre #E5484D;
  - "62%" en ExtraBold 40;
  - "Export stopped";
  - "Your video was not saved" en #1769FF;
  - pasos "Preparing video · Rendering frames · Encoding video · Saving to Photos": hecho con ✓ #33D98C, el que falló con "!" #E5484D sobre #FDEBEC y la palabra "Failed" en #1769FF, el resto con "—";
  - tarjeta "Nothing was lost" + "Your cut and export settings are still saved.";
  - "Try Again" (azul 3D);
  - "Change export settings" (blanco con borde azul);
  - "Export on our server (1080p)".
- **Figma:** 🔴 no existe en mis páginas. El código cita 1233:3199.

### 43. Exportación en el servidor — `EditorV2Sheets.swift` (`ExportSheet`)
- **Cómo se llega:** "Export on our server (1080p)" en nº 38 o nº 42.
- **Fondo:** **OSCURO VIEJO**: #060B18 forzado con `.preferredColorScheme(.dark)`, hoja a media altura.
- **Elementos:**
  - asa #3A5280;
  - "Export" en Bold 22 blanco;
  - textos en #AFC4E8: "Getting your video ready…", "Preparing your export…", "Getting your clips…", "Making your video…", "Finishing…";
  - "45%" en Bold 16 #19E6FF (cian viejo);
  - `ProgressView` cian;
  - "You can leave this screen. We'll let you know when it's ready.";
  - final: "Your video is ready." o "The export didn't finish. Please try again." + `PrimaryButton` "Try again".
- **Figma:** 🔴 no existe. **Pantalla con diseño oscuro viejo.**

### 44. Acciones del video terminado (Publicar en TikTok / Compartir / Guardar) — `FinishedExportActionsView.swift` + `TikTokPoster.swift` (`PostToTikTokButton`)
- **Cómo se llega:** solo dentro de nº 43, cuando el servidor termina. En Plan también, fuera de mi alcance.
- **Fondo:** el de nº 43, oscuro viejo.
- **Elementos:**
  - "Post to TikTok" / "Opening TikTok…" ("Publicar en TikTok"): Bold 16 blanco, cápsula azul 3D de 52 pt, con el asset `TikTokLogo` si existe;
  - "Share" (`ShareLink`, `.bordered` del sistema);
  - "Save" / "Saved" (`.borderedProminent` del sistema);
  - "Prepare to share" / "Preparing video…";
  - avisos en `.caption` #19E6FF: "Caption copied, paste it in TikTok", "Video saved to Photos", "Allow CutSell to add videos to Photos, then try again.";
  - error en rojo del sistema y "Retry".
- **Figma:** 🔴 no existe. Mezcla botones de sistema con el oscuro viejo.

### 45. "Did you post it?" (al volver de TikTok) — `TikTokPoster.swift` (`TikTokReturnPrompt`)
- **Elementos:** alerta del sistema "Did you post it?" ("¿Lo publicaste?") con "We'll mark the video as posted in your plan." y los botones "Yes, posted" y "Not yet".
- **Figma:** 🔴 no existe.

### 46. Notificaciones del sistema
- **Recordatorio:** "Time to post: <producto> on TikTok" + "<marca> · Caption and video are ready." ("El caption y el video están listos."). Está en `PlanWeekView.swift` (`PostReminders`). Al tocarla se abre Plan en ese día.
- **Corte listo:** "<corte> is ready" + "Open CutSell to review and export it.".
- **Exportación lista:** "Your video is ready" + "Open CutSell to save or share it.".
- **Figma:** 🔴 no existe.

### 47. Estadísticas de ediciones — `EditStats.swift` (`EditStatsPage`)
- **Cómo se llega:** Profile → "Your results". No está en la pestaña Edits.
- **Fondo:** claro.
- **Elementos:**
  - "Your results" / "Everything CutSell edited for you";
  - "This week · This month · Lifetime";
  - tarjeta azul "TIME SAVED" con horas en ExtraBold 44 blanco;
  - estadísticas con icono #1769FF sobre #EEF3FF;
  - "Last 8 weeks";
  - meta "N more to N videos";
  - nota "Hours saved = N min of editing by hand per video. A workday is 8 hours.".
- **Figma:** 🔴 no existe en mis páginas.

---

## Pantallas con diseño viejo o detalles fuera de White Indigo

- **Oscuro viejo de verdad:**
  - nº 43 `ExportSheet` (#060B18, texto blanco y cian #19E6FF);
  - nº 44 dentro de ella.
- **Texto blanco sobre fondo viejo que en claro sale casi blanco (probablemente ilegible, no confirmado en dispositivo):**
  - `ToastPillV4` (todos los toasts del editor);
  - `hintPill` ("Type here · drag · pinch", "Drag · pinch · turn with two fingers");
  - `zoomBadge`;
  - `soundDownloadPill`;
  - `brollPill`;
  - `RestoreBanner`.
- **Rojos viejos:**
  - #FF2E5E en nº 12;
  - #FF8A8A en nº 38;
  - #FF4D67 en el borrar de nº 31 y nº 35 (WI usa #E5484D).
- **Fuente SF en vez de Plus Jakarta:**
  - todos los títulos y textos de `PanelCardV5` (Volume, Speed, Sounds, Add overlay, Overlay, voz seleccionada);
  - la barra Curve;
  - el texto del subtítulo en los clips del timeline;
  - la cuenta atrás;
  - `FinishedExportActionsView` (`.caption` y botones del sistema).
- **Logo:** `ReadyBanner` usa `ProcessingLogo` con `.blendMode(.screen)` sobre una tarjeta clara (probable logo invisible).
- **Fondo del editor:** brillos rosa y lila (`EditorBackdrop`) que la base de Figma no tiene.
- El video del editor (y la pantalla completa negra) es oscuro por naturaleza; no cuenta como diseño viejo.

## Navegación del flujo

1. Barra → **Edits** → Ready (grid) | Processing | Products | Drafts | Posted
2. Tile "•••" → Personal / Use in Plan · Delete → "Delete project?"
3. Ready → Select → elegir → "Add to Plan · N videos" → Plan › Plan my week
4. Processing → fila → `CutProgressView` (procesando) → Ready ("Your cut is ready") → "Review & edit" → **Editor**
5. Tile lista → `DraftEditorView` ("Opening your cut…") → **EditorV4View**
6. Tile "created" → `UploadNotFinishedView` → "Resume upload" → `NewCutView`
7. Products → producto → `ProductDetailView` → Assign → `AssignProductSheet`
8. Editor → herramientas:
   - Edit (Split / Delete / Duplicate / Mirror / Rotate)
   - Audio (Voice Over → permiso de micrófono → grabar → 3·2·1 → escuchar → guardar · Music → CutSell sounds / Music from Files · Volume · Speed)
   - Text → hoja Fuentes/Estilo
   - Captions → hoja Fuentes/Estilo
   - Overlay → Add overlay → Photo/Video → selector de fotos → overlay seleccionado
   - Style → Auto Zoom AI
9. Editor → tocar subtítulo → escribir + panel 2b
10. Editor → tocar voz / sonido / overlay → su panel
11. Editor → "Export" → ajustes:
    - "Save to Photos" → progreso → "Saved to Photos" → Share / Done
    - "Add to Plan to post" → progreso → "Opening Week Plan…" → Plan my week
12. Fallo → "Export failed" → Try Again | Change export settings | "Export on our server (1080p)" → `ExportSheet` (oscuro) → "Post to TikTok" / Share / Save → TikTok → al volver, "Did you post it?"
13. Notificaciones:
    - "Your cut is ready" → banner o toast → Ready
    - "Time to post…" → Plan (ese día)

## Frames de Figma de mis páginas que NO existen en la app

- **"6 Overlay Galería" (1220:533):** la hoja propia "Cancel · Photos & Videos" no existe; la app usa el selector de Apple.
- **"1 Base" (1218:2) y todos los frames del editor:** la barra plana "Split · Delete · Mirror · Rotate | Overlay · Voice Over · Audio · V…" ya no existe como fila principal.
- **"16 Volumen" (1222:2781):** no existen la etiqueta "Level", las marcas 0/100/200, el botón redondo "Clean up" ni la nota "applies on export".
- **"18 Audio" (1222:3563):** no existen los iconos de cada sonido, el botón "My music" ni "Add Typing at playhead".
- **"17 Velocidad" (1222:3179):** no existe el subtítulo "Selected clip · 0:08→0:04".
- **"1 Base" (1218:2) y "Edits · Ready" (1213:2):** no existe la duración en las tiles del grid ("0:05").
- **Frames enteros que no existen:** ninguno. Todos los frames tienen una pantalla equivalente en la app, aunque con cambios.

## No se usan (código presente, no alcanzable en el build 78)

- `EditorV2View` (no se crea en ningún sitio).
- `DraftEditorView.editor(_:)` y lo que solo vive ahí: `TimelineEditorView`, `CaptionsPanelView`, `CaptionsView`, `VisualTimelineView`, `EditorExtrasView`, `ReorderList`, `DraftPlaybackView` (la vista) y el botón "Export 1080p".
- Los que cuelgan de esas vistas: `VoiceOverView`, `BrollView`, `OverlayView` y `StyleSheetView` (este último solo se usa en Create).
- `VoiceOverPanel` (EditorV4Panels.swift), `VoiceOverRecordSheet` y `AddCaptionsSheet` (solo en modo de capturas de Debug).
- `SoundPanelV4` / `SoundSlider`.
- `EditsFilterChip` (solo dentro de `chip()`, que no se usa).
- `ProcessingView` (ProcessingView.swift).
- `ExportScheduleContent` ("Ready to post / Post on / Day / Time / Schedule · … / Decide later"): **inalcanzable**, porque `schedulingOne` (ProjectsView) y `scheduling` (ProductsView) nunca reciben un valor.
- `BulkStartedSheet` no aparece desde Edits (no confirmado si se usa desde Create).
- `DesignScreenshots.swift` es un banco de pruebas.

## Conteo

- ✅ igual: **5**
  - nº 8 Toast ready
  - nº 11 hoja Assign
  - nº 33 VO grabar
  - nº 34 VO guardar
  - nº 35 VO seleccionada
- 🟠 distinta: **15**
  - nº 1, nº 2
  - nº 11 lista y nº 11 detalle (2)
  - nº 14, nº 20, nº 24, nº 25, nº 26, nº 27, nº 28, nº 29, nº 32, nº 36, nº 38
- 🔴 no existe: **29**
  - nº 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 30, 31, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47
- La pantalla Ready de `CutProgressView` no entra en el conteo: solo aparece como fondo en 1259:3101 y queda fuera de mis páginas.
