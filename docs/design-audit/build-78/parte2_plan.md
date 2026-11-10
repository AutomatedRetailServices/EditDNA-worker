# Auditoría de diseño · Parte 2 · PLAN (build 78 · commit 71ae403)

Fuente: código `/home/claude/buildclon/mobile/ios/CutSell/*.swift` (solo lectura) + Figma `rXfUuJQgwZIZJTuONvU1Z7`, página `3003:13 "03 · Plan"` y `3003:5863 "00 · Componentes y colores"`.

## Cómo leer los colores (resuelto desde el código)
- La app siempre está en modo claro: `AppPrefs.colorScheme` devuelve `.light` (YouView.swift:52). Por eso los colores viejos oscuros `Color(hex:)` se pintan con su versión clara de `LightPalette` (Theme.swift:166). Lo de abajo es el hex que **se ve**.
- `CS.accent` #1769FF · `CS.accentSoft` #EEF3FF · `CS.wiInk` #0B1020 · `CS.wiSecondary` #6B7385 · `CS.wiBorder` #E6E9EF · `CS.wiPage` #FFFFFF · `CS.wiCard` #F8F9FB · `CS.wiSurface2` #F3F5F9 · `CS.wiSuccess` #1FA971 · `CS.wiError` #E5484D · `CS.wiReady` #33D98C.
- Tokens viejos usados en Plan: `Color.ink` → #0B1020 · `CS.cyan` → #1769FF · `PlanStyle.background` (0x03050D/0x050B1C/0x030409) → #FFFFFF liso · `PlanStyle.muted` (0x99A3BD) → #6B7385 · `PlanStyle.border` (0x33405C) → #E6E9EF · `PlanStyle.card` (0x0F1726) → #F8F9FB · `Color(hex:0x1A2236)` → #F3F5F9 · `Color(hex:0x0A0F1A)` → ≈#F8F9FC · `CS.brandFull` / `PlanStyle.cta` → #1769FF liso · `CS.error` (0xFF4D67) → #FF4D67 (NO es el rojo WI #E5484D) · `Color(hex:0x33D98C)` → #33D98C · `Color(hex:0x143324)` → ≈#CBEBDB · `Color(hex:0xFFB840)` → #FFB840.
- `softPuff(.white)` = tarjeta "3D suave": degradado #FFFFFF→#EEF3FF, base 2 pt #D3DDF5, sombra #1769FF 16 %×0.55. `softPuff(.blue)` = degradado #2F7BFF→#0A55F0, base #0636B8 (no es #1769FF plano). `.ink` = #2A3045→#0B1020.
- Fuente: `CS.inter(...)` = **Plus Jakarta Sans** (Regular/Medium/SemiBold/Bold/ExtraBold). "Sec 17" = `CS.sectionTitle` = Jakarta Bold 17.
- `HomeBackdrop` = degradado #E7EEFF → #F5F8FF → #FFFFFF + 2 manchas borrosas (#1769FF 16 %, #4A5BFF 12 %).
- Barra de abajo `CSTabBar`: cápsula blanca h68, borde #E6E9EF, sombra #0D1433 10 %; activa = icono+texto #1769FF Bold 11 dentro de burbuja #EEF3FF; inactiva icono #0B1020, texto #6B7385 Medium 11. Assets `WITabCreate/WITabEdits/WITabPlan/WITabProfile`. Textos "Create"(Crear) "Edits"(Mis videos) "Plan" "Profile"(Perfil). **Figma (Nav/White 1166:182): sin burbuja, activo azul con un punto debajo** → 🟠 en todas las pantallas de Plan.

---

### 1. Plan (inicio del tab) — `PlanView.swift` (`PlanView`, var `home`)
- Cómo se llega: barra de abajo → "Plan". También al tocar "Plan" estando dentro de Plan (vuelve a la raíz).
- Fondo: claro · `HomeBackdrop` (degradado #E7EEFF→#F5F8FF→#FFFFFF + manchas azules).
- Elementos de arriba hacia abajo:
  1. Logo horizontal — imagen asset `LogoHorizontal`, 172 pt de ancho (WIHeader, alto 67). Sin botones a la derecha.
  2. "Plan" — título / #0B1020 / Jakarta ExtraBold 32, tracking −0.64.
  3. "What to post this week." (Qué publicar esta semana.) — subtítulo / #6B7385 / Medium 15. Si hay envío Bulk con fallo: "Plan needs attention" (El plan necesita atención) → ver pantalla 2.
  4. Botón grande "Week Plan" (Plan de la semana) — píldora 3D azul `softPuff(.blue)` h68, radio 24; SF Symbol `calendar.badge.plus` 22 semibold blanco; título Bold 17 blanco #FFFFFF; debajo "Pick your videos, we set the times" (Elige tus videos, nosotros ponemos las horas) o, si ya hay semana, "Change what and when you post" (Cambia qué y cuándo publicas), Medium 12 blanco 85 %; `chevron.right` 15 bold a la derecha. Abre Plan my week (pantalla 3).
  5. "This week" (Esta semana) — sección / #0B1020 / Bold 17.
  6. Tira de 7 días (los próximos 7 desde HOY, hoy primero, no Lun–Dom): cada día botón h64 radio 16; hoy = `softPuff(.blue)` texto blanco; los demás `softPuff(.white)` texto #0B1020; el día tocado (si no es hoy) con borde #1769FF 2.5 pt. Dentro: nombre corto del día ("Mon"… según idioma) SemiBold 11; número Bold 14; hasta 4 puntos de 6 pt (assets `PlanDotCyan` #19E6FF, `PlanDotMagenta` #FF21D6, `PlanDotGreen` #33D98C, `PlanDotAmber` #FFB840), los publicados al 35 %.
  7. Tarjeta del día elegido (ver Estados). Con videos: cabecera "Today · Tue 21" (Hoy · …) o "Tue 21", Bold 16 #0B1020; a la derecha "N to post" ({} por publicar) SemiBold 12 #1769FF o "All posted ✓" (Todo publicado ✓) #1FA971. Cada video = fila `softPuff(.white)` radio 20, padding 10: miniatura 50×66 radio 12 (relleno #E9ECF1); punto 7 pt + marca SemiBold 13 #0B1020 ("No brand" / Sin marca); producto Regular 12 #6B7385 ("Video"); a la derecha píldora de hora Bold 13 #1769FF sobre #EEF3FF h30, o "Posted" (Publicado) con SF `checkmark` SemiBold 12 #1FA971 (fila al 60 %). Tocar fila → This week → Día.
  8. "See the whole week ›" (Ver toda la semana ›) — enlace SemiBold 14 #1769FF (solo si hay semana). Abre This week.
  9. Barra de abajo `CSTabBar` (Plan activo).
- Estados:
  - Vacío (sin semana): tarjeta `softPuff(.white)` radio 20: "Plan your week" (Planea tu semana) Bold 15 #0B1020 + "When your videos are ready, CutSell spreads them over the week so your feed stays varied." (Cuando tus videos estén listos, CutSell los reparte en la semana para que tu feed tenga variedad.) o "You have N ready video(s). We spread them over the week, between products, so your feed stays varied." (Tienes N video(s) listo(s)…), Regular 12 #6B7385.
  - Hay semana pero nada ese día: "Nothing planned for today" (Nada planeado para hoy) / "Nothing planned for Tue 21" (Nada planeado para el …) + "We spread your N videos over D days, between products, so your feed stays varied." (Repartimos tus N videos en D días…).
  - Cargando: no hay indicador (carga en silencio; tirar hacia abajo refresca).
  - Error: no se muestra nada (`lastError` nunca se pinta).
  - Éxito: tarjeta con la lista de videos del día.
- Hojas/popups que abre: Plan my week (push), This week (push), Día (push), `MixSetupView` fullScreen (solo desde "New Sales Mix ›" de This week).
- Figma: 🟠 distinta — "Plan" `1166:107`. Diferencias: (a) Figma tiene subtítulo "What’s in progress and what to post." y la sección "In progress" con 2 tarjetas SALES MIX / BULK + "See all in Edits ›" → la app las quitó; (b) la app añade la píldora azul grande "Week Plan" que no está en Figma; (c) Figma muestra Lun→Dom con "hoy" (Tue) azul plano y días con borde fino h62; la app muestra los 7 días desde hoy, 3D suave h64; (d) Figma: una sola tarjeta plana #F8F9FB con borde "Today: post 1 CeraVe and 1 Dior video" + texto + enlace "Plan my week ›"; la app muestra filas de video con miniatura y píldora de hora, y el enlace es "See the whole week ›"; (e) fondo Figma blanco liso, app con degradado azulado `HomeBackdrop`; (f) barra de abajo con burbuja (ver arriba).
- Diseño: White Indigo ✅ (no es diseño viejo oscuro).

### 2. Plan · "Needs attention" (estado de error de un Bulk) — `NeedsAttention.swift` (`NeedsAttentionSection`, `NeedsAttentionRow`) dentro de `PlanView`
- Cómo se llega: automático en Plan cuando el envío Bulk más reciente tiene un video que falló de verdad (no "sin voz").
- Fondo: claro · `HomeBackdrop` (misma pantalla 1).
- Elementos (entre el subtítulo y la píldora "Week Plan"):
  1. Tarjeta resumen `wiCard()` → `softPuff(.white)` radio 20, min h130: anillo 70 pt (fondo #E6EAF2, progreso #1769FF 6 pt) con "%"; "N of M ready" ({} de {} listos) Bold 19 #0B1020; "N finished cuts are already in Edits" ({} cortes terminados ya están en Edits) Medium 11 #6B7385. Si aún hay videos en curso: puntos "N ready" ({} listos) #33D98C, "N active" ({} activos) #1769FF, "N queued · N failed" ({} en cola · {} fallidos) #6B7385, Medium 10; y asset `WIClock` + "1 video needs action" / "N videos need action" Medium 11 #0B1020.
  2. "Needs attention" (Necesita atención) Bold 17 + nota "Retry failed" (Reintenta los fallidos) Medium 12 #6B7385.
  3. Filas h84 `softPuff(.white)`: número "01" SemiBold 11 (#1769FF si falló, si no #6B7385); miniatura 46×60 radio 10 (#F3F5F9); nombre SemiBold 13 #0B1020; duración Medium 12 #6B7385; estado "COMPLETED" (COMPLETADO) / "FAILED" (FALLÓ) / "QUEUED" (EN COLA) / "PROCESSING" (PROCESANDO) SemiBold 10 #1769FF; a la derecha: botón "Retry" (Reintentar) SemiBold 11 #1769FF sobre blanco con borde #E5484D 1.2, o SF `checkmark.circle.fill` 22 #33D98C, o `WIClock` 22 #6B7385, o anillo girando.
- Estados: es en sí un estado de error. Retry → reintenta el ítem.
- Hojas/popups: ninguna.
- Figma: 🔴 no existe en mi página (el código cita 1233:3243 / 1233:3354, que están en otra página; no confirmado).

### 3. Plan my week ("Week Plan") — `PlanMyWeek.swift` (`PlanMyWeekView`)
- Cómo se llega: Plan → píldora "Week Plan"; Create (Home) → burbuja "Week Plan" (`AppRouter.pending = .planWeek`); guía de intro "Try Week Plan now"; Edits › Seleccionar › "Add to Plan"; Export › "Add to Plan to post"; Productos (estos 3 últimos abren el modo de la pantalla 4).
- Fondo: claro · `HomeBackdrop`.
- Elementos:
  1. "‹ Plan" — enlace `PlanBackButton` SemiBold 16 #1769FF (vuelve atrás).
  2. "Week Plan" (Plan de la semana) — título Bold 28 #0B1020; a la derecha "Select all" (Seleccionar todo) SemiBold 15 #1769FF.
  3. "Tap videos in the order you want to post." (Toca los videos en el orden en que quieres publicarlos.) Regular 14 #6B7385.
  4. Chips con scroll lateral: "All products" (Todos los productos) + una por marca; activa: fondo #0B1020 texto blanco; inactiva: #F3F5F9 borde #E6E9EF texto #0B1020; SemiBold 14, h34, cápsula.
  5. Rejilla 3 columnas (espacio 10): tarjeta proporción 0.72, radio 12, fondo #E9ECF1 + miniatura, degradado negro 55 % abajo, marca Bold 12 blanca ("No product" / Sin producto). Elegida: borde #1769FF 3 pt + círculo 24 azul con número Bold 12 blanco. No elegida: círculo blanco 85 % 24 pt. Ya programada: velo blanco 60 % y deshabilitada.
  6. Panel fijo abajo (blanco #FFFFFF, esquinas sup. 24, sombra #0D1433 8 %):
     - 4 "pills" `softPuff(.white)` radio 16 h56: etiqueta Medium 12 #6B7385 + valor SemiBold 16 #0B1020 + SF `chevron.down` 11 #6B7385: "Per day" (Por día) "4 videos" (ancho 110); "Between" (Entre) "9:00 AM – 9:00 PM"; "Days" (Días) "Mon – Sat"; "Starts" (Empieza) "Tomorrow" (Mañana) / "Today" (Hoy) / "Tue 21".
     - "Post on" (Publicar en) Medium 12 #6B7385 + 3 cápsulas h34 "TikTok" "Instagram" "Facebook" SemiBold 13; activa `softPuff(.blue)` texto blanco; resto `softPuff(.white)` texto #0B1020.
     - "Mix products automatically" (Mezclar productos automáticamente) Medium 14 #0B1020 + interruptor `CSSwitch` 51×31 (on #1769FF / off #D5DAE3, bola blanca 27).
     - Ayuda Regular 11 #6B7385: "Off: posts in the order you tapped (1, 2, 3…)." (Apagado: publica en el orden en que los tocaste (1, 2, 3…).) / "On: never the same product twice in a row." (Prendido: nunca el mismo producto dos veces seguidas.)
     - Botón "Build plan · N videos" (Armar plan · N video(s)) SemiBold 16 blanco + línea "Mon 26 → Tue 27 · 4 a day · reminders on" ({} → {} · {} al día · recordatorios prendidos/apagados) Medium 11; cápsula `softPuff(.blue)` h54; 45 % de opacidad y deshabilitado si no hay ninguno elegido.
  7. Barra de abajo `CSTabBar` (sí aparece, debajo del panel).
- Estados:
  - Vacío: "No Ready videos yet. Finished cuts show here." (Aún no hay videos listos. Los cortes terminados salen aquí.) Medium 13 #6B7385.
  - Cargando: al armar, el botón muestra `BrandRing` 22 (anillo de colores girando) en lugar del texto. Carga inicial sin indicador.
  - Error: alerta del sistema título "CutSell", mensaje "Your plan couldn’t be built. Check your connection and try again." (No se pudo crear tu plan. Revisa tu conexión y vuelve a intentarlo.), botón "OK".
  - Éxito: pide permiso de notificaciones (si recordatorios activos) y salta a This week.
- Hojas/popups: `PlanSettingSheet` (pantallas 5–8), alerta de error, permiso de notificaciones del sistema.
- Figma: 🟠 distinta — "Plan my week" `1258:3101`. Diferencias: (a) título Figma "Plan my week", app "Week Plan"; (b) la fila "Post on · TikTok / Instagram / Facebook" no existe en Figma; (c) Figma pills planas (#F8F9FB con borde), app 3D suave; (d) interruptor Figma 44×24, app 51×31; (e) Figma no tiene barra de abajo en esta pantalla, la app sí; (f) fondo Figma blanco, app `HomeBackdrop`. Igual: chips (negro activo), rejilla 3 columnas con números azules, textos de las 4 pills, línea de resumen del botón.
- Diseño: White Indigo ✅.

### 4. Plan my week · modo "videos elegidos" — `PlanMyWeek.swift` (`PlanMyWeekView`, var `chosenStrip`)
- Cómo se llega: Edits › Seleccionar › "Add to Plan"; Export › "Add to Plan to post" (tras exportar, texto "Opening Week Plan…"); Productos (preselección). `fromEdits = true`.
- Fondo: claro · `HomeBackdrop`.
- Elementos: igual que la 3 pero sin "Select all", sin texto de ayuda, sin chips ni rejilla. En su lugar una tarjeta `softPuff(.white)` radio 20 con "N videos chosen" (Elegiste N video(s)) Bold 15 #0B1020 y tira horizontal de miniaturas 74×106 radio 12 con marca Bold 11 blanca y número en círculo azul 22. El panel de abajo es el mismo.
- Estados: los mismos de la 3 (no muestra el texto vacío).
- Hojas/popups: las mismas de la 3.
- Figma: 🔴 no existe.

### 5. Hoja "Per day" — `PlanMyWeek.swift` (`PlanSettingSheet`, which = .perDay)
- Cómo se llega: Plan my week → pill "Per day".
- Fondo: claro #FFFFFF (`presentationBackground(CS.wiPage)`), alto 400.
- Elementos: asa #D5D9E2 40×5; "Per day" (Por día) Bold 20 #0B1020; tarjeta `softPuff(.white)` radio 16 con botón "−" (círculo 44 #F3F5F9, Bold 22 #0B1020), número Bold 34 #0B1020, "videos a day" (videos al día) Regular 12 #6B7385, botón "+" (círculo 44 #1769FF, blanco); 40 % si no se puede (1–50); "From 1 to 50 a day. The times are spread evenly inside “Between”." (De 1 a 50 al día. Las horas se reparten parejas dentro de “Entre”.) Regular 12 #6B7385; botón "Done" (Listo) Bold 16 blanco, cápsula `softPuff(.blue)` h50.
- Estados: sin vacío/carga/error.
- Figma: 🔴 no existe.

### 6. Hoja "Between" — `PlanSettingSheet` (.between)
- Cómo se llega: Plan my week → pill "Between".
- Fondo: claro #FFFFFF, alto 400.
- Elementos: asa; "Between" (Entre) Bold 20; tarjeta radio 16 (soft 3D) con fila "From" (Desde) SemiBold 15 #0B1020 + selector de hora compacto del sistema (tinte #1769FF), línea #E6E9EF 1 pt, fila "To" (Hasta); vista previa "4 a day: 9:00 AM · 1:00 PM · 5:00 PM · 9:00 PM" ({} al día: {}) Regular 12 #6B7385 (máx. 6 horas + " …"); "Done" (Listo).
- Figma: 🔴 no existe.

### 7. Hoja "Days" — `PlanSettingSheet` (.days)
- Cómo se llega: Plan my week → pill "Days".
- Fondo: claro #FFFFFF, alto 360.
- Elementos: asa; "Days" (Días) Bold 20; 7 círculos 42 pt "M" "T" "W" "T" "F" "S" "S" Bold 14 (activo #1769FF texto blanco; inactivo #F3F5F9 borde #E6E9EF texto #6B7385); mínimo 1 día; "Done" (Listo).
- Nota: las letras M/T/W/F/S y "Mon"/"Sat" del resumen NO tienen traducción → en español salen en inglés ("Mon – Sat").
- Figma: 🔴 no existe.

### 8. Hoja "Starts" — `PlanSettingSheet` (.starts)
- Cómo se llega: Plan my week → pill "Starts".
- Fondo: claro #FFFFFF, alto 400.
- Elementos: asa; "Starts" (Empieza) Bold 20; selector de fecha compacto del sistema (desde hoy, tinte #1769FF); enlace "Tomorrow" (Mañana) SemiBold 14 #1769FF (cierra la hoja); "Done" (Listo).
- Figma: 🔴 no existe.

### 9. Alerta de error al armar el plan — `PlanMyWeekView` (`.alert`)
- Cómo se llega: falla "Build plan".
- Fondo: alerta del sistema iOS (clara).
- Elementos: título "CutSell"; mensaje "Your plan couldn’t be built. Check your connection and try again." (No se pudo crear tu plan. Revisa tu conexión y vuelve a intentarlo.); botón "OK".
- Figma: 🔴 no existe.

### 10. Permiso de notificaciones (sistema) — `PlanMyWeekView.build()` → `UNUserNotificationCenter.requestAuthorization`
- Cómo se llega: primer "Build plan" con recordatorios activos.
- Fondo: alerta del sistema (texto de Apple, no controlado por la app).
- Figma: 🔴 no existe.

### 11. This week — `PlanWeekView.swift` (`PlanWeekView`)
- Cómo se llega: Plan → "See the whole week ›"; tras "Build plan"; tocar un recordatorio (abre This week → Día). (El destino `.thisWeek` de `AppRouter` solo lo usa `ExportScheduleContent`, que no se puede abrir.)
- Fondo: claro #FFFFFF liso (`PlanStyle.background`, token viejo mapeado a blanco).
- Elementos:
  1. "‹ Plan" SemiBold 16 #1769FF; a la derecha "Settings" (Ajustes) SemiBold 14 #1769FF.
  2. "This week" (Esta semana) Bold 28 #0B1020.
  3. "8 videos · 4 products · 2 per day" ({} video{} · {} producto{} · {} por día) Regular 13 #6B7385.
  4. Tarjeta por día `softPuff(.white)` radio 20, padding 13; hoy además borde #1769FF 1.5. Cabecera: "Mon 20" Bold 14 #0B1020 + etiqueta "POSTED" (PUBLICADO) Bold 9 #33D98C sobre ≈#CBEBDB 58×18, o "TODAY" (HOY) Bold 9 blanco sobre #1769FF 51×18. Tocar cabecera → Día.
  5. Rejilla 2 columnas por día: miniatura 36×58 radio 7 (#F3F5F9); punto 7 pt + marca SemiBold 11 #0B1020 ("No product" / Sin producto); producto (o código de mix) Regular 11 #6B7385; hora SemiBold 11 (#1769FF si pendiente y destacado, si no #6B7385). Tocar → hoja de acciones (12). Mantener pulsado → menú contextual (15).
  6. Aviso de huecos (si faltan videos): caja #FFF7E6 borde #F5D9A8 radio 14: "Videos for 4 of 6 days" (Videos para {} de {} días) SemiBold 13 #D97706; "Make 4 more to fill Fri and Sat." (Haz {} más para llenar {}.) Regular 12 #6B7385; "New Sales Mix ›" (Nuevo Sales Mix ›) SemiBold 13 #1769FF → abre Mix Setup.
  7. Si el plan termina antes: "Plan ends early: N of M videos fit without the same product back to back." (El plan termina antes: caben…) Regular 12 #FFB840 (ámbar sobre blanco, poco legible).
  8. Barra de abajo `CSTabBar`.
- Estados: vacío = solo título y "0 videos · 0 products · N per day" sin mensaje; cargando = sin indicador (tirar para refrescar); error = silencioso; éxito = tarjetas.
- Hojas/popups: `SlotActionsSheet` (12), `SwapVideoSheet` (13), `MoveSlotSheet` (14), menú contextual (15), `PlanWeekSettingsSheet` (16), `MixSetupView` (fullScreen, otra parte).
- Figma: 🟠 distinta — "This week" `1209:70`. Muy parecida en estructura y textos. Diferencias: (a) tarjetas Figma planas #F8F9FB con borde #E6E9EF, app 3D suave sin borde gris; (b) etiqueta POSTED en Figma con verde de éxito más oscuro sobre verde claro (≈#1FA971 / #EAF8F1, leído de la captura, no confirmado) vs app #33D98C / #CBEBDB; (c) el aviso "Plan ends early" ámbar no existe en Figma; (d) barra de abajo con burbuja.
- Diseño: tokens VIEJOS oscuros (`PlanStyle.*`, `Color.ink`, `CS.cyan`, `Color(hex:)`) que solo se ven White Indigo por el mapeo claro; no usa los tokens `wi*`.

### 12. Hoja de acciones de un video — `PlanMyWeek.swift` (`SlotActionsSheet`)
- Cómo se llega: This week → tocar un video.
- Fondo: claro #FFFFFF, alto 400.
- Elementos: asa #D5D9E2 40×5; "Tue 27 · 9:00 PM" Medium 14 #6B7385; tarjeta `softPuff(.white)` radio 16 con miniatura 40×52 radio 8 (#E9ECF1), producto Bold 17 #0B1020 ("Video"), "Dior · 0:12" Regular 13 #6B7385; 3 filas h56 `softPuff(.white)` radio 16, SemiBold 16, SF Symbol 17 medium + `chevron.right` 13 #6B7385: `arrow.left.arrow.right` "Swap video" (Cambiar video) #0B1020; `clock` "Change day or time" (Cambiar día u hora) #0B1020; `trash` "Remove from plan" (Quitar del plan) #E5484D.
- Estados: Quitar = sin confirmación y sin aviso de error (`try?`).
- Hojas/popups: Swap (13), Move (14).
- Figma: 🟠 distinta — "This week · Cambiar video" `1257:3329`. Igual en textos, iconos y orden. Diferencias: Figma tarjeta del video plana #F8F9FB con borde y filas blancas con borde #E6E9EF; app todo 3D suave sin borde. Figma muestra el fondo oscurecido (Dim); la app usa la atenuación estándar de iOS.

### 13. Hoja "Swap video" — `PlanMyWeek.swift` (`SwapVideoSheet`)
- Cómo se llega: Hoja de acciones → "Swap video".
- Fondo: claro #FFFFFF, altura completa (.large).
- Elementos: asa; "Swap video" (Cambiar video) Bold 20 #0B1020; "Tue 27 · 9:00 PM" Regular 13 #6B7385; rejilla 3 columnas igual que Plan my week (proporción 0.72, radio 12, marca Bold 12 blanca). Tocar uno = cambiar.
- Estados: vacío "Every Ready video is already in your plan." (Todos los videos listos ya están en tu plan.) Regular 13 #6B7385; error "Couldn’t swap it. Please try again." (No se pudo cambiar. Inténtalo de nuevo.) Regular 12 #E5484D; cargando sin indicador; éxito cierra la hoja.
- Figma: 🔴 no existe.

### 14. Hoja "Move" (cambiar día u hora) — `PlanWeekView.swift` (`MoveSlotSheet`)
- Cómo se llega: Hoja de acciones → "Change day or time"; o menú contextual → "Move to another day or time".
- Fondo: claro #FFFFFF, alto 420. Sin asa.
- Elementos: "Move Lip Glow" (Mover {}) Bold 20 #0B1020; selector de fecha del sistema con etiqueta "Day" (Día); selector de hora "Time" (Hora); botón "Move" (Mover) SemiBold 15 blanco, cápsula plana #1769FF h50 (no 3D).
- Estados: error "Couldn’t move it. Please try again." (No se pudo mover. Vuelve a intentarlo.) Regular 12 **#FF4D67** (rojo viejo, no #E5484D); éxito cierra.
- Figma: 🔴 no existe.
- Diseño: tokens viejos (`Color.ink`, `PlanStyle.cta`, `CS.error`), botón plano distinto al resto (3D).

### 15. Menú contextual de un video — `PlanWeekView` (`.contextMenu`)
- Cómo se llega: This week → mantener pulsado un video.
- Fondo: menú del sistema iOS.
- Elementos: "Move to another day or time" (Mover a otro día u hora); "Remove from this week" (Quitar de esta semana) en rojo del sistema.
- Figma: 🔴 no existe.

### 16. Hoja "Settings" de This week — `PlanMyWeek.swift` (`PlanWeekSettingsSheet`)
- Cómo se llega: This week → "Settings".
- Fondo: claro #FFFFFF, alto 260.
- Elementos: asa; "Settings" (Ajustes) Bold 20 #0B1020; fila h54 radio 16 (3D suave) "Remind me at each time" (Recordarme a cada hora) SemiBold 15 #0B1020 + `CSSwitch`; "A notification at each video’s time, to post it in one tap." (Un aviso a la hora de cada video, para publicarlo con un toque.) Regular 12 #6B7385.
- Estados: guarda al cambiar, error silencioso (`try?`).
- Figma: 🔴 no existe.

### 17. Día (Today / un día) — `PlanWeekView.swift` (`PlanDayView`)
- Cómo se llega: This week → tocar la cabecera de un día; Plan → tocar un video del día; tocar un recordatorio "Time to post".
- Fondo: claro #FFFFFF liso (`PlanStyle.background`).
- Elementos:
  1. "‹ This week" (sin traducción: queda en inglés en español) SemiBold 16 #1769FF.
  2. "Today · Tue 21" (Hoy · …) o "Tue 21" Bold 28 #0B1020.
  3. "2 videos to post" ({} video{} para publicar) / "Everything is posted 🎉" (Todo está publicado 🎉) Regular 13 #6B7385.
  4. Tarjeta principal `softPuff(.white)` radio 18, padding 13: miniatura 124×222 radio 12 (#F3F5F9) con etiqueta "SALES MIX" Bold 9 blanca sobre #1769FF si es mix y barra de progreso #1769FF al preparar; punto 8 + marca SemiBold 13 #0B1020; producto Bold 15 #0B1020; "9:00 AM · reminder on" ( · recordatorio activado) SemiBold 12 #1769FF; "H2 · B1 · C2 · 0:36" Regular 11 #6B7385; caja de caption ≈#F8F9FC borde #E6E9EF radio 12 min h132: "✦ SUGGESTED CAPTION" (✦ CAPTION SUGERIDO) o "✦ YOUR CAPTION" (✦ TU CAPTION) Bold 9 #1769FF; texto Regular 11 #0B1020 o "Writing your caption…" (Escribiendo tu caption…); "Edit" (Editar) y "Copy" (Copiar) SemiBold 11 #1769FF.
  5. Botón "Share to TikTok" (Compartir en {}) SemiBold 16 blanco, cápsula `softPuff(.blue)` h50; mientras prepara "Preparing… 45 %" (Preparando… {} %).
  6. Dos botones h46 cápsula `softPuff(.white)` SemiBold 13 #0B1020: "Save to Photos" (Guardar en Fotos) y "✓  Mark as posted" (✓  Marcar como publicado) / "✓  Posted" (✓  Publicado).
  7. Por cada otro video: "Tonight · 7:00 PM" o "Later · 9:00 AM" SemiBold 14 #0B1020 (**sin traducción**: queda en inglés en español); tarjeta `softPuff(.white)` radio 16: miniatura 40×60 radio 8, punto + marca SemiBold 12, producto Bold 14, estado Regular 11 #6B7385 "Posted" (Publicado) / "Writing caption…" (Escribiendo el caption…) / "Caption ready" (Caption listo), "›" Medium 20 #6B7385. Tocar = pasa a ser el principal.
  8. Aviso en línea SemiBold 12 #1769FF (ver Estados).
  9. Barra de abajo `CSTabBar`.
- Estados (avisos en línea): "Caption copied" (Caption copiado); "Caption copied · paste it in Instagram" (Texto copiado · pégalo en {}); "Saved to Photos" (Guardado en Fotos); "Marked as posted" (Marcado como publicado); "Caption saved" (Caption guardado); errores: "Allow CutSell to add videos to Photos." (Permite que CutSell guarde videos en Fotos.), "The video couldn’t be prepared. Check your connection and try again." (No se pudo preparar el video…), "Couldn’t save the video. Please try again." (No se pudo guardar el video…), "Couldn’t save that. Please try again." (No se pudo guardar…), "Couldn’t save the caption. Please try again." (No se pudo guardar el caption…). Todos en azul #1769FF (los errores no se ven rojos). Vacío: solo título + "Everything is posted 🎉".
- Hojas/popups: alerta "Post on … now?" (18); alerta "Edit caption" (19); permiso de Fotos del sistema; abre TikTok/Instagram/Facebook.
- Figma: 🟠 distinta — "Today" `1209:170`. Estructura, textos y tamaños casi iguales. Diferencias: (a) la capa Figma del botón azul se llama "Processing/ContinueInBackground" pero dibuja "Share to TikTok" (igual que la app); (b) Figma botones blanco con borde #E6E9EF y tarjetas planas #F8F9FB con borde; app 3D suave; (c) Figma "✓ Mark as posted" con un espacio, app con dos espacios; (d) los avisos de error de la app no tienen diseño en Figma.
- Diseño: tokens VIEJOS mapeados (`PlanStyle`, `Color.ink`, `CS.cyan`, `Color(hex:0x1A2236/0x0A0F1A)`), se ve claro.

### 18. Alerta "Post on TikTok now?" — `PlanDayView` (`.alert` confirming)
- Cómo se llega: Día → "Share to TikTok".
- Fondo: alerta del sistema.
- Elementos: título "Post on TikTok now?" (¿Publicar en TikTok ahora?); mensaje "Denim Jacket will be marked as posted and move to Edits › Posted. We'll open TikTok with the video and the caption copied." ({} quedará como publicado y pasará a Edits › Posted. Abriremos {} con el video y el texto copiado.); botones "Yes, I'm posting it" (Sí, lo voy a publicar) y "Not now" (Ahora no).
- Figma: 🔴 no existe.

### 19. Alerta "Edit caption" — `PlanDayView` (`.alert` editing)
- Cómo se llega: Día → "Edit" en la caja del caption.
- Fondo: alerta del sistema con campo de texto.
- Elementos: título "Edit caption" (Editar subtítulo — ojo: aquí traduce "subtítulo", en el resto "caption"); campo "Caption"; botones "Save" (Guardar) y "Cancel" (Cancelar).
- Figma: 🔴 no existe.

### 20. Recordatorio "Time to post" (notificación local) — `PlanWeekView.swift` (`PostReminders`)
- Cómo se llega: a la hora de cada video planeado (si "Remind me at each time" y Perfil › recordatorios están activos).
- Fondo: banner del sistema iOS.
- Elementos: título "Time to post: Denim Jacket on TikTok" (Hora de publicar: {} en {}); cuerpo "Old Navy · Caption and video are ready." (… · El caption y el video están listos.); sonido. Tocar → tab Plan → This week → ese Día.
- Figma: 🔴 no existe.

---

## Navegación del flujo
1. Barra de abajo "Plan" → **Plan** (1) [→ estado Needs attention (2) si un Bulk falló]
2. Plan → "Week Plan" → **Plan my week** (3) → pills → hojas Per day / Between / Days / Starts (5–8) → "Build plan" → [permiso notificaciones (10)] → **This week** (11) | error → alerta (9)
3. Create (Home) → burbuja "Week Plan" → tab Plan → **Plan my week** (3)
4. Edits › Seleccionar › "Add to Plan" / Export › "Add to Plan to post" / Productos → tab Plan → **Plan my week · videos elegidos** (4) → "Build plan" → **This week** (11)
5. Plan → "See the whole week ›" → **This week** (11)
6. This week → "Settings" → hoja **Settings** (16)
7. This week → tocar video → **Hoja de acciones** (12) → "Swap video" → **Swap** (13) | "Change day or time" → **Move** (14) | "Remove from plan" (quita directo)
8. This week → mantener pulsado → **menú contextual** (15) → Move (14) / quitar
9. This week → "New Sales Mix ›" → Mix Setup (fuera de esta parte)
10. This week → cabecera del día → **Día** (17); Plan → tocar video del día → This week → **Día** (17)
11. Día → "Share to TikTok" → **alerta Post on…** (18) → abre TikTok/Instagram/Facebook; Día → "Edit" → **alerta Edit caption** (19)
12. **Recordatorio** (20) → tab Plan → This week → **Día** (17)

## Frames de Figma de mi página que NO existen en la app
- "In progress" + tarjetas SALES MIX / BULK + "See all in Edits ›" dentro de "Plan" `1166:107` (nodos `1166:121`, `1166:122`, `1166:123`, `1166:132`) — quitados de la app (el código `progressCard` quedó sin usar).
- Tarjeta "Today: post 1 CeraVe and 1 Dior video" + enlace "Plan my week ›" (`1166:174`) — la app usa otro diseño de tarjeta (el código `cardTitle` quedó sin usar).
- "Programar publicación" `1259:3199` — existe en código (`ExportScheduleContent` en ExportSettingsView.swift) pero **no se puede abrir**: las variables `schedulingOne` (ProjectsView) y `scheduling` (ProductsView) nunca reciben valor, y Export › "Add to Plan to post" ahora abre Plan my week. En la práctica no existe en la app.

## No se usan
- `PlanGoalView` (PlanWeekView.swift, "Week Plan" Goal v2: HOW MANY VIDEOS / POSTING TIMES / DAYS / START / "Build my plan") — ninguna navegación la abre. Además es el **diseño VIEJO oscuro** (`PlanStyle`, `CS.brandFull`, etiquetas `CS.cyan` en mayúsculas, borde punteado #596685, caja 0x121A2B), solo se vería claro por el mapeo.
- `StartDateSheet` (solo desde PlanGoalView).
- `TimePickerSheet` (solo desde PlanGoalView y desde ExportScheduleContent, ambos inalcanzables).
- `ExportScheduleContent` (ver arriba).
- En `PlanView`: `progressCard(...)`, `weekStrip(...)` ("Coming up" / "See this week ›"), `cardTitle(...)`.
- En `PlanDayView`: `share(_:)` y su hoja `ShareSheet` (nunca se llama; "Share to …" usa `confirmAndPost`).
- `PlanWeekView.onSettings` (parámetro sin uso) y `PlanWeekView.DayRoute`.
- `PlanWeekStore.build` / `savePrefs` solo los usa PlanGoalView y el camino de respaldo para servidores viejos.

## Pantallas en diseño viejo / tokens viejos
- Visualmente todo Plan se ve White Indigo (la app está forzada a claro), pero `PlanWeekView`, `PlanDayView`, `MoveSlotSheet`, `StartDateSheet`, `TimePickerSheet` y `PlanGoalView` siguen escritos con tokens del diseño oscuro viejo (`PlanStyle.*`, `Color.ink`, `CS.cyan`, `Color(hex:)`, `CS.brandFull`, `CS.error`). Efectos visibles: rojo de error #FF4D67 en Move (no #E5484D); verde POSTED #33D98C (no #1FA971); texto ámbar #FFB840 sobre blanco en "Plan ends early"; botón "Move" plano mientras el resto es 3D.
- Azul de los botones 3D (`softPuff(.blue)`) = degradado #2F7BFF→#0A55F0, no el #1769FF plano del Figma de componentes (`3003:5863` "Botón/Principal").
- Textos sin traducción al español: "Tonight · …", "Later · …", letras de días "M T W T F S S", "Mon – Sat"/"Mon…Sun", "‹ This week" (el enlace atrás), "SALES MIX".

## Conteo
- ✅ igual: 0
- 🟠 distinta: 5 (Plan 1166:107 · Plan my week 1258:3101 · This week 1209:70 · Hoja de acciones 1257:3329 · Día 1209:170)
- 🔴 no existe en Figma: 15 (Needs attention en Plan · modo videos elegidos · Per day · Between · Days · Starts · alerta error · permiso notificaciones · Swap video · Move · menú contextual · Settings · alerta Post on · alerta Edit caption · recordatorio)
