# Auditoría de diseño iOS — Parte 1: Inicio, Grabar y Subir (build 78, commit 71ae403)

Fuente: código `mobile/ios/CutSell/*.swift` (solo lectura) + Figma `rXfUuJQgwZIZJTuONvU1Z7` páginas 3003:2, 3003:21, 3006:2, 3003:5863.

## Notas generales (leer primero)
- **Fuente**: `CS.inter(...)` y `CS.jakarta(...)` son **Plus Jakarta Sans** (Regular/Medium/SemiBold/Bold/ExtraBold). "ExtraBold" = `.heavy`. Solo el contador de la cámara y "Record with script" usan **SF (fuente del sistema)**.
- **El modo de color manda**: la app está forzada a **claro** (`AppPrefs.colorScheme = .light`). Los colores viejos escritos como `Color(hex: 0x…)` (diseño oscuro) se convierten solos a claro con `LightPalette` (p. ej. #0D121F→#FFFFFF, #0F1726→#F8F9FB, #33405C→#E6E9EF, #99A3BD→#6B7385). Pero la **cámara, el guion (teleprompter) y la grabación de Clon están forzadas a oscuro**, y ahí esos mismos colores salen con su valor viejo (cyan #19E6FF, #0D121F, etc.).
- `CS.cyan` = #1769FF en claro y **#19E6FF (cyan viejo) en oscuro**.
- **Botones "3D suaves"** (`softPuff`): no son un color plano. Azul = degradado #2F7BFF→#0A55F0 con base #0636B8; Negro = #2A3045→#0B1020; Blanco = #FFFFFF→#EEF3FF con base #D3DDF5 y sombra azul suave. En Figma los botones son planos (#1769FF, o blanco con borde negro).
- Textos: se muestran en inglés; entre paréntesis el español de `L10nES.swift`. "(sin ES)" = queda en inglés aunque el teléfono esté en español.

---

## A) Inicio / pestaña Create

### 1. Inicio (Create) — `HomeHubView.swift` (`HomeHubView`)
- Cómo se llega: abrir la app (pestaña por defecto) o tocar "Create" en la barra de abajo. Al final de la intro (`IntroView`) también vuelve aquí.
- Fondo: claro. `HomeBackdrop`: degradado vertical #E7EEFF → #F5F8FF → #FFFFFF + 2 manchas borrosas (#1769FF al 16 % arriba-derecha, #4A5BFF al 12 % izquierda). No es blanco puro.
- Elementos de arriba hacia abajo:
  1. Héroe animado `HomeLogoHero` (270 pt alto): bolsa "HomeBag" que rebota como gelatina + letras "CutSell.ai" (assets `HomeWord0…9`) que saltan una por una + monedas doradas "$" (#FFF3B0→#FFD34D→#F0A818, texto #A86A00) + manchas de color rosa #FFB3D9, lila #B9C8FF, #D9C2FF; sombra de piso #0D1433 16 %. No hay logo en la esquina.
  2. (Solo si la suscripción fue cancelada) banner de borrado — ver pantalla 3.
  3. "Hi there 👋" (Hola 👋) o "Hi {nombre} 👋" (Hola {nombre} 👋) / texto / #6B7385 / Plus Jakarta Sans Medium 15 / centrado.
  4. "What are we promoting" (¿Qué promocionamos) en #0B1020 + " today?" (hoy?) en #1769FF / título de dos colores / Plus Jakarta Sans ExtraBold 29 / centrado, máx. 2 líneas.
  5. "AI cuts silences, mistakes & repeats." (La IA corta silencios, errores y repeticiones.) / texto / #6B7385 / Medium 14 / centrado.
  6. Dos píldoras grandes lado a lado (58 alto, cápsula, texto blanco Bold 17, ícono 20):
     - "Record" (Grabar) / botón / 3D negro (#2A3045→#0B1020) / asset `WIRecord`.
     - "Upload" (Subir) / botón / 3D azul (#2F7BFF→#0A55F0) / asset `WIUpload`. Abre la galería directo en **Single**.
  7. Fila de 5 burbujas redondas (58 pt, círculo 3D blanco, ícono 22 #1769FF; nombre debajo SemiBold 11 #0B1020):
     - "Bulk" (Varios) `WIBulk` · "Combine" (Combinar) `WICombine` · "Sales Mix" (sin ES) `WISalesMix` · "Faceless" (sin ES) `WIMic` · "Week Plan" (Plan de la semana) `WIPlanCal`.
  8. Tarjeta "Stats" (Estadísticas) — **solo si ya hay ≥1 video editado**: título ExtraBold 19 #0B1020; reloj 3D azul `ClockFace` (74 pt, manecillas #0B1020 y #0A55F0); tiempo ahorrado ExtraBold 34 **#0A55F0** (cuenta hacia arriba, p. ej. "2h 20m"); "saved editing" (ahorrado editando) Bold 13 #6B7385; burbuja 3D rosa (#FF7DBA→#B83DFF) con el número de videos ExtraBold 23 blanco; "videos" (videos) Bold 11.5 #0B1020. Tarjeta blanca 3D, radio 32.
  9. Hoja "Recent" (Recientes): fondo blanco 85 %, borde blanco, radio 28, sombra #0D1433 6 %. "Recent" ExtraBold 19 #0B1020; "See all" (Ver todo) SemiBold 13 #1769FF a la derecha (va a la pestaña Edits). Hasta 3 miniaturas verticales 150 alto, radio 18, relleno #F3F5F9 mientras carga; si el corte se está procesando, rueda de carga #1769FF encima.
  10. Barra de abajo `CSTabBar` (ver pantalla 2).
- Estados:
  - Vacío (sin cortes): Recent muestra "Your cuts will show up here." (Tus cortes aparecerán aquí.) Medium 13 #6B7385 en caja #F8F9FB radio 18, 90 alto. Sin tarjeta Stats.
  - Cargando: miniaturas grises #F3F5F9; corte procesando = spinner azul.
  - Error: no hay estado de error en esta pantalla.
- Hojas/popups que abre: hoja "What are you promoting?" (antes de Record, Upload, Bulk, Combine, Sales Mix, Faceless); alerta Faceless; "Week Plan" NO abre hoja: salta a la pestaña Plan. Tocar una miniatura de Recent → `ProjectDetailView` (fuera de este alcance).
- Figma: 🟠 distinta — frame "Inicio" (1164:2). Diferencias exactas:
  - Figma tiene el logo horizontal arriba a la izquierda (1164:77); la app tiene el logo animado grande centrado (bolsa + letras saltando + monedas).
  - Figma: "Hi {nombre} 👋" y "What are we promoting today?" **alineados a la izquierda**, título en una sola tinta negra. App: centrados y "today?" en azul.
  - Figma: 4 tarjetas grandes 171×122 (Record "Camera + teleprompter", Faceless "Just your voice and product", Sales Mix "Multiple: hooks + bodies + CTAs for the same product", Plan my week "Post at your times") con ícono en cuadrito azul suave y subtítulo. App: 2 píldoras (Record negra, Upload azul) + 5 burbujas redondas sin subtítulos.
  - Figma: caja "Upload videos" punteada (1205:107) con círculo azul de subir, texto "AI cuts silences, mistakes & repeats." y 3 tarjetas azules Single "One video" / Bulk "Many at once" / Combine "Join into one". App: **no existe esa caja**; la frase "AI cuts silences…" quedó debajo del título y Single = botón "Upload".
  - Figma "Plan my week" → app "Week Plan".
  - App agrega la tarjeta "Stats" (no está en Figma).
  - Figma Recent: miniaturas 111×90 horizontales sobre la página; app: miniaturas 150 alto dentro de una hoja blanca redondeada.
  - Fondo Figma blanco #FFFFFF; app degradado azul claro.

### 2. Barra de pestañas — `MainTabView.swift` (`CSTabBar`)
- Cómo se llega: siempre visible abajo en Create (y en Edits/Plan/Profile).
- Fondo: cápsula blanca #FFFFFF flotante, 68 alto, borde 1 pt #E6E9EF, sombra #0D1433 10 %.
- Elementos (izq→der): "Create" (Crear) `WITabCreate` · "Edits" (Mis videos) `WITabEdits` · "Plan" (sin ES) `WITabPlan` · "Profile" (Perfil) `WITabProfile`. Íconos 24 pt; etiqueta 11 pt.
  - Activa: ícono y texto #1769FF, texto Bold 11, **burbuja cápsula #EEF3FF detrás**.
  - Inactiva: ícono #0B1020, texto Medium 11 #6B7385.
- Estados: solo activa/inactiva.
- Figma: 🟠 distinta — "Nav/White" (1164:59) y componente "Barra de abajo" (3003:5986). En Figma la pestaña activa es azul **con un puntito azul debajo** (Ellipse 1164:64) y **sin burbuja de fondo**; la app no tiene el punto y sí pone la burbuja #EEF3FF. Figma radio ~24 (rectángulo redondeado), app cápsula completa.

### 3. Aviso de borrado de cuenta — `HomeHubView.swift` (`DeletionWarningBanner`)
- Cómo se llega: aparece solo en Inicio cuando el servidor manda el aviso de suscripción cancelada.
- Fondo: tarjeta #FEF3F2, borde #FDA29B, radio 16.
- Elementos: ícono SF `exclamationmark.triangle.fill` #D92D20 + texto del servidor SemiBold 13 #7A271A (texto viene del servidor, no se traduce en la app).
- Estados: visible/oculto.
- Figma: 🔴 no existe. (Colores fuera de la paleta White Indigo: rojo #D92D20 en vez de #E5484D.)

### 4. Hoja "What are you promoting?" — `ProductPicker.swift` (`PromotingSheet`)
- Cómo se llega: Inicio → Record, Upload, Bulk, Combine, Sales Mix o Faceless (y desde guías "Try … now").
- Fondo: atenuado #0B1020 al 45 % (tocar afuera cierra); hoja blanca #FFFFFF con radio 28 arriba y manija gris #D5D9E2 40×5.
- Elementos de arriba hacia abajo:
  1. "What are you promoting?" (¿Qué vas a promocionar?) / título / #0B1020 / ExtraBold 22.
  2. Buscador 52 alto, radio 16, fondo blanco, **borde azul #1769FF 1.5 pt**; ícono `WISearch` #0B1020; placeholder "Search brand or product" (Busca marca o producto) #6B7385; texto escrito SemiBold 16 #0B1020.
  3. Lista (máx. 4 filas visibles de 52) dentro de caja radio 16 3D blanca:
     - Fila de **marca**: nombre Bold 15 #0B1020 + número de productos SemiBold 12 #6B7385 + (si su producto está elegido y cerrada) "· {producto}" SemiBold 13 #1769FF + flecha `WIChevronSmall` #6B7385 que gira 90° al abrir; línea #E6E9EF.
     - Filas de **producto** (al abrir la marca, sangría 32): nombre Medium 15 (Bold si elegido) #0B1020; elegido = fondo #EEF3FF + check `WICheck` #1769FF.
     - "+ Add product" (Agregar producto) Bold 13 #1769FF con `WIPlus`.
     - Fila fija abajo, fondo #EEF3FF: `WIPlus` + "New brand or product" (Nueva marca o producto) o, si escribió algo, "Add “{texto}” as a new product" (Agregar “{texto}” como producto nuevo) Bold 14 #1769FF.
  4. "Continue" (Continuar) / botón cápsula 52, 3D azul, Bold 16 blanco / al 40 % y desactivado hasta elegir producto.
  5. Solo en Bulk: "Mixed products? Detect for me" (¿Varios productos? Detéctalos por mí) / enlace / SemiBold 13 #1769FF.
- Estados: vacío (sin productos) = solo se ve la fila "New brand or product"; cargando = no hay indicador (la lista se rellena al llegar); con búsqueda = marcas que coinciden abiertas. No hay estado de error visible.
- Hojas/popups que abre: formulario de producto nuevo (pantalla 5). Al Continuar → cámara, galería, Sales Mix o alerta Faceless según el botón.
- No hay botón de cerrar (X): solo tocar el fondo.
- Figma: 🟠 distinta — "Hoja · What are you promoting?" (1215:50, hoja 1215:156). Diferencias: Figma muestra filas planas "CeraVe · Hydrating Cleanser" (marca gris + producto negro Bold) con flecha a la derecha; la app agrupa **por marca** (fila de marca con contador que se despliega y productos debajo con check). Figma no tiene "+ Add product" bajo la marca. Botón Continue en Figma plano #1769FF; app 3D degradado y apagado (40 %) sin producto. Fondo detrás: Figma muestra el Inicio de Figma (tarjetas), la app muestra su Inicio distinto. Textos del buscador, "Add “Cera…” as a new product", "Continue" y "Mixed products? Detect for me" iguales.

### 5. Formulario "producto nuevo" (dentro de la hoja) — `ProductPicker.swift` (`PromotingSheet.newProductForm`)
- Cómo se llega: hoja 4 → "New brand or product" / "Add “…” as a new product" / "+ Add product".
- Fondo: la misma hoja blanca.
- Elementos: título "What are you promoting?"; si vino de una marca: nombre de la marca Bold 15 #6B7385, si no campo "Brand" (Marca); campo "Product" (Producto) — campos 50 alto, radio 16, fondo blanco, borde #E6E9EF (azul #1769FF 1.5 pt al enfocar), SemiBold 16; botones lado a lado 44 alto cápsula: "Cancel" (Cancelar) 3D blanco Bold 14 #0B1020 y "Save & continue" (Guardar y seguir) 3D azul Bold 14 blanco (40 % hasta llenar ambos).
- Estados: guardando = spinner blanco en el botón. Error: no se muestra (si falla el servidor, sigue igual con los nombres).
- Figma: 🔴 no existe.

### 6. Alerta Faceless — `HomeHubView.swift` (`.alert` "Faceless video")
- Cómo se llega: Inicio → "Faceless" → hoja 4 → Continue.
- Fondo: alerta del sistema iOS (claro).
- Elementos: título "Faceless video" (Video sin cámara); mensaje "Coming soon. For now you can record a voice over inside the editor of any cut." (Próximamente. Por ahora puedes grabar una voz en off dentro del editor de cualquier corte.); botón "OK".
- Figma: 🔴 no existe (Figma solo tiene la tarjeta "Faceless — Just your voice and product").
- Nota: pide elegir producto ANTES de decir "Coming soon".

### 7. Aviso de IA antes del primer corte — `AIConsent.swift` (`AIConsentSheet`)
- Cómo se llega: la primera vez que se lanza un corte (después de "Choose your style" / "Create N cuts"), encima de todo.
- Fondo: claro #F3F5F9 (el viejo #0A1326 convertido), hoja de página completa, no se puede cerrar arrastrando.
- Elementos: "Before your first cut" (Antes de tu primer corte) Bold 22 #0B1020; "To make your cut, CutSell uses these two AI services:" (Para hacer tu corte, CutSell usa estos dos servicios de IA:) Regular 15 #6B7385; 2 viñetas con punto #1769FF: "Deepgram — …" y "Anthropic (Claude) — …" (nombre en negrita #0B1020 + texto gris) Regular 14; dos párrafos Regular 13 #6B7385; enlaces "Privacy Policy" (Política de privacidad) y "Terms of Use" (Términos de uso) SemiBold 13 #1769FF; botón "Agree" (Acepto) 52 alto **radio 16** (no cápsula) 3D azul SemiBold 16; "Not now" (Ahora no) SemiBold 14 #6B7385.
- Estados: Not now → el corte no arranca ("To make a cut, agree to the AI processing notice." en la pantalla de error).
- Figma: 🔴 no existe. Fondo gris #F3F5F9 en vez de blanco; radio 16 en vez de cápsula.

---

## B) Grabar

### 8. Permiso de cámara / micrófono — `AlertScreens.swift` (`PermissionScreen`, kind .camera / .microphone)
- Cómo se llega: Inicio → Record → Continue (hoja 4), si iOS todavía no dio permiso (o lo negó).
- Fondo: claro #FFFFFF (aunque la cámara esté en oscuro, usa colores fijos).
- Elementos: encabezado `WIHeader` con logo horizontal `LogoHorizontal` (172 pt) a la izquierda; X de cerrar (SF `xmark` 15 semibold #0B1020, círculo 3D blanco 40) **solo si ya se negó antes**; título "Camera" (Cámara) / "Microphone" (Micrófono) ExtraBold 32 #0B1020 (-2 %); tarjeta 3D blanca radio 20 con: ícono en círculo 72 #EEF3FF (cámara = asset `WIRecord` 32 #1769FF; micrófono = SF `mic` 28 #1769FF); "Allow camera access" (Permitir acceso a la cámara) / "Allow microphone access" (Permitir acceso al micrófono) ExtraBold 22 #0B1020; mensaje Medium 14 #6B7385: "CutSell.ai needs camera access so you can record video." (CutSell.ai necesita acceso a la cámara para que puedas grabar video.) / "CutSell.ai needs microphone access when you record video or voice-over." (CutSell.ai necesita acceso al micrófono cuando grabas video o voz en off.); botón cápsula 52 3D azul Bold 16: "Continue" (Continuar) o, si negado, "Open Settings" (Abrir Ajustes); "You can change this later in Settings." (Puedes cambiarlo luego en Ajustes.) Medium 12 #6B7385.
- Estados: primera vez / negado (cambia el botón y aparece X). Mientras revisa: pantalla negra #070910 (`CS.background` en oscuro).
- Nota: el texto del botón definido "Allow camera"/"Allow microphone" no se usa; el botón dice "Continue".
- Figma: 🔴 no existe en mis páginas.

### 9. Cámara — antes de grabar — `CreateCameraView.swift` (`CreateCameraView`)
- Cómo se llega: Inicio → Record → Continue (hoja 4) → (permisos OK). También desde guía "Script" (abre la hoja del guion) y desde Sales Mix → sección → "+" → "Record".
- Fondo: **oscuro natural** (video en vivo a pantalla completa; debajo `CS.screenBackground` #04060E→#090B1A→#020308). Forzado a modo oscuro.
- Elementos de arriba hacia abajo:
  1. Flecha atrás (asset `IconBack` 18×28, zona 44) arriba a la izquierda. Sin logo.
  2. Columna derecha de controles (62 pt ancho; íconos blancos; etiqueta SemiBold 13 blanca debajo):
     - "Flip" (Voltear) `IconFlip`.
     - "Flash" (sin ES) `IconFlash` / `IconFlashOn` (50 % si el teléfono no tiene flash).
     - "Grid" (Cuadrícula) `IconGrid` → líneas de tercios blancas 35 %.
     - "Mirror" (Espejo) `IconMirror`: apagado blanco; **encendido cyan #19E6FF (color viejo)**.
     - "Auto-pause" (Auto-pausa) SF `waveform` 22: encendido #1769FF, apagado blanco; en B-roll queda gris (45 %) y bloqueado. Encendido por defecto.
     - "Clon" (Clon) SF `person.2.fill` 24 blanco + etiqueta "BETA" (#1769FF, texto blanco ExtraBold 7.5, radio ~3) — solo en Record de Create y solo en iPhone compatible con AR de personas.
     - "Script" (Guion): cuadrito 36 radio 10 con "≡" Bold 16 blanco; apagado fondo negro 50 % + borde blanco 25 %; encendido fondo #1769FF y etiqueta **#4A8BFF** (no es token). Mantener apretado = abrir guion.
     - "Speed" (Velocidad) — solo con guion encendido: cápsula 54×28 negra 50 % con "1.0×" Bold 12 blanco; etiqueta SemiBold 11.
  3. (Si hay error de cámara) texto Medium 13 blanco en caja negra 55 % radio 14: "Camera and microphone permission are required to record." (Para grabar se necesita permiso de cámara y micrófono.) / "CutSell could not configure the camera." (CutSell no pudo configurar la cámara.) / "The recording could not be saved." (La grabación no se pudo guardar.).
  4. Chips de modo: "Talk" (Hablar) / "B-roll" (B-roll) — cápsula 34 alto; elegido #1769FF, Bold 15, 84 ancho; no elegido negro 35 %, SemiBold 14, 70 ancho; texto blanco. Talk por defecto.
  5. Chips de duración **en este orden: "10m" · "60s" · "3m"** (sin ES, quedan igual). Elegido: cápsula #1769FF 84×42 Bold 16 con sombra azul; los otros sin fondo SemiBold 15 blanco 54 ancho. Por defecto 60s.
  6. Botón grabar (116 pt): aro gris #45474F 104 → aro #1769FF 90 (sombra azul) → círculo #070910 78 → **círculo blanco 64**.
  7. A la izquierda del botón: "Upload" (Subir) — asset `IconUpload` 34 dentro de recuadro 54×58 radio 9 con **borde cyan #19E6FF 2 pt y brillo cyan (color viejo)**; texto Bold 14 blanco.
- Estados: revisando permisos = negro #070910; cámara lista; error (texto del punto 3).
- Hojas/popups que abre: hoja del guion (pantalla 13), Clon (pantallas 15-20), galería (Upload: cierra la cámara y abre la galería en el último modo usado, sin volver a preguntar el producto).
- Figma: 🔴 no existe en mis páginas. Colores viejos en cámara: cyan #19E6FF (Mirror encendido, borde del botón Upload), #4A8BFF (Script), #FF2E5E (cuadrado de parar), aro #45474F.

### 10. Cámara — grabando / esperando voz — `CreateCameraView.swift` (`CreateCameraView`, estado grabando)
- Cómo se llega: pantalla 9 → botón grabar (o 3·2·1 si el guion tiene cuenta atrás).
- Fondo: video en vivo (oscuro natural).
- Elementos:
  1. Flecha atrás desactivada.
  2. Píldora de tiempo arriba al centro (fondo negro 45 %, cápsula):
     - Sin auto-pausa: "00:12 / 60s" en **SF Mono Bold 15** blanco (sin ES).
     - Con auto-pausa (Talk + Auto-pause): mientras guarda voz: punto rojo #E5484D 9 pt + "REC" ExtraBold 14 #E5484D + "00:12 / 60s" SF Mono; entre frases: "Waiting for voice…" (Esperando tu voz…) SemiBold 14 **#B8BECB** (no es token) + contador (solo cuenta lo guardado).
  3. Columna de controles: **oculta**; si el guion está encendido se ve una columna corta: Flip (35 %, desactivado), Flash, Grid, Script, Speed.
  4. Chips Talk/B-roll: invisibles. Chips de duración: visibles pero bloqueados. Upload: bloqueado.
  5. Botón grabar: mismo aro con **cuadrado #FF2E5E 32 pt radio 8** (parar).
- Estados: guardando voz / esperando voz / se para solo al llegar al límite (10m/60s/3m) sin aviso; tras 60 s sin voz se para y muestra el aviso (pantalla 11).
- Figma: 🔴 no existe en mis páginas.

### 11. Aviso al parar (sin voz) — `CreateCameraView.swift` (`noticeCard`)
- Cómo se llega: al terminar una grabación con auto-pausa si se paró sola por 1 min sin voz, o si no se oyó ninguna voz.
- Fondo: negro 45 % sobre la cámara; tarjeta blanca #FFFFFF radio 20.
- Elementos: texto Bold 16 #0B1020 centrado: "Recording stopped — no voice for 1 min" (La grabación se detuvo: 1 min sin voz) o "We didn't hear any voice" (No escuchamos ninguna voz); botón "OK" 48 alto, **#1769FF plano, radio 14** (no cápsula), Bold 15 blanco.
- Estados: al tocar OK sigue igual que una grabación normal → "Choose your style".
- Figma: 🔴 no existe.

### 12. Error de grabación — `AlertScreens.swift` (`CreateProblemView` → `ProblemScreen`, caso .couldNotAdd)
- Cómo se llega: la grabación falla al guardarse.
- Fondo: claro #FFFFFF.
- Elementos: logo horizontal arriba; título "Couldn’t add this clip" (No se pudo añadir este clip) ExtraBold 32; "!" grande Bold 54 **#1769FF** (azul, no rojo); "Something went wrong" (Algo salió mal) ExtraBold 22 #0B1020; "Try recording the clip again. Your saved clips are still here." (Intenta grabar el clip otra vez. Tus clips guardados siguen aquí.) Medium 14 #6B7385; botones 270 ancho: "Back to clips" (Volver a los clips) cápsula 3D azul Bold 16; "Try again" (Intentar de nuevo) 3D blanco radio 16 SemiBold 15 #0B1020.
- Figma: 🔴 no existe en mis páginas.

### 13. Hoja del guion (teleprompter) — `Teleprompter.swift` (`TeleprompterSheet`)
- Cómo se llega: cámara → "Script" (sin guion todavía) o mantener apretado "Script", o "✎" en el guion; guía "Try teleprompter".
- Fondo: **OSCURO VIEJO** #0D121F, radio 28, manija #4D5770.
- Elementos: "Your script" (Tu guion) Bold 22 blanco; "It scrolls near the lens while you record." (Sube junto a la cámara mientras grabas.) Regular 13 #99A3BD; chip "Product: {marca · producto}" / "No product" (Producto: / Sin producto) SemiBold 12 blanco, cápsula #1F2638 30 alto (menú; bloqueado dentro de Sales Mix); caja de texto 200 alto fondo #0F1726 radio 16 con **borde degradado de marca (#19E5FF→#1769FF→#7D3BED→#FF21D6)**, placeholder "Paste or write what you'll say…" (Pega o escribe lo que vas a decir…) #99A3BD; "Paste" (Pegar) cápsula con borde #596685; "✦ Split for Sales Mix" (✦ Dividir para Sales Mix) cápsula #121A2B con borde degradado; aviso cyan #19E6FF SemiBold 12 (p. ej. "Split into Hook, Body, Benefit and CTA…"); filas "Scroll speed" (Velocidad) y "Text size" (Tamaño de letra) con "−" en círculo #1F2638, valor **cyan #19E6FF** Bold 14, "+" en círculo con degradado de marca; fila "Start after 3·2·1" (Empezar después de 3·2·1) con interruptor degradado (#33405C apagado) en caja #0F1726 borde #33405C; botón "Record with script" (Grabar con guion) 50 alto radio 25, #1769FF, texto **SF (sistema) SemiBold 15** (no Jakarta).
- Estados: botón desactivado sin texto; "Split" con spinner; errores: "Split for Sales Mix isn't available right now. Your script is saved." / "Sign in or continue to use Split for Sales Mix." (Entra o continúa para usar Dividir para Sales Mix.).
- Figma: 🔴 no existe en mis páginas. **Diseño oscuro viejo** (fondo #0D121F, cyan #19E6FF, degradado multicolor).

### 14. Guion sobre la cámara + ajustes + 3·2·1 — `Teleprompter.swift` (`TeleprompterOverlay`, `TeleprompterSettings`, `ScriptCountdown`)
- Cómo se llega: guion encendido en la cámara.
- Fondo: degradado negro 82 %→55 %→transparente arriba de la cámara.
- Elementos: líneas del guion centradas (actual Bold, otras Medium, color elegido blanco/amarillo #FFE14D/cyan #19E6FF); barra de progreso 2 pt (blanco 20 % + degradado de marca). Al tocar: marco punteado blanco 28 %, 4 esquinas blancas con **borde cyan #19E6FF**, píldoras negras 50 %: "✎", "❚❚ Auto-scroll"/"▶ Auto-scroll" (sin ES; activo en cyan #19E6FF), "⚙"; texto "Drag to move · pull a corner to resize" (Arrastra para mover · jala una esquina para cambiar el tamaño) Medium 11 blanco 80 %.
- "⚙" abre hoja "Script settings" (Ajustes del guion): oscura vieja #0D121F 330 alto: Scroll speed, Text size, "Text color" (Color de letra) con 3 círculos (borde seleccionado = degradado de marca, otros #33405C).
- 3·2·1: número Bold 96 blanco sobre negro 35 %.
- Figma: 🔴 no existe en mis páginas. Colores viejos (cyan, degradado multicolor, #0D121F).

### 15. Clon — inicio — `CloneViews.swift` (`CloneStartView`)
- Cómo se llega: cámara (Record de Create) → "Clon".
- Fondo: claro #FFFFFF.
- Elementos: X (SF `xmark` 18 bold #0B1020) arriba izquierda; "Clon" (Clon) ExtraBold 32 #0B1020 + etiqueta "BETA" (12, #1769FF); "Record yourself up to 4 times. All of you, in the same room." (Grábate hasta 4 veces. Todas tú, en el mismo lugar.) Medium 15 #6B7385; tarjeta blanca borde #E6E9EF radio 18 con 5 consejos (SF `checkmark.circle.fill` #1769FF + Medium 15 #0B1020): "Move your phone slowly — small moves are OK" (Mueve el teléfono despacio; los movimientos pequeños están bien), "Stand in a different spot each time" (Párate en un lugar distinto cada vez), "Don't walk around your clones" (No camines alrededor de tus clones), "When you finish talking, stay and react" (ES no confirmado), "Use headphones to hear your clones" (Usa audífonos para escuchar a tus clones); botón abajo "Start clone 1" (Empezar clon 1) #1769FF plano **radio 16** (no cápsula), Bold 16 blanco.
- Figma: 🔴 no existe.

### 16. Clon — grabar — `CloneViews.swift` (`CloneFlowView.recordScreen`)
- Cómo se llega: 15 → "Start clone 1", o "Add clone"/"Retake".
- Fondo: oscuro natural (cámara AR en vivo, con los clones anteriores en la escena).
- Elementos: X blanca en círculo negro 35 % (bloqueada grabando); al centro cápsula negra 45 % Bold 14 blanco: "Clone 2" (Clon 2) o "● REC 0:06 · Clone 2" (punto #E5484D); avisos en cápsula negra 50 % SemiBold 13 blanco: "Move your phone slowly to scan the room" (Mueve el teléfono despacio para escanear el lugar), "Slow down — keep the room in view" (Más despacio: mantén el lugar a la vista), "Use headphones to hear your clones"; botón grabar igual que la cámara (#45474F / #1769FF / #070910 / blanco; parar = cuadrado #FF2E5E), al 50 % hasta que el lugar esté escaneado. Guion encima si está encendido.
- Figma: 🔴 no existe.

### 17. Clon — vista previa de la toma — `CloneViews.swift` (`previewScreen`)
- Cómo se llega: 16 → parar.
- Fondo: claro #FFFFFF.
- Elementos: "Clone N" (Clon N) ExtraBold 20 #0B1020; video en bucle radio 18; fila "Retake" (Repetir) y "Add clone" (Agregar clon) — blancos, borde #E6E9EF, radio 16, Bold 16 #0B1020 ("Add clone" solo si hay menos de 4); "Finish" (Terminar) #1769FF radio 16.
- Figma: 🔴 no existe.

### 18. Clon — procesando — `CloneViews.swift` (`processingScreen`)
- Fondo: claro #FFFFFF.
- Elementos: anillo `BrandRing` (degradado #FF1F3D→#E040FB→#8B5CF6→#2F7BFF→#00E5FF, gira) con logo; "Creating your clones…" (Creando tus clones…) Bold 17 #0B1020; "NN %" Medium 13 #6B7385.
- Figma: 🔴 no existe.

### 19. Clon — resultado — `CloneViews.swift` (`resultScreen`)
- Fondo: claro #FFFFFF.
- Elementos: "Your clones" (Tus clones) ExtraBold 20 + "BETA"; video en bucle radio 18; "Retake last" (Repetir el último) blanco + "Use this video" (Usar este video) azul #1769FF, ambos radio 16.
- Al usar: vuelve a la cámara y sigue como una grabación → "Choose your style".
- Figma: 🔴 no existe (lo más parecido es "Paso 5 · Resultado final" 3006:76 de la propuesta Split, que tiene "‹ Back", "Your split video" y un solo botón "Use this video").

### 20. Clon — falló — `CloneViews.swift` (`failedScreen`)
- Fondo: claro #FFFFFF.
- Elementos: mensaje Bold 16 #0B1020: "The take could not be saved. Try again." (No se pudo guardar la toma. Inténtalo de nuevo.) / "The camera could not start. Try again." (La cámara no pudo empezar. Inténtalo de nuevo.) / "We couldn't create your clones. Try again." (No pudimos crear tus clones. Inténtalo de nuevo.); "Try again" (Intentar de nuevo) azul; "Close" (Cerrar) blanco.
- Figma: 🔴 no existe.

### 21. "Choose your style" — `StyleSheetView.swift` (`StyleSheetView` dentro de `BottomSheetOverlay`)
- Cómo se llega: al parar una grabación (o OK del aviso 11, o "Use this video" de Clon); también desde la galería con el botón "Create…/Combine…".
- Fondo: atenuado negro 45 %; hoja blanca #FFFFFF radio 28 arriba, manija #D5DAE3 40×5 (arrastrar hacia abajo o tocar afuera la cierra — **en grabación eso borra el video grabado**).
- Elementos: "Choose your style" (Elige tu estilo) ExtraBold 22 #0B1020; "You can change this later in the editor." (Puedes cambiarlo luego en el editor.) Medium 13 #6B7385; tarjeta 3D blanca radio 16 con 2 filas (65 alto, línea #E6E9EF): "Overlap audio" (Superponer audio) Bold 15 + "Smooth audio between cuts, no abrupt jumps." (Audio suave entre cortes, sin saltos bruscos.) Medium 12 gris; "Auto Zoom AI" (Auto Zoom IA) + "Zooms in on key moments automatically." (Hace zoom en los momentos clave automáticamente.); interruptor 51×31 encendido #1769FF / apagado #D5DAE3; casilla "Use this style for my next videos" (Usar este estilo en mis próximos videos) Medium 13 #0B1020, cuadrito 20 radio 6 encendido **#1677FF** (casi el azul, no exacto) con `IconCheck`, apagado borde #E6E9EF; botón grande 52 radio 26, 3D azul, SemiBold 16 blanco: "Create 1 cut" (Crear 1 corte) / "Create N cuts" (Crear N cortes) / "Combine N clips" (Combinar N clips).
- Estados: ambos interruptores encendidos por defecto (o lo guardado).
- Figma: 🔴 no existe en mis páginas.

### 22. Processing — `ProcessingScreen.swift` (`CutProgressView` → `ProcessingContent`)
- Cómo se llega: "Create 1 cut" (Single, Combine, grabación, Bulk de 1 video) → (aviso de IA la 1.ª vez) → esta pantalla (pantalla completa). También desde el banner "Your cut is ready" de otra pestaña.
- Fondo: claro, `HomeBackdrop` (degradado #E7EEFF→#FFFFFF con manchas azules), no blanco puro.
- Elementos de arriba hacia abajo:
  1. Barra: atrás `WIBack` en círculo 3D blanco 40; "Processing" (Procesando) Bold 17 #0B1020 centrado; "Edits" (Mis videos) SemiBold 15 #1769FF a la derecha.
  2. Primer cuadro del video 100×156 radio 20, borde blanco 3 pt, sombra #1769FF 28 % (sin imagen: #EEF3FF + logo `LogoIcon`).
  3. Nombre del corte Medium 13 #6B7385 (vacío si es un nombre técnico tipo "IMG_1234.MOV").
  4. "Hold on…" (Un momento…) ExtraBold 30 #0B1020; "We’re saving you hours per day…" (Te estamos ahorrando horas al día…) Medium 15 #6B7385.
  5. Anillo `BrandRing` 150 girando (degradado #FF1F3D→#E040FB→#8B5CF6→#2F7BFF→#00E5FF con brillo) + logo `LogoIcon`.
  6. Paso Bold 17 #0B1020 + porcentaje ExtraBold 26 #1769FF (o "…" si no hay %). Pasos posibles: "Getting your video ready" (Preparando tu video), "Getting your video from Photos" (Trayendo tu video de Fotos), "Downloading from iCloud" (Descargando de iCloud), "Creating your project" (Creando tu proyecto), "Preparing…" (Preparando…), "Uploading audio…" (Subiendo audio…), "Starting your cut" (Empezando tu corte), "Cutting your video…" (Cortando tu video…), "Cutting your video… listening" (… escuchando), "… choosing the best takes" (… eligiendo las mejores tomas), "… building the edit" (… armando la edición), "… almost done" (… casi listo). En Bulk se antepone "Cut 2 of 5 · " (Corte 2 de 5).
  7. (Opcional) línea de subida en segundo plano Medium 13 gris; (opcional) aviso de subida trabada Medium 13 **#B26A00** + "Try again" (Intentar de nuevo) Bold 15 #1769FF.
  8. Nota: `WIInfo` 18 #0B1020 + "Your video keeps uploading if you leave this screen or CutSell." (Tu video sigue subiendo aunque salgas de esta pantalla o de CutSell.) o "Processing continues if you leave this screen." (El procesamiento sigue aunque salgas de esta pantalla.) Medium 13 #0B1020, tarjeta 3D blanca radio 18, 52 alto.
  9. "Continue in background" (Seguir en segundo plano) cápsula 52 **negra plana #0B1020**, Bold 16 blanco.
  10. "We’ll notify you when your cut is ready." (Te avisamos cuando tu corte esté listo.) Medium 12 #6B7385.
- Estados: cargando (los pasos); en cola (pantalla 23); error (pantalla 24); éxito → Ready (pantalla 25).
- Popups: "Upload paused" / "Couldn’t start the cut" (pantalla 26); problemas de archivo (pantalla 32).
- Figma: 🟠 distinta — "Single · Processing" (1179:2). Diferencias: fondo Figma blanco, app degradado azul; Figma muestra el nombre de archivo "cutsell01001902848.mp4", la app oculta nombres técnicos (muestra el nombre del corte o nada); Figma paso "Structuring your cut 68%", la app usa "Cutting your video…" + sub-pasos (no existe "Structuring your cut"). La parte de abajo del frame de Figma está tapada por una nota amarilla ("Animación del ring"), así que nota/botón inferior: no confirmado. Anillo, "Hold on…", subtítulo, barra con "Edits" azul: iguales.

### 23. Processing — en cola — `ProcessingScreen.swift` (`ProcessingContent`, queued)
- Cómo se llega: el servidor tiene el corte pero aún no lo empieza.
- Fondo: igual que 22.
- Elementos: título de barra "Queued" (En cola); "Your cut is next in line" (Tu corte es el siguiente) Bold 24 **#4A5263** (gris fuera de token); anillo; "Waiting to start" (Esperando para empezar) Bold 17; "0%" ExtraBold 26 #1769FF; "Waiting for a processing slot" (Esperando un espacio para procesar) Medium 13 gris; nota "We’ll start automatically when a slot opens." (Empezaremos solos cuando haya un espacio.) en caja 3D blanca radio 14; botón "Continue in background" **azul plano #1769FF**; "We’ll notify you when processing begins." (Te avisaremos cuando empiece el procesamiento.).
- Figma: 🔴 no existe en mis páginas.

### 24. Processing — error — `ProcessingScreen.swift` (`ProcessingContent`, failed)
- Cómo se llega: el servidor falla el corte.
- Fondo: igual que 22.
- Elementos: barra "Processing issue" (Problema al procesar); miniatura; "We couldn’t finish this cut" (No pudimos terminar este corte) ExtraBold 22 #4A5263; anillo quieto con insignia roja "!" (#E5484D, borde blanco); "Processing stopped" (El procesamiento se detuvo) SemiBold 15 #E5484D; "Error" (sin ES) ExtraBold 22 #E5484D; "Your original recording is safe" (Tu grabación original está a salvo) Medium 12 gris. Variante sin voz: "We couldn’t hear anyone talking" (No escuchamos a nadie hablando) ExtraBold 19 #E5484D + "CutSell edits videos where someone speaks. Try a video with your voice." (CutSell edita videos donde alguien habla. Prueba con un video con tu voz.). Abajo: caja 3D blanca "Your original video is safe. Nothing was lost." (**sin ES**, queda en inglés) o, si el reintento falla, "Couldn’t retry this cut. Please upload it again." (No se pudo reintentar este corte. Vuelve a subirlo.) en rojo; "Retry processing" (Reintentar procesamiento) cápsula 3D azul (spinner mientras reintenta); "Try again—the engine will resume from this stage." (Inténtalo de nuevo: el motor seguirá desde esta etapa.) Medium 12 gris.
- Figma: 🔴 no existe en mis páginas.

### 25. Ready — `ProcessingScreen.swift` (`ReadyContent`)
- Cómo se llega: Processing termina (misma pantalla cambia sola) o banner/toast "… is ready · View".
- Fondo: igual que 22 (degradado azul claro), con scroll.
- Elementos: atrás `WIBack` (círculo 3D blanco 40); primer cuadro 132×206 radio 20 con botón play `WIPlay` blanco en círculo #0B1020 55 % (tocar = editor); "Your cut is ready" (Tu corte está listo) ExtraBold 26 #0B1020; nombre del corte Medium 13 gris; si hay duración del material: "YOU JUST SAVED" (ACABAS DE AHORRAR) Bold 11 #1769FF espaciado, tiempo ExtraBold 54 #0B1020 (p. ej. "13:46"), "of editing time back in your day" (de edición que recuperas en tu día) Medium 14 gris; tarjeta 3D blanca 96 alto con 3 columnas separadas por líneas #E6E9EF: íconos `WIStatFootage` / `WIStatTrims` / `WIStatFinal` 20 #1769FF, valor ExtraBold 20, etiqueta Medium 12 gris: "your footage" (tu material), "clips kept" (clips conservados), "your final cut" (tu corte final) ("–" mientras carga); tarjeta azul (degradado plano #1769FF) radio 20, 84 alto: reloj `WIClockWhite` en círculo blanco 18 %, total ExtraBold 24 blanco, "Your total editing time saved" (Tu tiempo total de edición ahorrado) Medium 12 blanco 85 %, insignia "5+ WORKDAYS" (5+ JORNADAS) Bold 10 en cápsula blanca 20 %; botones: "Review & edit" (Revisar y editar) cápsula 3D blanca Bold 15 #0B1020 y "Export video" (Exportar video) cápsula 3D azul Bold 15 blanco (**los dos abren el editor**).
- Estados: números "–" mientras carga; bloque "YOU JUST SAVED" y tarjeta azul solo si hay datos.
- Figma: 🟠 distinta — "Single · Ready" (1179:30). Diferencias: etiqueta del centro Figma "smart trims" (ícono tijeras) → app "clips kept"; botón "Review & edit" en Figma blanco plano **con borde negro** (Botón/Secundario 3003:5998) → app blanco 3D sin borde negro; "Export video" Figma azul plano → app azul 3D degradado; Figma muestra nombre de archivo "cutsell01001902848.mp4" → app nombre del corte (o nada); fondo Figma blanco → app degradado azul. Orden, textos y tarjeta azul "42h 18m / 5+ WORKDAYS": iguales. Íconos de las stats: no confirmado si son los mismos dibujos.

### 26. "Upload paused" / "Couldn’t start the cut" — `AlertScreens.swift` (`ProblemScreen`, desde `CreateFlowView`)
- Cómo se llega: falla la subida durante Processing.
- Fondo: claro #FFFFFF.
- Elementos: logo arriba; título ExtraBold 32 "Upload paused" (Subida en pausa) o "Couldn’t start the cut" (No se pudo iniciar el corte); "!" Bold 54 #1769FF; "The upload stopped" (La subida se detuvo) / "The upload didn’t start" (La subida no empezó) ExtraBold 22; mensaje = error real + "Your video is saved on this iPhone. Try again to continue where it stopped." o "Check your connection and try again. Your video is still here." (traducción no confirmada porque el texto se arma con el error); "Try again" (Intentar de nuevo) cápsula azul; "Go to Edits" (Ir a Mis videos) o "Cancel" (Cancelar) blanco 3D radio 16.
- Figma: 🔴 no existe en mis páginas.

### 27. Avisos "listo" en cualquier pestaña — `MainTabView.swift` / `ProcessingQueue.swift` (`ReadyBanner`, `ReadyToast`, `GroupReadyToast`)
- Cómo se llega: un corte termina mientras se está en otra parte (tras "Continue in background").
- `ReadyBanner` (arriba): tarjeta #F8F9FB radio 18 con **borde #1769FF 50 % y brillo azul**; logo `ProcessingLogo` con mezcla "screen" (sobre fondo claro casi no se ve — posible problema, no confirmado en pantalla); "Your cut is ready" (Tu corte está listo) SemiBold 15 #0B1020; "Tap to review it" (Toca para revisarlo) Regular 12 #6B7385; X SF `xmark` #6B7385. Toque → Processing/Ready.
- `ReadyToast` / `GroupReadyToast` (abajo, sobre la barra): tarjeta blanca radio 14, check SF verde #1FA971 en círculo #EAF8F1, "{nombre} is ready" ({nombre} está listo) SemiBold 14 #0B1020, "View" (Ver) Bold 14 #1769FF.
- Figma: 🔴 no existe en mis páginas (el código cita 1134:3027 / 1213:2 de otro archivo).

---

## C) Subir

### 28. Galería (Single / Bulk / Combine) — `GalleryPickerView.swift` (`GalleryPickerView`)
- Cómo se llega: Inicio → "Upload" (Single) / "Bulk" / "Combine" → hoja 4 → Continue; o cámara → "Upload"; o Plan → Bulk.
- Fondo: claro #FFFFFF (colores viejos convertidos).
- Elementos de arriba hacia abajo:
  1. X (SF `xmark` 18 semibold #0B1020) a la izquierda; "Recents" (Recientes) SemiBold 22 #0B1020 centrado; línea #E6E9EF.
  2. (Acceso limitado) aviso: "CutSell sees only the videos you picked." (CutSell solo ve los videos que elegiste.) Medium 13 #0B1020 + enlaces "Select more photos" (Elegir más fotos) y "Change permission" (Cambiar permiso) SemiBold 13 #1769FF; fondo casi blanco (el viejo #0A1A36 convertido ≈ #FFFFFF, no se distingue del fondo).
  3. Cuadrícula de 3 columnas (separación 6), cuadros radio 12, relleno #F3F5F9 mientras carga; duración abajo-derecha SemiBold 12 blanco con sombra. Elegido: borde #1769FF 2 pt + círculo arriba-derecha 24 pt **#0D9EFF con borde cyan #1AF2FF (colores viejos)** con número SemiBold 12 **#0B1020 (texto oscuro sobre azul)** en Single/Combine, o `IconCheck` en Bulk.
  4. Chip "B-roll" abajo-izquierda en cada clip elegido: apagado cápsula blanca borde #E6E9EF, SF `film` + "B-roll" Bold 11 #0B1020; encendido #1769FF, SF `checkmark`, texto blanco; 24 alto.
  5. Panel de abajo (#FFFFFF): "1 video selected" / "3 videos selected" (3 videos seleccionados) / "3 clips selected" (**sin ES** para clips) Regular 14 #6B7385; selector "Single · Bulk · Combine" (Uno · Varios · Combinar) fondo #F3F5F9 radio 12 borde #E6E9EF, elegido #1769FF **radio 10 (no cápsula)** texto blanco SemiBold 13; ayuda Regular 13 #6B7385: "One video → one edit." (Un video → una edición.) / "Each video becomes a separate edit." (Cada video se vuelve una edición aparte.) / "One edit in the order you selected." (Una edición en el orden que elegiste.); botón 52 radio 26 3D azul SemiBold 16: "Create 1 cut" / "Create N cuts" / "Combine N clips" (al 45 % sin nada elegido).
- Estados: cargando permiso = `ProgressView` **blanco sobre fondo blanco (no se ve)**; vacío = "No videos in your library yet." (Todavía no hay videos en tu galería.) Regular 14 #6B7385; sin permiso = pantalla 30; límite = pantalla 31.
- Hojas/popups: "Choose your style" (21); tarjeta de acceso limitado (29); problemas (31).
- Figma: 🔴 no existe en mis páginas. Restos viejos: círculo #0D9EFF/#1AF2FF.

### 29. Tarjeta "Gallery access" (acceso limitado) — `AlertScreens.swift` (`GalleryAccessCard`)
- Cómo se llega: abrir la galería con acceso "Limitado".
- Fondo: atenuado #0B1020 45 %; tarjeta blanca 322 ancho radio 28.
- Elementos: "Gallery  access" (Acceso a la galería) ExtraBold 22 (en inglés tiene **doble espacio**); "Choose videos from your library to create a CutSell edit." (Elige videos de tu galería para crear una edición de CutSell.) Medium 14 gris; "Select More Photos…" (Elegir más fotos…) cápsula 3D azul; "Keep Current Selection" (Mantener selección actual) cápsula 3D blanca.
- Figma: 🔴 no existe en mis páginas.

### 30. Galería sin permiso — `AlertScreens.swift` (`PermissionScreen`, kind .photos)
- Cómo se llega: galería con permiso negado.
- Fondo: #FFFFFF. Elementos: logo; X (cerrar); "Gallery" (Galería) ExtraBold 32; círculo #EEF3FF con `WIUpload` #1769FF; "Allow gallery access" (Permitir acceso a la galería); "Choose videos from your library to create a CutSell edit."; "Open Settings" (Abrir Ajustes); "You can change this later in Settings.".
- Figma: 🔴 no existe en mis páginas.

### 31. Problemas al elegir/subir — `AlertScreens.swift` (`CreateProblemView` → `ProblemScreen` / `LimitScreen`)
- Cómo se llega: galería (11.º clip) o al exportar de Fotos.
- Fondo: #FFFFFF (ProblemScreen) o hoja blanca sobre atenuado 45 % (LimitScreen).
- Variantes (todas: título ExtraBold 32, "!" azul Bold 54, encabezado ExtraBold 22, texto Medium 14 gris, botón azul + botón blanco):
  - "Clip limit reached" (Llegaste al límite de clips) / "10 clips maximum" (Máximo 10 clips) / "Process these clips or remove one before adding another." / "Review clips" (Revisar clips) + "Remove a clip" (Quitar un clip).
  - "Can’t add this video" (No se puede añadir este video) / "Unsupported media" (Archivo no compatible) / "Back to library" (Volver a la galería) + "View supported formats" (Ver formatos compatibles) → "Supported video files" (Videos compatibles) / "Formats and limits" / "MP4, MOV and M4V…" / "Choose a video" + "Back".
  - "Couldn’t add this clip" (archivo no disponible).
  - `LimitScreen` "Upload limit exceeded" (Se superó el límite de subida): insignia aviso (círculo #FFF4E0, `WIWarn` #F5A524), tarjeta "Selected video" (Video seleccionado) con tamaño en rojo #E5484D, botón "Choose another video" (Elegir otro video).
- Figma: 🔴 no existe en mis páginas.

### 32. Hoja "Your N videos are being edited" (Bulk) — `ProcessingQueue.swift` (`BulkStartedSheet`)
- Cómo se llega: galería en Bulk con 2+ videos → "Create N cuts" → estilo → (aviso IA). Bulk NO muestra la pantalla Processing.
- Fondo: atenuado negro 60 %; hoja 400 alto radio 28 — colores viejos convertidos a claro: fondo #FFFFFF, borde #E6E9EF, manija #D5DAE3.
- Elementos: "Your {N} videos are being edited" (Tus {N} videos se están editando) Bold 22 #0B1020; "We'll notify you when your videos are ready. You can keep using CutSell." (Te avisamos cuando tus videos estén listos. Puedes seguir usando CutSell.) Regular 14 #6B7385; dos tarjetas #F8F9FB radio 16 con "→" entre ellas: "Processing" (Procesando) SemiBold 15 #1769FF con borde #1769FF + "In Edits, in order" (En Mis videos, en orden); "Ready" (Listos) SemiBold 15 **blanco #FFFFFF sobre #F8F9FB → casi invisible (bug)** + "Each one, as soon as it’s done" (Cada uno, apenas termine); "One notification when the whole group is ready." (Un solo aviso cuando todo el grupo esté listo.) Regular 12 gris; "Go to Edits" (Ir a Mis videos) cápsula 3D azul; "Stay here" (Quedarme aquí) SemiBold 15 #0B1020.
- Figma: 🔴 no existe en mis páginas. Hecha con tokens del diseño oscuro (se ve clara solo por la conversión automática).

### 33. Sales Mix (armado) — `MixView.swift` (`MixSetupView`)
- Cómo se llega: Inicio → burbuja "Sales Mix" → hoja 4 → Continue (pantalla completa).
- Fondo: claro #FFFFFF (degradado oscuro viejo #03050D→#050B1C→#030409 convertido a blanco). **Diseño oscuro viejo convertido automáticamente**, no rediseñado en White Indigo.
- Elementos de arriba hacia abajo:
  1. "✕" Medium 20 #0B1020 a la izquierda; "Sales Mix" (sin ES) Bold 18 #0B1020 centrado.
  2. "Record a few versions of each part. CutSell cleans every clip once and builds every combination for you." (Graba varias versiones de cada parte. CutSell limpia cada clip una vez y arma todas las combinaciones por ti.) Regular 13 #6B7385.
  3. Tarjeta producto (54 alto, #F8F9FB, borde #E6E9EF, radio 14): "PRODUCT" (PRODUCTO) Bold 10 #1769FF; "{marca} · {producto}" o "Choose the product" (**no se traduce**, el código no lo pasa por L10n) SemiBold 15 #0B1020; "Choose"/"Change" (Elegir/Cambiar) SemiBold 13 #1769FF.
  4. 4 tarjetas de sección (#F8F9FB, borde #E6E9EF, radio 16): número en círculo 24 #1769FF Bold 12 blanco; nombre "Hook" / "Body" / "Benefit" / "CTA" SemiBold 15 #0B1020 (**no se traducen**: el nombre se muestra tal cual); "0 clips" (0 clips) Regular 12 gris; menú "≡" (Move up/Move down = Subir/Bajar) y "•••" (Rename/Delete section = Cambiar nombre/Eliminar sección) en #6B7385; fila de miniaturas 42×46 radio 8 (#F3F5F9 + duración) y cuadro "+" #1769FF con borde punteado #C3C9D7.
  5. "+  Add section" (+  Añadir sección) SemiBold 14 #6B7385, borde punteado #C3C9D7 radio 14.
  6. Barra fija abajo (#F7F8FC, línea #CDD3E2): "Add clips to every part" (Añade clips a cada parte) o "N videos · about S s each" (N videos · unos S s cada uno) SemiBold 15 #0B1020; fórmula "3 hooks × 2 bodies…" Regular 11 gris; "Pick how many ›" (Elegir cuántos ›) SemiBold 13 #1769FF; "Make N videos" (Hacer N videos) cápsula 50 **#1769FF plana**, SemiBold 15 blanco (50 % si falta algo).
- Popups: diálogo del sistema "Add clips" (Añadir clips): "Choose from Photos" (Elegir de Fotos) → galería Mix (pantalla 28 sin selector de modo, sin chips B-roll y sin hoja de estilo; botón "Add 1 clip"/"Add N clips" = Añadir N clips) / "Record" (Grabar) → cámara (pantalla 9 sin botón Clon, con el guion de esa sección) / "Cancel"; alerta "Rename section" (campo "Name", "Save"/"Cancel"); hoja "How many videos?" (34); hoja de producto (35).
- Estados: agregando = negro 50 % + "Getting your clips…" (Trayendo tus clips…) con spinner #1769FF; enviando = negro 70 % + anillo 84 + paso (p. ej. "Preparing your Sales Mix…" = Preparando tu Sales Mix…) SemiBold 14 blanco + "Keep CutSell open until the audio is sent…" Regular 12 **#6B7385 sobre negro (poco contraste)**; error = alerta "CutSell" con "Your Sales Mix couldn’t start. Check your connection and try again." o "One clip couldn’t be added. Try another one.".
- Al empezar: va directo a la pestaña Edits · Processing (no usa la pantalla Processing).
- Figma: 🔴 no existe (solo la tarjeta "Sales Mix" del Inicio de Figma).

### 34. Hoja "How many videos?" — `MixView.swift` (`MixHowManySheet`)
- Cómo se llega: Sales Mix → "Pick how many ›".
- Fondo: atenuado negro 60 %; hoja #FFFFFF (viejo #0D121F convertido), borde #E6E9EF, radio 28, manija #D5DAE3.
- Elementos: "How many videos?" (¿Cuántos videos?) Bold 22 #0B1020; "{N} combinations are possible" (Se pueden hacer {N} combinaciones) Regular 13 gris; píldoras "6" "12" "18" "All N" (Todo N) 40 alto: elegida #1769FF texto blanco; **no elegidas fondo #F8F9FB con texto BLANCO → invisible (bug)**; "We mix the clips so each video starts and ends differently." (Mezclamos los clips para que cada video empiece y termine distinto.) Regular 12 gris; "PREVIEW" (VISTA PREVIA) Bold 10 #1769FF; filas "Video 1" (Video 1) SemiBold 13 #0B1020 + nombres 12 gris, línea #CED5E4; "+ N more" (+ N más); consejo "Tip: post these over several days…" (Consejo: publícalos en varios días…) Medium 11 **naranja #FFB840 sobre #FDFBFA con borde #E9D8BD (poco contraste)**; "Make N videos" (Hacer N videos) cápsula #1769FF.
- Figma: 🔴 no existe.

### 35. Hoja de producto del Sales Mix — `MixView.swift` (`MixProductPicker`)
- Cómo se llega: Sales Mix → tarjeta de producto "Choose/Change".
- Fondo: hoja del sistema, fondo #FFFFFF (520 alto o grande).
- Elementos: "Which product is this Sales Mix for?" (¿Para qué producto es este Sales Mix?) Bold 20 #0B1020; filas 48 alto: imagen `ProductsRadio` + punto #1769FF si elegido, "{marca} · {producto}" Regular 15 #0B1020, línea #CED5E4; "+  New brand or product" (+  Nueva marca o producto) SemiBold 15 #1769FF → alerta del sistema con campos "Brand (e.g. CeraVe)" / "Product (e.g. Hydrating Cleanser)" y "Use it" (Usarlo) / "Cancel"; error "Couldn’t add it. Please try again.".
- Figma: 🔴 no existe. (Es distinta de la hoja White Indigo "What are you promoting?" — dos selectores de producto diferentes en el mismo flujo.)

### 36. Faceless (Subir sin cara)
- No es una pantalla: ver alerta 6 ("Coming soon"). `HomeHubView.facelessVideoReady = false`.
- Figma: 🔴 (solo la tarjeta del Inicio de Figma).

---

## Pantallas en DISEÑO OSCURO VIEJO o con colores viejos
- **Oscuro viejo de verdad (se ven oscuras)**: Hoja del guion `TeleprompterSheet` (#0D121F, cyan #19E6FF, degradado multicolor, botón en fuente SF) y `TeleprompterSettings` (#0D121F).
- **Hechas con tokens oscuros viejos y aclaradas solas** (no rediseñadas, contrastes rotos): `MixSetupView`, `MixHowManySheet` (píldoras con texto blanco invisible), `MixProductPicker`, `BulkStartedSheet` ("Ready" blanco invisible), `AIConsentSheet` (fondo gris #F3F5F9), aviso de acceso limitado de la galería (fondo invisible), spinner blanco de la galería.
- **Cámara** (oscuro natural por el video, correcto) pero con colores viejos: cyan **#19E6FF** en "Mirror" encendido y en el borde/brillo del botón "Upload"; **#4A8BFF** en "Script"; **#FF2E5E** en el cuadrado de parar (Clon también); aro #45474F; guion: esquinas y "Auto-scroll" cyan #19E6FF, barra de progreso con degradado multicolor; "Waiting for voice…" #B8BECB.
- Galería: círculo de número #0D9EFF con borde cyan #1AF2FF.
- Otros colores fuera de la paleta: #0A55F0 (número de Stats y relojes), #1677FF (casilla), #4A5263 (títulos en cola/error), #B26A00 (subida trabada), #D92D20/#7A271A/#FEF3F2 (banner de borrado), rosa/lila del logo animado.

---

## Navegación del flujo
1. App → Inicio (Create) → "Record" → Hoja "What are you promoting?" → "Continue" (o "Save & continue") → [Permiso Cámara → "Continue" → Permiso Micrófono → "Continue"] → Cámara → botón grabar → Cámara grabando → botón parar → [Aviso "We didn't hear any voice"/"Recording stopped — no voice for 1 min" → "OK"] → "Choose your style" → "Create 1 cut" → [Aviso IA "Before your first cut" → "Agree"] → Processing → (sola) Ready → "Review & edit" / "Export video" → Editor.
2. Cámara → "Script" → Hoja "Your script" → "Record with script" → [3·2·1] → Cámara grabando con guion.
3. Cámara → "Clon" → Clon inicio → "Start clone 1" → Clon grabar → parar → Vista previa → "Add clone" → Clon grabar … → "Finish" → Creando clones → Resultado → "Use this video" → "Choose your style" → "Create 1 cut" → Processing → Ready.
4. Cámara → "Upload" → Galería (último modo usado) → …(como 5).
5. Inicio → "Upload" → Hoja "What are you promoting?" → "Continue" → Galería (Single) → tocar video → "Create 1 cut" → "Choose your style" → "Create 1 cut" → [Aviso IA] → Processing → Ready.
6. Inicio → "Bulk" → Hoja (con "Mixed products? Detect for me") → Galería (Bulk) → elegir 2+ → "Create N cuts" → "Choose your style" → "Create N cuts" → Hoja "Your N videos are being edited" → "Go to Edits" (pestaña Edits · Processing) o "Stay here".
7. Inicio → "Combine" → Hoja → Galería (Combine, números en orden) → "Combine N clips" → "Choose your style" → "Combine N clips" → Processing → Ready.
8. Inicio → "Sales Mix" → Hoja → Sales Mix → tarjeta producto → Hoja de producto; sección "+" → "Add clips" → "Choose from Photos" (Galería Mix → "Add N clips") o "Record" (Cámara sin Clon) → "Make N videos" (o "Pick how many ›" → "How many videos?" → "Make N videos") → pestaña Edits · Processing.
9. Inicio → "Faceless" → Hoja → "Continue" → Alerta "Faceless video / Coming soon" → "OK".
10. Inicio → "Week Plan" → pestaña Plan (fuera de este alcance).
11. Processing → "Continue in background" o atrás → Inicio; luego banner "Your cut is ready" / toast "… is ready · View" → Ready.
12. Processing → "Edits" (arriba a la derecha) → pestaña Edits · Processing.

## Frames de Figma de mis páginas que NO existen en la app
- "Upload/Dropzone" (1205:107) con "Upload videos" y tarjetas "Tile/Single" (1205:115), "Tile/Bulk" (1205:122), "Tile/Combine" (1205:130) — dentro de "Inicio" 1164:2.
- Tarjetas del Inicio con subtítulo: "Tile/Record" (1164:40), "Tile/Faceless" (1195:115), "Tile/Sales Mix" (1195:103), "Tile/Plan my week" (1164:47) — en la app son píldoras/burbujas sin subtítulo.
- Logo fijo "Logo/Official · Horizontal" (1164:77) en el Inicio (la app usa el logo animado).
- Página 09 · Record Split (propuesta, la app no tiene Split): "Paso 1 · Escoger Split" (3006:3), "Paso 2 · Grabar parte 1" (3006:25), "Paso 3 · Parte 1 lista" (3006:41), "Paso 4 · Grabar parte 2 (viendo la parte 1)" (3006:60), "Paso 5 · Resultado final" (3006:76).
- Componentes de 00 sin uso igual en la app: "Volver / Editor (cuadrito)" (3003:6008) no se usa en estas pantallas; "Botón/Secundario" con borde negro (3003:5998) no se usa (la app usa blanco 3D sin borde).

## No se usan (código no alcanzable desde la navegación real)
- `ProcessingView.swift` (`ProcessingView`) — nadie la abre.
- `StageDots` (`ProcessingScreen.swift`) — definida, sin uso.
- `OptionCard`, `FilterChip` (`Theme.swift`) — definidas, sin uso en estos flujos (no se encontró uso fuera de Theme).
- `PermissionScreen.Kind.buttonTitle` ("Allow camera" / "Allow microphone" / "Allow gallery") — textos nunca mostrados.
- `CreateProblem.tooLong` ("This video is too long") — inalcanzable porque `CreateLimits.sizeAndLengthLimitsOn = false`.
- `HomeHubView.facelessVideoReady = false` — no hay pantalla Faceless.
- (`NewCutView` / `CameraCaptureView` sí se usan, pero solo desde Edits para reanudar subidas: fuera de este alcance.)

## Conteo
- ✅ igual: **0**
- 🟠 distinta: **5** (Inicio 1164:2, Barra de pestañas 1164:59, Hoja "What are you promoting?" 1215:50, Processing 1179:2, Ready 1179:30)
- 🔴 no existe en mis páginas de Figma: **31** (pantallas 3, 5-21, 23, 24, 26-36)

## Errores visuales encontrados de paso (para la lista de arreglos)
1. `BulkStartedSheet`: "Ready" en blanco sobre #F8F9FB (invisible).
2. `MixHowManySheet`: píldoras no elegidas con texto blanco sobre #F8F9FB (invisible).
3. Galería: spinner blanco sobre blanco al pedir permiso; aviso de acceso limitado sin fondo visible.
4. Duraciones de la cámara en orden raro "10m · 60s · 3m".
5. Sin ES: "Sales Mix", "Faceless", "Plan", "Flash", "Error", "Your original video is safe. Nothing was lost.", "3 clips selected", "Auto-scroll", nombres de secciones "Hook/Body/Benefit/CTA", "Choose the product".
6. Cerrar "Choose your style" tras grabar borra la grabación sin preguntar.
7. "Review & edit" y "Export video" en Ready hacen lo mismo (abren el editor).
