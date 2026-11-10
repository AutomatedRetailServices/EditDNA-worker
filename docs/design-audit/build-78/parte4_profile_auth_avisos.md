# Auditoría de diseño — Parte 4: Arranque, onboarding, barra de pestañas, Perfil, errores y avisos
Código: `/home/claude/buildclon` · rama `feature/clon-ar-autopausa` · commit `71ae403` (TestFlight build 78) · solo lectura.
Figma: archivo `rXfUuJQgwZIZJTuONvU1Z7`, páginas 3003:59 "08 · Profile", 3003:48 "07 · Errores y avisos", 3003:5863 "00 · Componentes y colores".

## 0. Reglas de color que hay que saber para leer este informe
- La app está **forzada a modo claro** (`AppPrefs.colorScheme = .light`, `LightPalette.enabled = true`). Por eso todo `Color(hex:)` / `CS.xxx` "oscuro" se pinta con su equivalente claro de `LightPalette`. Los hex de abajo ya son los que se ven en pantalla.
  - `0x050814 / 0x070910 / 0x03050B` → **#FFFFFF** · `0x0C1730 / 0x0F1726 / 0x141C2E` → **#F8F9FB** · `0x0A1326 / 0x1F2638` → **#F3F5F9** · `0x19345F / 0x33405C` → **#E6E9EF** · `0xF7FAFF / Color.ink` → **#0B1020** · `0xAFC4E8 / 0x6F86AE / 0x99A3BD` (textSecondaryF / textMutedF) → **#6B7385** · `CS.cyan / CS.link` → **#1769FF**.
  - Colores "vivos" NO cambian: `CS.error` = **#FF4D67** (no #E5484D), `0x0E46D9`, `0xFFB020`.
  - Excepción: `CreateCameraView` usa `.preferredColorScheme(.dark)` → su estado "checking" pinta `CS.background` = **#070910** (negro).
- Fuente: `CS.inter(...)` = **Plus Jakarta Sans** (Regular/Medium/SemiBold/Bold/ExtraBold). `.heavy` = ExtraBold. Donde dice "SF Pro" es fuente del sistema (NO White Indigo).
- `softPuff(...)` (botones y tarjetas "3D bubbly"):
  - `.blue`: degradado **#2F7BFF → #0A55F0**, base **#0636B8** 2 pt debajo, sombra #0A55F0 ~19 %, radio 7, y 4.
  - `.white`: degradado **#FFFFFF → #EEF3FF**, base **#D3DDF5** 2 pt, sombra #1769FF ~9 %.
  - `.ink`: **#2A3045 → #0B1020**, base #000000.
  - Figma dibuja todo esto **plano** (botón #1769FF liso, tarjeta #F8F9FB + borde 1 pt #E6E9EF). Esta diferencia se repite en casi todas las pantallas; la marco como "3D vs plano".
- `wiCard()` con relleno por defecto = `softPuff(.white)` radio ≥ 16 (3D, sin borde). Con relleno propio = plano con borde 1 pt y sombra #0D1433 8 %.
- `HomeBackdrop` (fondo de casi todas las pantallas claras): degradado vertical **#E7EEFF → #F5F8FF → #FFFFFF** + 2 brillos difuminados (#1769FF 16 %, #4A5BFF 12 %). Figma usa blanco liso #FFFFFF.

---

## A. ARRANQUE Y ONBOARDING

### 1. Pantalla de lanzamiento (Launch Screen) — `project.yml` (`INFOPLIST_KEY_UILaunchScreen_Generation: YES`)
- Cómo se llega: al tocar el ícono, lo dibuja iOS antes de que corra el código.
- Fondo: generado por iOS sin color ni imagen → color de sistema (blanco en modo claro; **negro si el iPhone está en modo oscuro** — no confirmado en dispositivo).
- Elementos: ninguno (no hay logo).
- Estados: —
- Hojas/popups: —
- Figma: 🔴 no existe.

### 2. Splash de carga (bootstrap) — `CutSellApp.swift` (`RootView`, rama `isBootstrapping`)
- Cómo se llega: siempre al abrir, mientras `appState.bootstrap()` corre.
- Fondo: claro, `ScreenBackground` = `HomeBackdrop` (#E7EEFF→#FFFFFF + brillos).
- Elementos (centrados, separación 18):
  1. Logo "CutSell.ai" — asset `LogoHorizontal`, 172 pt de ancho.
  2. Anillo de espera `BrandRing(size: 72)` girando (1 vuelta / 2,5 s): degradado #FF1F3D → #E040FB → #8B5CF6 → #2F7BFF → #00E5FF → #FF1F3D con brillo neón #00E5FF.
- Encima aparece el **diálogo de sistema de notificaciones** ("“CutSell” Would Like to Send You Notifications" — texto de iOS), porque `notificationCenter.requestPermission()` se llama ANTES de `bootstrap()`. El splash se queda esperando la respuesta. No hay pantalla propia previa.
- Estados: cargando (esta pantalla). Error → pantalla 3.
- Figma: 🔴 no existe en mis páginas (el componente "Ring de espera (gira)" 3003:5965 sí existe en 00 · Componentes y coincide con `BrandRing`).

### 3. Error de arranque — `CutSellApp.swift` (`RootView`, `ContentUnavailableView`)
- Cómo se llega: `bootstrap()` lanza error (p. ej. falla el Keychain). Un 401 NO llega aquí (va a Bienvenida).
- Fondo: claro, fondo de sistema (#FFFFFF) — **no** usa HomeBackdrop.
- Elementos:
  1. SF Symbol `exclamationmark.triangle` gris de sistema.
  2. Título "Couldn’t start CutSell" (No se pudo iniciar CutSell) — fuente de sistema SF Pro, estilo ContentUnavailable (bold ~22).
  3. Descripción = `error.localizedDescription` (texto de iOS, normalmente en inglés técnico) — SF Pro, gris de sistema.
  4. Botón "Try Again" (Intentar de nuevo) — `.borderedProminent`, color tinte por defecto (azul de sistema **#007AFF**, no hay AccentColor en Assets — no confirmado).
- Estados: solo error.
- Figma: 🔴 no existe. ⚠️ No es White Indigo (fuente y azul del sistema).

### 4. Bienvenida / Iniciar sesión con Apple — `AppleSignIn.swift` (`WelcomeView` + `AppleSignInButton`)
- Cómo se llega: no hay sesión guardada; sesión de invitado antigua; 401 de una sesión Apple; después de "Log Out" o "Delete My Account".
- Fondo: claro, `HomeBackdrop`.
- Elementos (padding 24, separación 24):
  1. Spacer.
  2. Logo `LogoHorizontal` 172 pt.
  3. Texto centrado (separación 14):
     - Teléfono con invitado viejo: "Sign in to keep using CutSell. Your videos move to your account." (Entra para seguir usando CutSell. Tus videos pasan a tu cuenta.)
     - Normal: "Create your account to start. Your videos stay safe on every iPhone you use." (Crea tu cuenta para empezar. Tus videos quedan seguros en cualquier iPhone que uses.)
     - Fuente **SF Pro `.body` (17 Regular)**, color `.secondary` de sistema (≈#3C3C43 al 60 %).
  4. Spacer.
  5. Botón nativo **Sign in with Apple** ("Sign in with Apple" / "Iniciar sesión con Apple", lo dibuja iOS) — estilo **negro** (#000000, texto blanco), alto 50; con `ProgressView` encima mientras trabaja.
  6. Texto de error (solo si falla), rojo de sistema (#FF3B30), SF Pro `.footnote`:
     - "Sign in with Apple didn't finish. Please try again." (El inicio de sesión con Apple no terminó. Vuelve a intentarlo.)
     - "Apple didn't return a sign-in token. Please try again." (Apple no devolvió un token de inicio de sesión. Vuelve a intentarlo.)
     - "Couldn't sign in right now. Please try again." (No se pudo iniciar sesión ahora. Vuelve a intentarlo.)
  7. "One tap with Face ID. You can hide your email." (Un toque con Face ID. Puedes ocultar tu correo.) — SF Pro `.footnote` (13), `.secondary`.
- Estados: vacío/normal, cargando (spinner sobre el botón), error (texto rojo), éxito → Tabs (y si toca, alerta 5). Cancelar en la hoja de Apple no muestra nada.
- Hojas/popups: hoja de sistema de Apple ID.
- No hay botón "continuar sin cuenta" (la función `continueWithoutAccount()` existe pero no se usa).
- Figma: 🔴 no existe. ⚠️ Mezcla fondo White Indigo con textos de sistema (SF Pro, gris y rojo de iOS): no usa Plus Jakarta ni #6B7385 ni #E5484D.

### 5. Alerta "Signed in" — `CutSellApp.swift` (`RootView.alert`)
- Cómo se llega: después de Sign in with Apple, si ese Apple ID ya tenía cuenta y quedaron proyectos de invitado sin mover.
- Alerta de sistema (fondo blanco iOS, SF Pro).
- Título "Signed in" (Sesión iniciada). Mensaje: "This Apple ID already had a CutSell account, so you're now in that account. The N project(s) made without an account on this iPhone could not be moved to it." (Este Apple ID ya tenía una cuenta de CutSell, así que ahora estás en esa cuenta. Los N proyecto(s) hechos sin cuenta en este iPhone … — la última parte "could not be moved to it." no tiene traducción confirmada). Botón "OK".
- Figma: 🔴 no existe.

### 6. Intro (4 pantallas de bienvenida) — `IntroGuide.swift` (`IntroView`)
- Cómo se llega: la primera vez que aparece `MainTabView` (tras iniciar sesión), encima de todo; o Perfil → Help & Legal → "Replay intro". Al terminar va a Create.
- Fondo: claro, **#FFFFFF** (`CS.wiPage`) liso.
- Elementos de arriba hacia abajo:
  1. Logo `LogoHorizontal` 172 pt centrado.
  2. "Skip" (Omitir) — botón texto, Plus Jakarta SemiBold 15, #6B7385, a la derecha.
  3. Ilustración (390×370, pasable con swipe), detrás asset `IntroGlow` 420×420:
     - Página 1: tarjeta "RAW · 1:46" (CRUDO · 1:46) Bold 10 #6B7385, tarjeta #F8F9FB borde 1 pt #E6E9EF radio 16, barras **#591F2E / #404D6B** (⚠️ colores del diseño oscuro viejo); flecha "↓" Bold 26 #1769FF; tarjeta "CUT · 0:40" (CORTE · 0:40) Bold 10 #1769FF, borde 1,5 #1769FF, barras #1769FF; foto `IntroAvatarCut` 110×96 radio 14.
     - Página 2: teléfono `IntroAvatarPhone` 160×284 radio 20 borde 2 #1769FF, rótulo "Look at this!" (¡Mira esto!) Bold 13 blanco sobre negro 60 %; 4 círculos 48 (#F8F9FB, borde #E6E9EF) con "T", "♪", "◇", "🎙" Bold 18 #1769FF.
     - Página 3: pastillas "Hook" / "Body" (Cuerpo) / "Benefit" (Beneficio) / "CTA" Bold 12 blanco sobre #1769FF radio 15; fotos `IntroAvatarClip` 40×50; tarjeta "= 24 videos" Bold 20 #0B1020 fondo #F8F9FB borde 1,5 #1769FF radio 16. ("Hook", "CTA", "= 24 videos" sin traducción.)
     - Página 4: 5 días "M T W T F" (L M M J V) Bold 13, día 2 azul #1769FF con texto blanco, otros #F8F9FB borde #E6E9EF; puntitos #19E6FF y #FF21D6; notificación falsa: cuadro "C" blanco sobre #1769FF, "CutSell · now" (CutSell · ahora) SemiBold 11 #6B7385, "Time to post: Rhino Straps 🔥" (Hora de publicar: Rhino Straps 🔥) SemiBold 14 #0B1020, tarjeta #F8F9FB borde #E6E9EF radio 18; botón falso "Post to TikTok" (Publicar en TikTok) Bold 16 blanco sobre #1769FF radio 26.
  4. Título — ExtraBold 28 #0B1020 centrado: "Upload your raw video" (Sube tu video crudo) / "Edit in seconds" (Edita en segundos) / "Make many videos at once" (Haz muchos videos a la vez) / "Plan your week" (Planea tu semana).
  5. Texto — Medium 15 #6B7385 centrado: "CutSell removes silences, mistakes and repeats for you." (CutSell quita los silencios, errores y repeticiones por ti.) / "Add captions, zoom, sounds and your voice. Everything is editable." (Agrega subtítulos, zoom, sonidos y tu voz. Todo se puede editar.) / "Use Bulk or Sales Mix to turn a few clips into dozens of videos." (Usa Varios o Sales Mix para convertir unos clips en decenas de videos.) / "Pick your times. We remind you and have each video ready to post." (Elige tus horarios. Te avisamos y tenemos cada video listo para publicar.)
  6. Puntos: activo 22×8 #1769FF, otros 8×8 #E6E9EF, radio 4.
  7. Botón "Next" (Siguiente) / última página "Get started" (Empezar) — Bold 16 blanco, alto 52, cápsula `softPuff(.blue)` 3D.
- Estados: solo navegación entre páginas.
- Figma: 🔴 no existe en mis páginas (el código cita frames 1145:2773/2802/2826/2858 de otro archivo).

### 7. Aviso de IA (consentimiento) — `AIConsent.swift` (`AIConsentSheet`)
- Cómo se llega: antes del primer corte (Create → subir/grabar → justo antes de procesar), si nunca se aceptó. Lo presenta UIKit como hoja `.pageSheet` de pantalla grande; no se puede cerrar deslizando.
- Fondo: claro, **#F3F5F9** (`Color(hex: 0x0A1326)` en claro).
- Elementos (padding 24, separación 14):
  1. "Before your first cut" (Antes de tu primer corte) — Bold 22 #0B1020.
  2. "To make your cut, CutSell uses these two AI services:" (Para hacer tu corte, CutSell usa estos dos servicios de IA:) — Regular 15 #6B7385.
  3. Viñeta (círculo 6 #1769FF) + "**Deepgram** — turns the speech in your video's audio into text." — nombre bold #0B1020, resto #6B7385, Regular 14.
  4. Viñeta + "**Anthropic (Claude)** — reads that text to pick your best takes and, if you did not pick a product, to suggest one. It never receives the video, the audio or any picture from it."
     - ⚠️ En español probablemente queda en inglés: el código traduce `" — \(texto)"` y no existe la plantilla " — {}" (las frases sueltas sí existen: "convierte lo que dices en el audio del video en texto.", "lee ese texto para elegir tus mejores tomas y, si no elegiste un producto, para sugerirte uno…"). No confirmado en dispositivo.
  5. "No other AI service receives your content. Cloudflare and Amazon Web Services store your videos, and Modal renders your cut." (Ningún otro servicio de IA recibe tu contenido. Cloudflare y Amazon Web Services guardan tus videos, y Modal genera tu corte.) — Regular 13 #6B7385.
  6. "They receive it only so CutSell can make your cut. CutSell never sells your content. You can delete your cuts or your account at any time." (Lo reciben solo para que CutSell haga tu corte. CutSell nunca vende tu contenido. Puedes borrar tus cortes o tu cuenta cuando quieras.) — Regular 13 #6B7385.
  7. Enlaces "Privacy Policy" (Política de privacidad) y "Terms of Use" (Términos de uso) — SemiBold 13 #1769FF.
  8. Botón "Agree" (Acepto) — `PrimaryButton` SemiBold 16 blanco, alto 52, **radio 16** (rectángulo redondeado, no cápsula), `softPuff(.blue)`.
  9. "Not now" (Ahora no) — texto SemiBold 14 #6B7385 centrado.
- Estados: si "Not now", el corte falla con "To make a cut, agree to the AI processing notice." (Para hacer un corte, acepta el aviso de procesamiento con IA.) dentro de la pantalla 31 ("Couldn’t start the cut").
- Hojas/popups: `LegalView` (pantalla 25).
- Figma: 🔴 no existe.

### 8. Permiso de Cámara — `AlertScreens.swift` (`PermissionScreen`, kind `.camera`)
- Cómo se llega: Create → Record (cámara) cuando la cámara no está permitida. Antes, un instante de pantalla negra #070910 (estado "checking", la cámara está forzada a oscuro).
- Fondo: claro, **#FFFFFF** liso.
- Elementos:
  1. Cabecera `WIHeader`: logo `LogoHorizontal` 172 a la izquierda; a la derecha botón cerrar 40×40 (SF `xmark` SemiBold 15 #0B1020, círculo `softPuff(.white)`) — **solo si ya se negó antes**. ⚠️ La primera vez no hay forma de volver salvo continuar.
  2. Título de pantalla "Camera" (Cámara) — ExtraBold 32, tracking −0,64, #0B1020.
  3. Tarjeta centrada (`wiCard` 3D blanca, padding 20):
     - Ícono asset `WIRecord` 32×32 #1769FF dentro de círculo 72 #EEF3FF.
     - "Allow camera access" (Permitir acceso a la cámara) — ExtraBold 22 #0B1020.
     - "CutSell.ai needs camera access so you can record video." (CutSell.ai necesita acceso a la cámara para que puedas grabar video.) — Medium 14 #6B7385.
     - Botón `WIPillButton` "Continue" (Continuar) o, si ya se negó, "Open Settings" (Abrir Ajustes) — Bold 16 blanco, alto 52, cápsula `softPuff(.blue)`.
     - "You can change this later in Settings." (Puedes cambiarlo luego en Ajustes.) — Medium 12 #6B7385.
- Estados: "Continue" lanza el diálogo de sistema ("CutSell uses the camera to record the videos you want to edit."); negado → "Open Settings" abre Ajustes.
- Figma: 🔴 no existe en mis páginas.

### 9. Permiso de Micrófono — `AlertScreens.swift` (`PermissionScreen`, kind `.microphone`)
- Igual que 8, con: título "Microphone" (Micrófono); ícono **SF Symbol `mic`** 28 (los otros usan assets); "Allow microphone access" (Permitir acceso al micrófono); "CutSell.ai needs microphone access when you record video or voice-over." (CutSell.ai necesita acceso al micrófono cuando grabas video o voz en off.). Texto del diálogo iOS: "CutSell uses the microphone to record the sound of your videos and your voice-overs."
- Figma: 🔴.

### 10. Permiso de Galería negado — `AlertScreens.swift` (`PermissionScreen`, kind `.photos`) desde `GalleryPickerView.swift`
- Cómo se llega: Create → Upload (Single/Bulk/Combine) con Fotos negado/restringido. Siempre `deniedBefore: true`.
- Igual que 8, con: título "Gallery" (Galería); ícono asset `WIUpload` 30×30; "Allow gallery access" (Permitir acceso a la galería); "Choose videos from your library to create a CutSell edit." (Elige videos de tu galería para crear una edición de CutSell.); botón "Open Settings" (Abrir Ajustes); botón X arriba a la derecha.
- Estado "no decidido": spinner `ProgressView().tint(.white)` sobre fondo blanco (CS.background en claro = #FFFFFF) → ⚠️ spinner blanco invisible (no confirmado en dispositivo) mientras sale el diálogo de iOS ("CutSell opens the videos you choose to edit and the photos and videos you add over them.").
- Figma: 🔴.

### 11. Acceso limitado a Fotos — `AlertScreens.swift` (`GalleryAccessCard`)
- Cómo se llega: abrir la galería con permiso "Limitado".
- Fondo: velo #0B1020 al 45 % sobre la galería.
- Elementos (tarjeta 322 de ancho, #FFFFFF, radio 28, padding 26/30):
  1. "Gallery  access" (Acceso a la galería) — ExtraBold 22 #0B1020 (⚠️ el texto en inglés tiene doble espacio).
  2. "Choose videos from your library to create a CutSell edit." — Medium 14 #6B7385.
  3. Botón "Select More Photos…" (Elegir más fotos…) — `WIPillButton` primario 3D azul.
  4. Botón "Keep Current Selection" (Mantener selección actual) — `WIPillButton .outline`: texto Bold 16 #0B1020 sobre `softPuff(.white)` (**sin borde negro**).
- Hojas: selector limitado de iOS.
- Figma: 🔴 no existe en mis páginas. Comparado con el componente "Botón/Secundario" (3003:5998, blanco + borde 1,5 pt #0B1020, plano) → el botón outline de la app no tiene borde.

### 12. Notificaciones — permiso
- No hay pantalla propia: diálogo de iOS al abrir la app (ver 2). Estado visible luego en Preferences (toggles) — no hay fila de permiso de notificaciones en Privacy & Data.
- Figma: 🔴.

### 13. Prueba / Paywall / Suscripción
- **No existe en la app.** No hay paywall, trial ni compra. "Plan & Usage" solo cuenta cortes del mes contra un número fijo (30) que no se aplica. El único rastro de suscripción es el aviso de borrado (pantalla 19) que manda el servidor.
- Figma: la tarjeta "CUTSELL PRO · 62 of 250 edits used" (1174:26) sugiere un plan Pro que la app no tiene.

---

## B. BARRA DE PESTAÑAS

### 14. Barra de pestañas — `MainTabView.swift` (`MainTabView` + `CSTabBar`)
- Cómo se llega: después de iniciar sesión. Abre en Create. Cada pantalla pone `CSTabBar` con `.safeAreaInset(edge: .bottom)` (Create en `HomeHubView`, Edits en `ProjectsView`, Plan en `PlanView`, Profile en `YouView`).
- Fondo de la barra: cápsula flotante **#FFFFFF**, borde 1 pt **#E6E9EF**, sombra #0D1433 10 % radio 16 y +8; alto 68; margen lateral 16; 2 pt sobre el indicador de inicio.
- Pestañas (iguales en ancho):
  1. "Create" (Crear) — asset `WITabCreate`.
  2. "Edits" (Mis videos) — asset `WITabEdits`.
  3. "Plan" (Plan — sin clave, sale igual) — asset `WITabPlan`.
  4. "Profile" (Perfil) — asset `WITabProfile`.
  - Ícono 24×24; activo #1769FF, inactivo #0B1020.
  - Texto 11: activo **Bold** #1769FF, inactivo Medium #6B7385.
  - Activo dentro de una **burbuja cápsula #EEF3FF** (padding v 7, h 4), con animación de resorte.
- Figma: 🟠 distinta — "Barra de abajo" 3003:5986 / "Nav/White" 1174:98:
  - Figma: rectángulo **radio 24**, alto **66**; app: cápsula, alto 68.
  - Figma: íconos **26**; app: 24.
  - Figma: activo **SemiBold** + **punto azul 5×5 debajo**, sin fondo; app: Bold, **burbuja azul suave #EEF3FF**, sin punto.
  - Figma: sombra 0 4 16 #0D1433 8 %; app: 0 8 16 10 %.
  - Colores (#1769FF / #6B7385 / #FFFFFF / #E6E9EF) y textos: iguales.

---

## C. PERFIL (pestaña Profile) Y SUS PÁGINAS

### 15. Perfil (resumen) — `YouView.swift` (`YouView`)
- Cómo se llega: barra de pestañas → "Profile".
- Fondo: claro, `HomeBackdrop` (#E7EEFF→#FFFFFF + brillos); franja de la barra de estado blanca.
- Elementos (padding lateral 20):
  1. `WIHeader`: logo `LogoHorizontal` 172, nada a la derecha.
  2. "Profile" (Perfil) — ExtraBold 32, −0,64, #0B1020.
  3. Tarjeta de cuenta (alto 76, `wiCard` 3D blanca): avatar círculo 44 con degradado = #1769FF liso, inicial Bold 18 blanca; nombre Bold 16 #0B1020 (nombre real; si no, la parte del correo; si no, "Your account" (Tu cuenta) / "Guest" (Invitado)); debajo Medium 13 #6B7385 el correo o "Signed in with Apple" (Sesión iniciada con Apple) / "Using CutSell without an account." (Usando CutSell sin cuenta.); flecha asset `WIChevronSmall` 18 #6B7385. Toca → Account & Security.
  4. (Si el servidor lo manda) aviso rojo de borrado — ver 19.
  5. Tarjeta de uso (`softPuff(.blue)` 3D, radio 24, padding 18/12/16):
     - "THIS MONTH" (ESTE MES) Bold 11, tracking 0,88, blanco 85 %.
     - Pastilla "Manage" (Gestionar) SemiBold 12 blanco + asset `WIManageArrow` 14, fondo blanco 18 %, alto 26.
     - "N of 30 edits used" (N de 30 ediciones usadas) Bold 17 blanco.
     - Barra 6 pt: fondo blanco 25 %, relleno blanco.
     - "Resets Dec 1" (Se reinicia el 1 dic) Medium 12 blanco 85 %.
     - Toca → Plan & Usage.
  6. Tarjeta menú (`wiCard` 3D blanca), 7 filas de 60, ícono en cuadro 36 (#FFFFFF, borde #E6E9EF, radio 11, ícono 20 #0B1020), título Bold 15 #0B1020, detalle Medium 12 #6B7385, `WIChevronSmall`, separador #E6E9EF desde x 61:
     1. `WIClock` "Your results" (Tus resultados) / "Videos edited, video time and hours saved" (Videos editados, tiempo de video y horas ahorradas)
     2. `WIAccount` "Account & Security" (Cuenta y seguridad) / "Profile, sign-in and devices" (Perfil, inicio de sesión y dispositivos)
     3. `WIPlanUsage` "Plan & Usage" (Plan y uso) / "Subscription, usage and history" (Suscripción, uso e historial)
     4. `WIPrivacy` "Privacy & Data" (Privacidad y datos) / "Permissions, consent and deletion" (Permisos, consentimiento y eliminación)
     5. `WIPrefs` "Preferences" (Preferencias) / "Language, camera and notifications" (Idioma, cámara y notificaciones)
     6. `WIConnections` "Connections" (Conexiones) / "TikTok · coming soon" (TikTok · próximamente)
     7. `WIHelp` "Help & Legal" (Ayuda y legal) / "Guides, support and policies" (Guías, soporte y políticas)
  7. Barra de pestañas (Profile activa).
- Estados: sin nombre → inicial de "Your account"; con aviso → banner rojo.
- Hojas: cada fila abre `YouPagesView` a pantalla completa.
- Figma: 🟠 distinta — "Profile" 1174:2:
  - Fondo: Figma blanco liso; app degradado azul claro con brillos.
  - Tarjeta usuario y tarjeta menú: Figma #F8F9FB + borde 1 pt #E6E9EF plano; app blanca 3D sin borde.
  - Tarjeta azul: Figma **"CUTSELL PRO"**, **"62 of 250 edits used"**, azul plano #1769FF; app **"THIS MONTH"**, **"N of 30 edits used"**, azul 3D (#2F7BFF→#0A55F0).
  - App tiene una fila más arriba: **"Your results"** (no está en Figma). Figma: 6 filas (tarjeta 360 alto); app: 7 (420).
  - Barra de pestañas: ver 14.
  - Textos de las 6 filas, íconos, tamaños de letra y avatar: iguales.

### 16. Your results — `EditStats.swift` (`EditStatsPage`)
- Cómo se llega: Perfil → "Your results".
- Fondo: claro, `HomeBackdrop` (marco común `YouScreen`: botón atrás `WIHeaderIcon` asset `WIBack` 40 en círculo 3D blanco; título ExtraBold 22 #0B1020; subtítulo Medium 13 #6B7385).
- Elementos:
  1. Atrás + "Your results" (Tus resultados); "Everything CutSell edited for you" (Todo lo que CutSell editó por ti).
  2. Tarjeta héroe (radio 24, degradado **#1769FF → #5B3BFF → #B13CF2** + brillos #00E5FF 35 % y #FF1F3D 25 %, sombra #1769FF 35 %):
     - Selector "Lifetime" (Desde el inicio) · "This month" (Este mes) · "This week" (Esta semana) — SemiBold 13; activo texto #1769FF sobre cápsula blanca; resto blanco; fondo blanco 18 %.
     - "TIME SAVED" (TIEMPO AHORRADO) Bold 11 blanco 80 %.
     - Número que cuenta hacia arriba (p. ej. "6.7 h" / "20 min") ExtraBold 44 blanco.
     - "That's 40 min of your day" / "That's X of an 8-hour workday" / "That's X workdays of 8 hours" (Son … min de tu día / Son … de una jornada de 8 horas / Son … jornadas de 8 horas) SemiBold 13 blanco 90 %.
     - Anillo de 24 h 92×92: fondo blanco 22 %, trazo #00E5FF→blanco→#FFD166; dentro "N%" o "N.N" ExtraBold 17 + "of 24 h" (de 24 h) / "days" (días) SemiBold 10.
  3. Dos tiles (`wiCard` 3D radio 18): ícono SF `film.stack` / `clock.arrow.circlepath` #1769FF en cuadro 34 #EEF3FF; número ExtraBold 26 #0B1020; "Videos edited" (Videos editados) / "Video time" (Tiempo de video) o "Adding up…" (Sumando…) SemiBold 12 #6B7385.
  4. Tarjeta "Last 8 weeks" (Últimas 8 semanas) Bold 15; a la derecha "N videos · X saved" SemiBold 12 #1769FF; 8 barras (actual #5B3BFF→#1769FF, otras #9DB8FF→#C9D8FF); etiquetas "d MMM" y "This week" (Esta semana).
  5. Tarjeta meta: círculo 48 #FFD166→#FF8A3D con SF `trophy.fill` blanco; "Your first goal" (Tu primera meta) o "N videos reached!" (¡N videos logrados!) Bold 14; barra #F1E7D6 con relleno #FFD166→#FF8A3D; "N more to M videos" (Faltan N para M videos) Medium 12 #6B7385.
  6. Nota: "Hours saved = 20 min of editing by hand per video. A workday is 8 hours." (Horas ahorradas = 20 min de edición a mano por video. Una jornada son 8 horas.) Medium 11 #6B7385.
- Estados: cargando → "Adding up…"; vacío → todo en 0 y "Your first goal".
- Figma: 🔴 no existe. Usa colores fuera de la paleta (morados, naranjas, cian).

### 17. Account & Security — `YouView.swift` (`AccountPage`)
- Cómo se llega: Perfil → tarjeta de cuenta o fila "Account & Security".
- Fondo: claro, `HomeBackdrop`.
- Elementos (filas `YouRow`: **título en SF Pro (sistema) 14 Regular** #0B1020, detalle Plus Jakarta Medium 11 #6B7385, "›" Medium 20 #6B7385, separador #E6E9EF; tarjetas `youCard` 3D blanca radio 22; etiquetas de sección SemiBold 12 #1769FF):
  1. Atrás + "Account & Security" (Cuenta y seguridad); "Your CutSell identity across iPhone and web" (Tu identidad de CutSell en iPhone y web).
  2. "PROFILE" (PERFIL): "Display name" (Nombre) / nombre o "Add your name" (Añade tu nombre) ›; "Email" (Correo) / correo o "Hidden by Apple" (Oculto por Apple); "Sign-in method" (Método de inicio de sesión) / "Sign in with Apple" (Iniciar sesión con Apple) [o "No account (this iPhone only)" (Sin cuenta (solo este iPhone)) — en la práctica inalcanzable].
  3. "SECURITY" (SEGURIDAD): "Apple account" (Cuenta de Apple) / "Connected with Apple" (Conectada con Apple); "Sign out" (Cerrar sesión) / "This iPhone only" (Solo este iPhone) — título en **#FF4D67** (`CS.error`, no #E5484D).
     - (Rama invitado, inalcanzable hoy: "Sign in with Apple to keep your projects on every iPhone you use." + botón Apple.)
  4. "Account ID  •  abc123••••wxyz" (ID de cuenta) Medium 11 #6B7385; "This ID links your projects across iPhone and web." (Este ID une tus proyectos entre iPhone y web.) Medium 11,5 #6B7385.
  5. Tarjeta sola: "Delete account" (Eliminar cuenta) en **#FF4D67** / "Permanently removes account and eligible data" (Borra para siempre la cuenta y sus datos).
- Popups:
  - Alerta con campo "Display name" (Nombre): campo "Your name" (Tu nombre), botones "Save" (Guardar), "Cancel" (Cancelar).
  - Hoja de acción "Log out of CutSell?" (¿Cerrar sesión en CutSell?): mensaje "Your projects stay in your account. Sign in with Apple again to see them." (Tus proyectos se quedan en tu cuenta. Inicia sesión con Apple otra vez para verlos.); botones "Log Out" (Cerrar sesión), "Cancel". → Bienvenida.
  - Hoja de acción "Delete CutSell account?" (¿Eliminar tu cuenta de CutSell?): mensaje "This permanently deletes your CutSell projects and account data and signs out all indexed sessions." (Esto borra para siempre tus proyectos y datos de CutSell y cierra todas las sesiones.); botón rojo de sistema "Delete My Account" (Eliminar mi cuenta), "Cancel". → Bienvenida. ⚠️ El inglés dice "indexed sessions" (raro).
  - Error al borrar: alerta "CutSell" / "Your account isn't fully deleted yet. Tap Delete My Account again to finish." (Tu cuenta todavía no se borró por completo. Toca Eliminar mi cuenta otra vez para terminar.) / "OK".
- Figma: 🔴 no existe. ⚠️ Títulos de fila en SF Pro, rojo #FF4D67 fuera de paleta.

### 18. Plan & Usage — `YouView.swift` (`PlanUsagePage`)
- Cómo se llega: Perfil → tarjeta azul o fila "Plan & Usage".
- Fondo: claro, `HomeBackdrop`.
- Elementos: atrás + "Plan & Usage" (Plan y uso); "Your edits this month" (Tus ediciones de este mes); sección "THIS MONTH" (ESTE MES): "Edits processed" (Ediciones procesadas) SemiBold 14 #0B1020 + "N / 30" SemiBold 14; barra 6 pt fondo #F3F5F9 relleno **#0E46D9** (azul fuera de paleta); separador; fila "Editing time saved" (Tiempo de edición ahorrado) / "X h YY min"; debajo "Resets Dec 1" (Se reinicia el 1 dic) Medium 11 #6B7385.
- No hay suscripción, historial ni botón de compra (aunque la fila dice "Subscription, usage and history").
- Figma: 🔴 no existe.

### 19. Aviso de borrado por suscripción cancelada — `HomeHubView.swift` (`DeletionWarningBanner`)
- Cómo se llega: el servidor manda `subscription.warning` (o `warning_es` en español) en `/v1/account/me`. Se ve en Perfil (debajo de la tarjeta de cuenta) y en Create.
- Elementos: SF Symbol `exclamationmark.triangle.fill` **#D92D20**; texto del servidor (no está en la app) SemiBold 13 **#7A271A**; fondo **#FEF3F2**, borde 1 pt **#FDA29B**, radio 16, padding 14.
- Figma: 🔴 no existe. ⚠️ Rojos fuera de la paleta White Indigo (deberían ser #E5484D / #FFF7F7 / #FDECEC).

### 20. Privacy & Data — `YouView.swift` (`PrivacyPage`)
- Cómo se llega: Perfil → "Privacy & Data".
- Fondo: claro, `HomeBackdrop`.
- Elementos:
  1. Atrás + "Privacy & Data" (Privacidad y datos); "Control access, storage and consent" (Controla el acceso, el almacenamiento y el consentimiento).
  2. "IPHONE PERMISSIONS" (PERMISOS DEL IPHONE): "Photos" (Fotos) / "Camera" (Cámara) / "Microphone" (Micrófono) con valor Medium 12 #6B7385 + "  ›": "Full access" (Acceso total), "Selected Photos" (Fotos elegidas), "Allowed" (Permitido), "Denied" (Denegado), "Not asked yet" (Todavía no se pidió). Toca → Ajustes de iOS.
  3. "AI & CONTENT" (IA Y CONTENIDO): "Cloud & AI processing" (Procesamiento en la nube y con IA) / "Allowed" o "Not asked yet" (sin acción; no se puede retirar el consentimiento).
  4. "DATA & RETENTION" (DATOS Y CONSERVACIÓN): "Clear local cache" (Borrar caché local) / "On this iPhone" (En este iPhone) → "Cleared" (Borrada).
  5. Enlaces "Privacy Policy  ›" (Política de privacidad) y "Terms of Use  ›" (Términos de uso) SemiBold 13 #1769FF.
- Hojas: `LegalView` (25).
- Figma: 🔴 no existe.

### 21. Preferences — `YouView.swift` (`PreferencesPage`)
- Cómo se llega: Perfil → "Preferences".
- Fondo: claro, `HomeBackdrop`.
- Elementos (interruptores `YouToggleRow`: título Medium 14 #0B1020, Toggle de iOS tinte #1769FF, nota Medium 11 #6B7385):
  1. Atrás + "Preferences" (Preferencias); "Make CutSell work the way you do" (Haz que CutSell funcione a tu manera).
  2. "GENERAL" (sin traducción): "App language" (Idioma de la app) / "System Default" (Predeterminado del sistema) o "English"/"Español" ›; "Haptic feedback" (Vibración) [toggle].
  3. "CAMERA & EXPORT" (CÁMARA Y EXPORTACIÓN): "Use CutSell camera defaults" (Usar los ajustes de cámara de CutSell) + nota "Uses the resolution, frame rate and HDR saved inside CutSell when supported by this iPhone." (Usa la resolución, los fotogramas y el HDR guardados en CutSell cuando este iPhone los admite.); "Auto-save exports to Photos" (Guardar exportaciones en Fotos); "Default export quality" (Calidad de exportación) / "4K Ultra HD / 1080p HD" (sin acción); "Storage" (Almacenamiento) / "N MB" ›.
     - ⚠️ El toggle "Auto-save exports to Photos" no tiene separador y la fila siguiente sí (divider: false en medio de la tarjeta).
  4. "NOTIFICATIONS" (NOTIFICACIONES): "Processing complete" (Procesamiento terminado); "Time to post reminders" (Recordatorios para publicar); "Product updates" (Novedades del producto).
- Figma: 🔴 no existe.

### 22. App Language — `YouView.swift` (`LanguagePage`)
- Cómo se llega: Preferences → "App language".
- Fondo: claro, `HomeBackdrop`.
- Elementos: atrás + "App Language" (Idioma de la app); "Choose the language used throughout CutSell" (Elige el idioma de todo CutSell); tarjeta 3D con 3 opciones (`YouChoiceRow`: título **SF Pro 17** (SemiBold si elegido) #0B1020, detalle Medium 11 #6B7385, radio 22 aro 2 pt #1769FF o #6B7385, punto 11 #1769FF):
  "System Default" (Predeterminado del sistema) / "Matches iPhone language" (Igual que el idioma del iPhone); "English" / "United States"; "Español" / "Estados Unidos / Latinoamérica".
  Nota "Changes apply immediately" (Los cambios se aplican al instante) SemiBold 15 + "Navigation, processing states, export screens and account settings use the selected language." + "Privacy and legal pages open in the selected language when available." (traducidas); "Video filenames and technical metadata remain unchanged." (Los nombres de los videos y los datos técnicos no cambian.) Medium 11 #6B7385.
- Estados: al cambiar, toda la app se redibuja y vuelve a esta página.
- Figma: 🔴 no existe.

### 23. Storage — `YouView.swift` (`StoragePage`)
- Cómo se llega: Preferences → "Storage".
- Fondo: claro, `HomeBackdrop`.
- Elementos: atrás + "Storage" (Almacenamiento); "On this iPhone" (En este iPhone); total "N MB / N.N GB" Bold 34 #0B1020 + "On this iPhone" Medium 13 #6B7385; sección "ON THIS IPHONE" (sin traducción): "Videos you uploaded" (Videos que subiste), "Previews" (Vistas previas), "Copies being uploaded" (Copias que se están subiendo), "Voices and sounds" (Voces y sonidos), "Exports" (Exportaciones) con tamaño; botón "Free up space" (Liberar espacio) SemiBold 16 blanco, fondo **#1769FF plano**, radio 16, alto 52 (con spinner blanco mientras trabaja); "Freed N MB" (Liberado: N MB) Medium 13 #6B7385; nota "Storage" + "Your originals stay in Photos. Freeing space only removes CutSell's copies; they come back when needed." (Tus originales se quedan en Fotos…).
- Figma: 🔴. ⚠️ Botón plano (las demás pantallas usan 3D).

### 24. Connections — `YouView.swift` (`ConnectionsPage`)
- Cómo se llega: Perfil → "Connections".
- Elementos: atrás + "Connections" (Conexiones); "Link your accounts to post from CutSell" (Vincula tus cuentas para publicar desde CutSell); tarjeta: "TikTok" / "Post and track your videos" (Publica y sigue tus videos); "Instagram Reels" / "Later" (Más adelante); "YouTube Shorts" / "Later" — títulos **SF Pro 17** (Regular o SemiBold, mezclado entre filas), insignia "SOON" (PRONTO) Bold 9 #FFB020, fondo #FFB020 12 %, borde 80 %. Nota "Until then: Post to TikTok" (Mientras tanto: Publicar en TikTok) / "We save the video, copy your caption and open TikTok. You just paste and post." (Guardamos el video, copiamos tu caption y abrimos TikTok. Solo pegas y publicas.).
- Figma: 🔴. ⚠️ Naranja #FFB020 fuera de paleta (paleta: aviso #F5A524).

### 25. Privacy Policy / Terms of Use — `AIConsent.swift` (`LegalView`)
- Cómo se llega: Privacy & Data → enlaces; Aviso de IA → enlaces. (Help & Legal NO tiene enlaces legales: la variable existe pero ninguna fila la usa.)
- Hoja con NavigationStack; fondo #FFFFFF; título inline "Privacy Policy" (Política de privacidad) / "Terms of Use" (Términos de uso) de sistema; botón "Done" (Listo) de sistema arriba a la derecha (azul de sistema — no confirmado).
- Contenido de `Legal/privacy_policy.md` / `terms_of_use.md`: "## " → SemiBold 17 #0B1020; "- " → "•  " Regular 14 #6B7385; resto Regular 14 #6B7385 (Markdown en línea).
- Figma: 🔴.

### 26. Help & Support — `YouView.swift` (`HelpPage`)
- Cómo se llega: Perfil → "Help & Legal" (⚠️ la fila dice "Help & Legal" pero la página se llama "Help & Support" y no tiene nada legal).
- Fondo: claro, `HomeBackdrop`.
- Elementos:
  1. Atrás + "Help & Support" (Ayuda y soporte); "Get help, recover the app and report an issue" (Pide ayuda, recupera la app y reporta un problema).
  2. "HELP TOOLS" (HERRAMIENTAS DE AYUDA): "Report a Problem" (Reportar un problema) / "Send feedback tied to the edit" (Envía comentarios sobre la edición) → Diagnostics; "Contact Support" (Contactar a soporte) / "support@cutsell.ai" → Mail; "Refresh App Data" (Actualizar datos de la app) / "Fix stuck screens · keeps your projects" (Arregla pantallas trabadas · conserva tus proyectos); "Storage & Originals" (Almacenamiento y originales) / "Manage local cache and 4K source files · N MB" (Administra la caché local y los originales en 4K) → ⚠️ borra la caché al instante, sin confirmar.
  3. "LEARN" (APRENDER): "How to use CutSell" (Cómo usar CutSell) / "Step-by-step guides for every feature" (Guías paso a paso de cada función) → 27; "Replay intro" (Ver la introducción otra vez) / "See the 4 welcome screens again" (Mira de nuevo las 4 pantallas de bienvenida) → cierra Perfil y muestra 6.
  4. Nota "You control what is shared" (Tú decides qué se comparte) / "No video, audio or transcript is sent unless you choose it in the report." (No se envía video, audio ni transcripción a menos que lo elijas en el reporte.).
  5. Aviso (tras una acción) SemiBold 12 #1769FF: "App data refreshed. Your projects are safe." (Datos de la app actualizados. Tus proyectos están a salvo.) / "Cache cleared on this iPhone. Your projects are safe." (Caché borrada en este iPhone. Tus proyectos están a salvo.).
  6. "CutSell 1.0  •  Build 78  •  iOS 17+" Medium 12 #6B7385 (5 toques rápidos → Diagnostics).
- Figma: 🔴.

### 27. How to use CutSell (lista de guías) — `IntroGuide.swift` (`GuidePage`)
- Cómo se llega: Help & Support → "How to use CutSell".
- Fondo: claro **#FFFFFF** liso (no usa HomeBackdrop, a diferencia de las otras páginas de Perfil).
- Elementos: "‹ Help" (‹ Ayuda) SemiBold 16 #1769FF; "How to use CutSell" (Cómo usar CutSell) **Bold 28** #0B1020; "Short guides, 3–5 steps each." (Guías cortas, de 3 a 5 pasos.) Regular 13 #6B7385; buscador alto 40, "⌕" + "Search guides" (Buscar guías), fondo #F8F9FB, borde #E6E9EF, radio 12; 9 filas de 52 (fondo #F8F9FB, borde #E6E9EF, radio 14, planas): glifo en cuadro 36 radio 10 (#F3F5F9, o #1769FF con glifo blanco para Sales Mix), título SemiBold 14, "N steps" (N pasos) Regular 11 #6B7385, "›":
  "Upload: Single, Bulk, Combine", "Sales Mix: many videos from a few clips", "Editor basics", "Keyframes and zoom", "Voice over", "Teleprompter", "Products in My videos", "Week Plan", "Post to TikTok".
- Estados: búsqueda sin resultado → "No guide matches that." (Ninguna guía coincide.).
- Figma: 🔴 no existe en mis páginas. ⚠️ Estilo distinto al resto de Perfil (botón atrás de texto, tarjetas planas con borde, título Bold 28 en vez de ExtraBold 22).

### 28. Detalle de una guía — `IntroGuide.swift` (`GuideDetailPage`)
- Cómo se llega: lista de guías → una fila.
- Fondo: claro #FFFFFF.
- Elementos: "‹ Guides" (‹ Guías) SemiBold 16 #1769FF; nombre Bold 28 #0B1020; subtítulo Regular 13 #6B7385; pasos con círculo 32 #1769FF y número Bold 14 blanco, línea 2 pt #E6E9EF, título SemiBold 15 #0B1020, texto Regular 13 #6B7385; botón "Try … now" (p. ej. "Try Upload now") SemiBold 15 blanco, alto 50, radio 25, **#1769FF plano** + sombra #1969FF 25 % → cierra Perfil y abre la función.
- Figma: 🔴.

### 29. Diagnostics (Report a Problem / Send report) — `DiagnosticsCenter.swift` (`DiagnosticsView`)
- Cómo se llega: Help & Support → "Report a Problem", o 5 toques en la versión. Hoja.
- Fondo: claro, **lista agrupada de sistema** (#F2F2F7 / celdas blancas, SF Pro) — no confirmado el gris exacto.
- Elementos: título inline "Diagnostics" (Diagnóstico); "Done" (Listo) a la izquierda; "Send report" (Enviar reporte) a la derecha (ShareLink con un .txt); secciones "App" (Build / Version / iPhone), "Audio" (líneas monoespaciadas), "Photos" (Permission: "Full access" / "Limited (only the photos you picked)" / "Denied" / "Restricted" / "Not asked yet"), "Sounds on the timeline (last preview)" (líneas + "Video sound: N%"), "Last N events" (o "No events yet." (Todavía no hay eventos.)).
- Figma: 🔴. ⚠️ Pantalla 100 % de sistema, sin White Indigo.

---

## D. ERRORES, AVISOS, TOASTS Y BANNERS

### 30. Upload paused (subida detenida, guardada) — `CreateFlowView.swift` → `AlertScreens.swift` (`ProblemScreen`)
- Cómo se llega: Create → la subida falla o se traba (StallWatch) después de guardar el registro. Pantalla completa sobre el Processing.
- Fondo: claro #FFFFFF liso.
- Elementos (plantilla `ProblemScreen`):
  1. `WIHeader` con logo 172.
  2. "Upload paused" (Subida en pausa) ExtraBold 32 #0B1020.
  3. "!" Bold 54 #1769FF.
  4. "The upload stopped" (La subida se detuvo) ExtraBold 22 #0B1020.
  5. "<detalle>\n\nYour video is saved on this iPhone. Try again to continue where it stopped." (… Tu video está guardado en este iPhone. Vuelve a intentarlo para seguir donde se detuvo.) Medium 14 #6B7385, interlínea 4. El detalle es "Error: <motivo corto>." o el texto del StallWatch.
  6. Botones (ancho 270, separación 12): "Try again" (Intentar de nuevo) `WIPillButton` 3D azul alto 52; "Go to Edits" (Ir a Mis videos) SemiBold 15 #0B1020, `softPuff(.white)` radio 16, alto 52.
- Figma: 🟠 usa la plantilla de "Formato no soportado" (1233:3066) pero este texto no tiene frame propio.

### 31. Couldn’t start the cut (no empezó) — `CreateFlowView.swift` (`ProblemScreen`)
- Cómo se llega: falla antes de guardar nada (sin red, consentimiento IA negado, etc.).
- Igual que 30: "Couldn’t start the cut" (No se pudo iniciar el corte); "The upload didn’t start" (La subida no empezó); "<detalle>\n\nCheck your connection and try again. Your video is still here." (… Revisa tu conexión y vuelve a intentarlo. Tu video sigue aquí.); botones "Try again" / "Cancel" (Cancelar).
- Este es el único "error de red" visible de Create. No hay pantalla de "sesión expirada": un 401 manda a Bienvenida (4) sin explicación.
- Figma: 🟠 (misma plantilla, sin frame propio).

### 32. Unsupported media — `AlertScreens.swift` (`CreateProblemView` `.unsupportedMedia`)
- Cómo se llega: Create → exportar de Fotos un formato que no es mp4/mov/m4v.
- Fondo: claro #FFFFFF.
- Elementos: logo; "Can’t add this video" (No se puede añadir este video) ExtraBold 32; "!" Bold 54 #1769FF; "Unsupported media" (Archivo no compatible) ExtraBold 22; "This video format isn’t supported yet. Choose another video or review the formats CutSell accepts." (Este formato de video todavía no es compatible. Elige otro video o revisa los formatos que acepta CutSell.) Medium 14 #6B7385; "Back to library" (Volver a la galería) 3D azul cápsula 52; "View supported formats" (Ver formatos compatibles) SemiBold 15 #0B1020 3D blanco radio 16 → 33.
- Figma: 🟠 "Formato no soportado" 1233:3066 — igual en textos y disposición; cambia: botón principal Figma **alto 56** con texto **oscuro alineado a la izquierda** (parece error del Figma), app alto 52 texto blanco centrado; secundario Figma **#F8F9FB plano con borde**, texto a la izquierda en 2 líneas; app blanco 3D centrado; Figma botón principal rectángulo radio ~16, app cápsula.

### 33. Supported video files — `AlertScreens.swift` (`.supportedFormats`)
- Cómo se llega: 32 → "View supported formats".
- Igual que 32: "Supported video files" (Videos compatibles); "!" azul (el tono `.info` no cambia nada); "Formats and limits" (Formatos y límites); "MP4, MOV and M4V\nH.264, HEVC and ProRes\nUp to 10 minutes and 1 GB" (MP4, MOV y M4V / H.264, HEVC y ProRes / Hasta 10 minutos y 1 GB); "Choose a video" (Elige un video) / "Back" (Atrás) — los dos hacen lo mismo.
- ⚠️ Dice "Up to 10 minutes and 1 GB" pero esos límites están apagados (`sizeAndLengthLimitsOn = false`).
- Figma: 🔴 no hay frame en mis páginas.

### 34. Clip limit reached — `AlertScreens.swift` (`.clipLimit`) desde `GalleryPickerView`
- Cómo se llega: galería en Bulk/Combine al elegir el clip 11.
- Igual que 32: "Clip limit reached" (Llegaste al límite de clips); "10 clips maximum" (Máximo 10 clips); "Process these clips or remove one before adding another." (Procesa estos clips o quita uno antes de añadir otro.); "Review clips" (Revisar clips) / "Remove a clip" (Quitar un clip).
- Figma: 🟠 "Límite de clips" 1233:3101 — mismas diferencias de botones que 32.

### 35. Couldn’t add this clip — `AlertScreens.swift` (`.couldNotAdd`)
- Cómo se llega: la cámara no pudo guardar la toma (`recordingFailed`), o Fotos no pudo exportar el video (`.unavailable`, p. ej. iCloud).
- Igual que 32: "Couldn’t add this clip" (No se pudo añadir este clip); "Something went wrong" (Algo salió mal); "Try recording the clip again. Your saved clips are still here." (Intenta grabar el clip otra vez. Tus clips guardados siguen aquí.); "Back to clips" (Volver a los clips) / "Try again" (Intentar de nuevo). ⚠️ Desde la galería el texto habla de "grabar" aunque el video venga de Fotos.
- Figma: 🟠 "Error de cámara" 1233:3112 — mismas diferencias de botones.

### 36. Video too long / Upload limit exceeded (hoja) — `AlertScreens.swift` (`LimitScreen`)
- Cómo se llega: **nunca en build 78** (`CreateLimits.sizeAndLengthLimitsOn = false`).
- Si se activara: velo #0B1020 45 %; hoja `WISheetCard` blanca radio 28, asa 40×5 #D5D9E2; `WIStatusBadge .warning` 56 (asset `WIWarn` #F5A524 en círculo #FFF4E0); "This video is too long" (Este video es demasiado largo) o "Upload limit exceeded" (Se superó el límite de subida) ExtraBold 22; "Choose a video up to 10 minutes. Your original file remains unchanged." / "This file is X. CutSell accepts files up to 1 GB. Your original file is unchanged." Medium 14 #6B7385; fila "Selected video" (Video seleccionado) Bold 14 + duración/tamaño Medium 12 **#E5484D**, miniatura = rectángulo gris #F3F5F9 36×48; botón "Choose another video" (Elegir otro video) 3D azul.
- Figma: 🟠 "Hoja · Video muy largo / muy pesado" 1176:2888 — Figma: ícono **reloj** (la app usa `WIWarn`), **miniatura real** del video, texto "12:18 · 620 MB", insignia **"Max 10 min"** a la derecha, texto "Your original file is not changed."; app: sin insignia, sin miniatura, "remains unchanged". (Y la app no la muestra nunca.)

### 37. Processing issue (error al procesar) — `ProcessingScreen.swift` (`ProcessingContent`, `failed = true`)
- Cómo se llega: el servidor no pudo hacer el corte (Single), o abrir un corte fallado desde el aviso/Edits.
- Fondo: claro (el contenedor `CutProgressView`; no confirmado si es HomeBackdrop o blanco).
- Elementos: `WIBackBar` (atrás `WIBack` 40 3D, "Processing issue" (Problema al procesar) Bold 17 #0B1020, "Edits" (Mis videos) SemiBold 15 #1769FF); primer cuadro 100×156 radio 20 borde blanco 3 + sombra azul; nombre de archivo Medium 13 #6B7385; "We couldn’t finish this cut" (No pudimos terminar este corte) ExtraBold 22 **#4A5263**; anillo quieto 150 con "!" blanco en círculo 22 #E5484D; "Processing stopped" (El procesamiento se detuvo) SemiBold 15 #E5484D; "Error" ExtraBold 22 #E5484D; "Your original recording is safe" (Tu grabación original está a salvo) Medium 12 #6B7385.
  - Sin voz: "We couldn’t hear anyone talking" (No escuchamos a nadie hablando) ExtraBold 19 #E5484D + "CutSell edits videos where someone speaks. Try a video with your voice." (CutSell edita videos donde alguien habla. Prueba con un video con tu voz.).
  - Nota 3D blanca radio 14: "Your original video is safe. Nothing was lost." (⚠️ **sin traducción** — sale en inglés en español) o, si falla el reintento, "Couldn’t retry this cut. Please upload it again." (No se pudo reintentar este corte. Vuelve a subirlo.) en #E5484D.
  - Botón "Retry processing" (Reintentar procesamiento) Bold 16 blanco cápsula **3D azul** 52; "Try again—the engine will resume from this stage." (Inténtalo de nuevo: el motor seguirá desde esta etapa.) Medium 12 #6B7385.
- Figma: 🟠 "Error al procesar" 1233:3123 — Figma: botón **rojo #E5484D**; nota con **borde azul** y fondo #F8F9FB; nombre de archivo **oscuro y más grande**; app: botón azul (decisión PO 24 oct según comentario), nota blanca 3D sin borde, nombre gris 13.

### 38. Queued (en cola) — `ProcessingScreen.swift` (`queued = true`)
- Cómo se llega: el servidor tiene el corte pero no hay espacio.
- Elementos: barra "Queued" (En cola) + "Edits"; cuadro y nombre; "Your cut is next in line" (Tu corte es el siguiente) Bold 24 #4A5263; anillo girando 150; "Waiting to start" (Esperando para empezar) Bold 17 #0B1020; "0%" ExtraBold 26 #1769FF; "Waiting for a processing slot" (Esperando un espacio para procesar) Medium 13 #6B7385; nota 3D blanca "We’ll start automatically when a slot opens." (Empezaremos solos cuando haya un espacio.); botón "Continue in background" (Seguir en segundo plano) Bold 16 blanco sobre **#1769FF plano** cápsula 52; "We’ll notify you when processing begins." (Te avisaremos cuando empiece el procesamiento.).
- Figma: 🟠 "En cola" 1233:3155 — Figma dice "Waiting for a processing slot **• 2 ahead**" (la app no muestra cuántos hay delante); nota Figma con **borde azul**; título Figma ExtraBold más grande; resto igual.

### 39. Export failed — `ExportSettingsView.swift` (`ExportFailedView`)
- Cómo se llega: editor → Export → falla la exportación en el iPhone.
- Fondo: claro, `HomeBackdrop`.
- Elementos: `WIBackBar` "Export failed" (La exportación falló); anillo quieto 110 con "!" rojo #E5484D; "72%" ExtraBold 40 #0B1020; "Export stopped" (La exportación se detuvo) Medium 15 #6B7385; "Your video was not saved" (Tu video no se guardó) Medium 12 #1769FF; tarjeta de pasos "Preparing video" (Preparando el video), "Rendering frames" (Renderizando cuadros), "Encoding video" (Codificando el video), "Saving to Photos" (Guardando en Fotos) (✓ #33D98C hecho / "!" el que falló / "—" pendiente); tarjeta "Nothing was lost" (No se perdió nada) / "Your cut and export settings are still saved." (Tu corte y tus ajustes de exportación siguen guardados.); "Try Again" (Intentar de nuevo) Bold 16 3D azul **radio 18**; "Change export settings" (Cambiar ajustes de exportación) SemiBold 16 #0B1020 fondo blanco borde 1,5 #1769FF radio 18; enlace "Export on our server (1080p)" (Exportar en nuestro servidor (1080p)) SemiBold 13 #1769FF.
- Figma: 🟠 "Error al exportar" 1233:3199 — Figma: botón atrás cuadrado redondeado gris; botones **radio ~12–16 planos**; la app añade el enlace "Export on our server (1080p)" que no está en Figma; tiempos por paso (0:08, 0:41) en Figma — no confirmado si la app los pinta (código de `step` no revisado entero).

### 40. Needs attention (Bulk/Plan) — `NeedsAttention.swift` (`NeedsAttentionSection`) en `PlanView`
- Cómo se llega: Plan, cuando un envío Bulk tiene videos fallidos.
- Elementos: tarjeta resumen con anillo % (Bold 17), "N of M ready" (N de M listos) Bold 19, "N finished cuts are already in Edits" (N cortes terminados ya están en Edits); variante con "N ready" (#33D98C) · "N active" (#1769FF) · "N queued · N failed" y "1 video needs action" (1 video necesita acción); título "Needs attention" (Necesita atención) + "Retry failed" (Reintenta los fallidos); filas "01…" con nombre SemiBold 13, duración, estado SemiBold 10 #1769FF y botón "Retry" (Reintentar) SemiBold 11 #1769FF.
- Figma: 🟠 "Bulk · Needs attention" 1233:3243 y "Plan · Needs attention" 1233:3354 — comparación detallada **no confirmada** (pertenece a Plan). Diferencia visible: en Figma el botón "Retry" es **rojo con borde rojo**, estado "FAILED" azul; logo centrado arriba.

### 41. Toast "<nombre> is ready · View" — `ProcessingQueue.swift` (`ReadyToast`) en `MainTabView`
- Cómo se llega: termina un corte (Single/Combine) mientras se usa la app; 84 pt sobre el fondo, en cualquier pestaña; se va a los 6 s.
- Elementos: tarjeta blanca **#FFFFFF plana** con borde 1 pt #E6E9EF y sombra 8 %, radio 14, alto 46; check SF `checkmark` Bold 11 #1FA971 en círculo 22 #EAF8F1; "cutsell…849 is ready" (… está listo) SemiBold 14 #0B1020; "View" (Ver) Bold 14 #1769FF → abre el corte.
- Figma: 🔴 no existe en mis páginas (código cita 1213:2 / 1134:3027).

### 42. Toast de grupo Bulk — `ProcessingQueue.swift` (`GroupReadyToast`)
- Cómo se llega: termina un Bulk entero; 8 s.
- Igual que 41 con texto: "Your video is ready ✨" (Tu video está listo ✨) / "Your N videos are ready ✨" (Tus N videos están listos ✨) / "1 video needs attention" (1 video necesita atención) / "N videos need attention" / "1 ready, 1 needs attention" (1 listo, 1 necesita atención) / "N ready, 1 needs attention" (N listos, 1 necesita atención) …; "View" → Edits filtro Ready.
- Figma: 🔴.

### 43. Banner superior "export lista" — `MainTabView.swift` (`ReadyBanner`)
- Cómo se llega: una exportación termina mientras se está en otra parte de la app.
- Elementos: logo asset `ProcessingLogo` 34 con `.blendMode(.screen)` (⚠️ sobre fondo claro casi no se ve — no confirmado); "<título> is ready" o "Your video is ready" (Tu video está listo) SemiBold 15 #0B1020; "Open the cut in Edits to save or share it" (Abre el corte en Mis videos para guardarlo o compartirlo) Regular 12 #6B7385; X SF `xmark` Bold 13 #6B7385 (32×32). Fondo #F8F9FB radio 18, borde #1769FF 50 %, sombra **#1769FF 25 % radio 12** (resplandor del diseño oscuro viejo con cian).
- Figma: 🔴. ⚠️ Estilo heredado del diseño oscuro (borde/brillo "cyan" convertido a azul).

### 44. Aviso "Did you post it?" — `TikTokPoster.swift` (`.alert`)
- Cómo se llega: al volver de TikTok tras "Post to TikTok".
- Alerta de sistema: "Did you post it?" (¿Lo publicaste?) / "We'll mark the video as posted in your plan." (Marcaremos el video como publicado en tu plan.) / "Yes, posted" (Sí, publicado) · "Not yet" (Todavía no).
- Figma: 🔴.

### 45. Otras alertas de sistema dentro de la app (todas fondo blanco de iOS, SF Pro; Figma 🔴 para todas)
- "CutSell" + mensaje de error + "OK" — editor (B-roll, Captions, Overlay, Voice over, Draft editor, EditorV2/V4), Mix, Product picker, Products, Edits, Plan my week, Week plan. Mensaje = texto del error (en V4 pasado por `plainWords`).
- "Couldn’t create cut" (No se pudo crear el corte) + mensaje + "OK" — `NewCutView`.
- "Delete project?" (¿Eliminar proyecto?) / "This permanently removes the project, its uploaded media, exports, overlays, timeline assets and feedback." / "Delete Project" (Eliminar proyecto) · "Cancel" — `ProjectsView`.
- "Faceless video" (Video sin cámara) / "Coming soon. For now you can record a voice over inside the editor of any cut." (Próximamente…) / "OK" — Create.
- "Discard recording?" (¿Descartar grabación?) / "Your voice-over hasn’t been saved." / "Discard" · "Keep recording" — editor.
- "Music from Files" (Música de Archivos) / "Use only music you own or have rights to." / "OK" · "Cancel" — editor.
- "Rename cut" (Cambiar nombre del corte) [campo "Name"] / "Save" · "Cancel"; "Add text" (Añadir texto) [campo "Text"] / "It appears on the video at the playhead for 3 seconds." / "Add" · "Cancel"; "Rename section"; "New brand or product" (campos "Brand (e.g. CeraVe)", "Product (e.g. Hydrating Cleanser)") / "Use it" o "Move here" · "Cancel"; "Add clips" (hoja: "Choose from Photos" · "Record" · "Cancel").
- "Your plan ends early" (Tu plan termina antes) / mensaje / "OK"; "Post on TikTok now?" / "<producto> will be marked as posted and move to Edits › Posted. We'll open TikTok with the video and the caption copied." / "Yes, I'm posting it" · "Not now"; "Edit caption" [campo "Caption"] / "Save" · "Cancel" — Plan.

### 46. Notas en línea (no son pantallas)
- Edits: `WINote` "You can close the app" (Puedes cerrar la app) / "Editing keeps going on our servers. We’ll notify you as each cut is ready." y la nota de grupo "We'll notify you when your videos are ready. You can keep using CutSell." — ícono `WINoteIcon` 20 en cuadro 36 blanco borde #E6E9EF, título Bold 14, texto Medium 12 #6B7385, tarjeta 3D blanca.
- Processing: "Your video keeps uploading if you leave this screen or CutSell." / "Processing continues if you leave this screen." con ícono `WIInfo`; trabado: texto #B26A00 (fuera de paleta) + "Try again" Bold 15 #1769FF.
- Cámara: "Recording stopped — no voice for 1 min" (La grabación se detuvo: 1 min sin voz) / "We didn't hear any voice" (No escuchamos ninguna voz) — pertenece a Record (otra parte).

### Lo que NO existe (pedido en el alcance)
- Pantalla de "sin minutos / límite del plan": no existe (el 30 de Plan & Usage no se aplica).
- Pantalla de "sesión expirada": no existe (401 → Bienvenida sin aviso).
- Pantalla de "sin conexión" general: no existe; solo el detalle "Error: …" en 31 y alertas "CutSell".
- Paywall / prueba / compra: no existe.

---

## E. TOKENS: Figma "00 · Componentes y colores" (3003:5863) vs `Theme.swift` / `WhiteIndigo.swift`

| Figma | Hex Figma | Código | ¿Igual? |
|---|---|---|---|
| Azul marca | #1769FF | `CS.accent` 0x1769FF | ✅ |
| Texto | #0B1020 | `CS.wiInk` 0x0B1020 | ✅ |
| Texto gris | #6B7385 | `CS.wiSecondary` 0x6B7385 | ✅ |
| Fondo | #FFFFFF | `CS.wiPage` 0xFFFFFF | ✅ (pero casi todas las pantallas usan `HomeBackdrop` degradado, no blanco) |
| Tarjeta | #F8F9FB | `CS.wiCard` 0xF8F9FB | ✅ token / 🟠 uso: `wiCard()` por defecto pinta 3D blanco #FFFFFF→#EEF3FF |
| Superficie | #F3F5F9 | `CS.wiSurface2` 0xF3F5F9 | ✅ |
| Borde | #E6E9EF | `CS.wiBorder` 0xE6E9EF | ✅ |
| Azul suave | #EEF3FF | `CS.accentSoft` 0xEEF3FF | ✅ |
| Éxito | #1FA971 | `CS.wiSuccess` 0x1FA971 | ✅ |
| Error | #E5484D | `CS.wiError` 0xE5484D | ✅ token / 🟠 `CS.error` = #FF4D67 se usa en Account (Sign out, Delete account) |
| Aviso | #F5A524 | `CS.wiWarning` 0xF5A524 | ✅ token / 🟠 Connections usa #FFB020 |
| Timeline | #E9ECF1 | no hay token; escrito a mano en `EditorV4Timeline`, `EditorV4View`, `EditorV5Panels` | 🟠 |
| Fuente Plus Jakarta Sans | — | `CS.inter()` → PlusJakartaSans-* | ✅ (pero SF Pro en `YouRow`, `YouChoiceRow`, Welcome, alertas, Diagnostics) |

Componentes:
- "Logo / Horizontal" 3003:5928 → asset `LogoHorizontal` (`BrandLogo` 172) ✅.
- "Ring de espera (gira)" 3003:5965 → `BrandRing` ✅ (mismo degradado, una vuelta / 2,5 s).
- "Barra de abajo" 3003:5986 → `CSTabBar` 🟠 (ver 14).
- "Botón/Principal" 3003:5990: #1769FF plano, 350×52, radio 26, Bold 16 blanco → `WIPillButton(.primary)` / `PrimaryButton`: 🟠 **3D** (#2F7BFF→#0A55F0 + base #0636B8). `PrimaryButton` usa SemiBold (no Bold).
- "Botón/Neutro" 3003:5994: #0B1020 plano → `WIPillButton(.neutral)`: 🟠 3D (#2A3045→#0B1020). Processing usa `CS.neutralButton` plano ✅.
- "Botón/Secundario" 3003:5998: blanco + **borde 1,5 #0B1020**, Bold 16 #0B1020 → `WIPillButton(.outline)` / `SecondaryButton`: 🟠 blanco 3D **sin borde**.
- "Volver / Redondo (pantallas)" 3003:6003: 40×40, **#F3F5F9 + borde 1 #E6E9EF**, radio 20 → `WIHeaderIcon("WIBack")`: 🟠 círculo blanco 3D sin borde.
- "Volver / Editor (cuadrito)" 3003:6008: 26×26 — del editor, fuera de mi alcance (no comparado).
- Colores extra en el código que NO están en la paleta Figma: #0A55F0, #2F7BFF, #0636B8, #D3DDF5 (3D), #E7EEFF/#F5F8FF/#4A5BFF (backdrop), #FF4D67, #0E46D9, #FFB020, #D92D20/#7A271A/#FEF3F2/#FDA29B (aviso de borrado), #4A5263 (títulos de Processing), #B26A00, #33D98C (`wiReady`), #D5D9E2/#D5DAE3 (asas), #591F2E/#404D6B (intro).

## F. Pantallas con restos del DISEÑO OSCURO VIEJO o fuera de White Indigo
- Ninguna pantalla de mi alcance se ve oscura en build 78 (la app está forzada a claro), salvo el instante negro #070910 al abrir la cámara (estado "checking").
- Restos del oscuro: `YouStyle` y `LegalView`/`AIConsentSheet`/`GuidePage` siguen escritos con hex oscuros (#050814, #0C1730, #0A1326, #0F1726, #33405C…) que solo se ven bien porque `LightPalette` los convierte; si `LightPalette` o el modo forzado cambian, vuelven a verse oscuros.
- `ReadyBanner` (43): borde y brillo "cyan" del diseño viejo, logo con blend `.screen`.
- Intro (6): barras #591F2E / #404D6B del diseño viejo.
- No White Indigo (sistema): Splash de lanzamiento (1), error de arranque (3), textos de Bienvenida (4), alertas (5, 17, 44, 45), Diagnostics (29), barra de LegalView (25).
- Fuente SF Pro en vez de Plus Jakarta: títulos de fila de Account, Privacy, Preferences, Connections, Storage, Plan & Usage (`YouRow`) y opciones de Language (`YouChoiceRow`).
- Rojos fuera de paleta: #FF4D67 (Account), aviso de borrado (19).

---

## Navegación
1. Tocar ícono → Launch Screen (iOS) → Splash (logo + anillo) + diálogo de notificaciones de iOS.
2. Splash → (error) "Couldn’t start CutSell" → "Try Again" → Splash.
3. Splash → (sin sesión / invitado viejo / 401) Bienvenida → Sign in with Apple → (si quedó algo sin mover) alerta "Signed in" → Tabs.
4. Splash → (sesión Apple válida) Tabs (Create).
5. Tabs (primera vez o "Replay intro") → Intro 1 → 2 → 3 → 4 → "Get started"/"Skip" → Create.
6. Create → Record → (sin permiso) Permiso Cámara → Permiso Micrófono → cámara. Error de toma → "Couldn’t add this clip".
7. Create → Upload → (Fotos negado) Permiso Galería · (limitado) "Gallery access" · (11 clips) "Clip limit reached".
8. Create → crear corte → (primera vez) Aviso de IA → Processing → (falla) "Upload paused" / "Couldn’t start the cut" / "Unsupported media" → "View supported formats" → "Supported video files"; (cola) Queued; (falla servidor) Processing issue.
9. Cualquier pestaña → toast "<x> is ready · View" / toast de grupo / banner de export lista → corte / Edits.
10. Tabs → Profile → [aviso de borrado] → tarjeta cuenta / "Account & Security" → Display name (alerta) · Sign out (hoja) → Bienvenida · Delete account (hoja) → Bienvenida (o alerta de error).
11. Profile → tarjeta azul / "Plan & Usage".
12. Profile → "Your results".
13. Profile → "Privacy & Data" → Ajustes iOS · Privacy Policy / Terms of Use (hoja).
14. Profile → "Preferences" → "App language" → App Language · "Storage" → Storage.
15. Profile → "Connections".
16. Profile → "Help & Legal" (página "Help & Support") → "Report a Problem" → Diagnostics → "Send report" · "Contact Support" → Mail · "How to use CutSell" → Guías → Guía → "Try … now" → pestaña de la función · "Replay intro" → Intro.
17. Editor → Export → (falla) Export failed. Plan → Needs attention. TikTok → "Did you post it?".

## Frames de Figma de mis páginas que NO existen en la app
- 1176:2888 "Hoja · Video muy largo / muy pesado" — el código existe (`LimitScreen`) pero **nunca se muestra** en build 78 (límites apagados).
- 1174:26 "Card/Plan" con "CUTSELL PRO" / "62 of 250" — la app no tiene plan Pro (muestra "THIS MONTH" / "of 30").
- (Todos los demás frames de 07 y 08 tienen pantalla equivalente, con diferencias.) Página 00: "Volver / Editor (cuadrito)" 3003:6008 no comparado (editor).

## No se usan (código que no se alcanza)
- `AppleSignInSheet` (AppleSignIn.swift) — nadie lo abre.
- `AppState.continueWithoutAccount()` — no hay botón.
- `AppearancePage` (YouView.swift) — la fila está dentro de `if false`.
- `LimitScreen` ("This video is too long" / "Upload limit exceeded") — `sizeAndLengthLimitsOn = false`.
- `WIOriginalSafeCard` (WhiteIndigo.swift) — sin uso.
- `WIStatusBadge` tipos `.error` e `.info` — solo se usa `.warning` (en `LimitScreen`, inalcanzable).
- `ReadyBanner` por defecto ("Your cut is ready" / "Tap to review it") — `CutTracker.readyBanner` nunca se asigna; solo se ve la variante de export.
- `PermissionScreen.Kind.buttonTitle` ("Allow camera" / "Allow microphone" / "Allow gallery") — el botón usa "Continue"/"Open Settings".
- `ProblemScreen.Tone.info` — no cambia nada visual.
- `HelpPage.legal` / su `.sheet` — ninguna fila lo abre.
- Ramas de invitado en Perfil ("Guest", "Using CutSell without an account.", "No account (this iPhone only)", botón Apple en Account) — en la práctica inalcanzables (los invitados van a Bienvenida).
- `ProductsView` no se abre desde Perfil (solo desde Edits / guía "Products").

## Conteo (pantallas/popups con comparación Figma)
- ✅ igual: **0**
- 🟠 distinta: **12** — Barra de pestañas (14), Profile (15), Upload paused (30), Couldn’t start the cut (31), Unsupported media (32), Clip limit (34), Couldn’t add this clip (35), Hoja muy largo (36, inalcanzable), Processing issue (37), Queued (38), Export failed (39), Needs attention (40, comparación no confirmada). (30 y 31 usan la plantilla de 1233:3066 sin frame propio.)
- 🔴 no existe en Figma (mis páginas): **33** — 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 41, 42, 43, 44, 45 (grupo de alertas), 46 (notas).
- Tokens de color: 11 de 12 iguales (Timeline #E9ECF1 sin token).
