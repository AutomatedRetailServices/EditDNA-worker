# Bitácora de builds de CutSell (iOS / TestFlight)

Este archivo es el registro permanente de cada versión de la app de iPhone de CutSell que se mandó a TestFlight. El número de build es el mismo número que muestra TestFlight. El más nuevo va primero (78 arriba, 1 abajo). Las fechas y horas están en hora del Este (ET), y son el momento en que corrió la subida a TestFlight. Cuando dice "commit" quiere decir "versión guardada del código": el "commit de la app" es el código que se armó para ese build, y el "commit del pedido a TestFlight" es una versión chiquita que a veces se guarda solo para pedir la subida (en ese caso aparece aparte). La "rama" es la línea de trabajo de donde salió ese código, según el historial de git.

Regla: después de CADA build nuevo, agregar su entrada al inicio de este archivo, en el mismo commit del build. Si el build cambia pantallas, actualizar también docs/design-audit/.

## Resumen

| Build | Fecha (ET) | Resultado | Rama | Commit | En una línea |
|---|---|---|---|---|---|
| 78 | 2026-11-09 13:44 | ✅ subido | feature/clon-ar-autopausa | `71ae403e` | Modos Talk / B-roll en Grabar; la cámara ya cabe en la pantalla |
| 77 | 2026-11-08 20:52 | ✅ subido | feature/clon-ar-autopausa | `e5895e3a` | Clon AR (beta), auto-pausa por voz y B-roll en Grabar; aviso de borrado |
| 76 | 2026-11-07 15:18 | ✅ subido | build/apple-signup | `a2bb7fd0` | Sign in with Apple para todos; los invitados deben iniciar sesión |
| 75 | 2026-11-07 15:15 | ❌ falló | build/apple-signup | `f55d379c` | Falló la subida, no llegó a TestFlight |
| 74 | 2026-11-06 18:11 | ✅ subido | build/apple-signup | `ec168b3c` | La primera vez pide Sign in with Apple; no más cuentas de invitado nuevas |
| 73 | 2026-11-06 17:03 | ✅ subido | build/privacy-create | `04c49670` | Estilo 3D suave en Create; texto de privacidad corregido |
| 72 | 2026-11-04 18:43 | ✅ subido | build/clon-and-style | `96b960a1` | Estilo 3D suave en todas las pantallas que quedaban planas |
| 71 | 2026-11-04 14:24 | ✅ subido | build/clon-and-style | `980368ae` | "Plan my week" pasa a llamarse "Week Plan" |
| 70 | 2026-11-04 12:49 | ✅ subido | build/clon-and-style | `f8fd8824` | Plan rediseñado; armar el plan en un solo paso; texto se mueve con los dedos |
| 69 | 2026-10-30 21:37 | ✅ subido | build/clon-and-style | `f68f6c28` | Ajustes del Texto y la hoja Fonts / Style |
| 68 | 2026-10-30 19:33 | ✅ subido | build/clon-and-style | `32e1f587` | Editor: nueva barra, herramienta Texto, Duplicar, hoja Fonts / Style |
| 67 | 2026-10-30 12:07 | ✅ subido | build/clon-and-style | `1b839550` | Aviso "casi no hay voz en este video" en el editor |
| 66 | 2026-10-29 21:45 | ✅ subido | build/clon-and-style | `0e8a9663` | Videos de galería sin voz ya no fallan; Clon ya no sale más claro |
| 65 | 2026-10-29 20:49 | ✅ subido | build/clon-and-style | `6b1ff40b` | Solo pruebas automáticas del Clon |
| 64 | 2026-10-29 19:58 | ✅ subido | build/clon-and-style | `3df55022` | Split se convierte en Clon (beta) con silueta |
| 63 | 2026-10-29 15:58 | ✅ subido | feature/home-bubbly | `96140e74` | Ícono de la app con la bolsa más grande |
| 62 | 2026-10-29 13:09 | ✅ subido | feature/home-bubbly | `8283de25` | Estilo 3D "bubbly" en todas las pantallas y en el editor |
| 61 | 2026-10-29 10:41 | ✅ subido | feature/home-bubbly | `0fa60d58` | Nuevo inicio de Create en 3D; videos pesados empiezan más rápido |
| 60 | 2026-10-28 10:49 | ✅ subido | build/split-and-speed | `670442d0` | Mismo código del build 59, subido otra vez |
| 59 | 2026-10-27 19:20 | ✅ subido | build/split-and-speed | `670442d0` | Nuevo "Split clone" en Grabar |
| 58 | 2026-10-27 17:44 | ✅ subido | feature/white-indigo | `d26267d9` | Silenciar / volumen por parte ya funciona de verdad |
| 57 | 2026-10-27 15:32 | ✅ subido | feature/white-indigo | `fa43632e` | Volumen dice qué parte cambia; botones sin cortarse |
| 56 | 2026-10-26 20:25 | ✅ subido | feature/white-indigo | `1475a61f` | "Your results" más animado |
| 55 | 2026-10-26 18:45 | ✅ subido | feature/white-indigo | `5850059a` | "Your results" por semana, mes y total |
| 54 | 2026-10-26 18:19 | ✅ subido | feature/white-indigo | `d7a1e05b` | Nueva pantalla Perfil › Your results |
| 53 | 2026-10-26 17:41 | ✅ subido | feature/white-indigo | `e75e9a7b` | Una sola entrada al Plan desde Edits, Products y Export |
| 52 | 2026-10-26 14:48 | ✅ subido | feature/white-indigo | `19a27279` | Edits › Posted y confirmar publicación en TikTok |
| 51 | 2026-10-26 13:03 | ✅ subido | feature/white-indigo | `96f72a49` | Plan "opción B": los próximos 7 días |
| 50 | 2026-10-25 16:10 | ✅ subido | feature/white-indigo | `82387ee8` | Inicio del Plan por día; elegir varios videos para el Plan |
| 49 | 2026-10-24 17:39 | ✅ subido | feature/white-indigo | `97ffa9d3` | Plan my week: filas Desde / Hasta que caben |
| 48 | 2026-10-24 17:24 | ✅ subido | feature/white-indigo | `3dc96de1` | This week con dos videos por fila |
| 47 | 2026-10-24 17:07 | ✅ subido | feature/white-indigo | `2c84b832` | Plan my week en una sola pantalla; Export "WHERE TO?" |
| 46 | 2026-10-24 14:47 | ✅ subido | feature/white-indigo | `415f1987` | Plan · Today con un solo botón de TikTok |
| 45 | 2026-10-24 12:25 | ✅ subido | feature/white-indigo | `6f79bd27` | Anillo de carga único; pantallas de error en White Indigo |
| 44 | 2026-10-23 21:56 | ✅ subido | feature/white-indigo | `6a1ec805` | Editor en colores White Indigo |
| 43 | 2026-10-23 20:09 | ✅ subido | feature/white-indigo | `939d8204` | Cámara, Perfil y hojas en White Indigo |
| 42 | 2026-10-23 18:50 | ✅ subido | feature/white-indigo | `d21b4f2d` | Hoja "What are you promoting?" antes de crear |
| 41 | 2026-10-23 17:57 | ✅ subido | feature/white-indigo | `a29a6e0c` | Nuevo Create White Indigo; avisos push de Apple |
| 40 | 2026-10-23 16:32 | ✅ subido | feature/white-indigo | `707ac939` | Primer build con el diseño blanco "White Indigo" |
| 39 | 2026-10-23 11:20 | ✅ subido | feature/ios-build24 | `61f37247` | Bulk / Mix: un solo aviso cuando todo está listo |
| 38 | 2026-10-23 05:12 | ✅ subido | feature/ios-build24 | `39a80d21` | Aviso "está tardando un poco más" |
| 37 | 2026-10-22 21:59 | ✅ subido | feature/ios-build24 | `568af596` | Exportar en el iPhone desde el original (hasta 4K) |
| 36 | 2026-10-22 18:01 | ✅ subido | feature/ios-build24 | `97acf237` | Videos pesados se aligeran antes de subir |
| 35 | 2026-10-22 13:16 | ✅ subido | feature/ios-build24 | `9c162aa2` | Solo modo oscuro por ahora |
| 34 | 2026-10-22 12:45 | ✅ subido | feature/ios-build24 | `277e759c` | Intro, guía de uso y teleprompter |
| 33 | 2026-10-22 09:43 | ✅ subido | feature/ios-build24 | `8eefff75` | Toda la app en español; nuevo menú You / Perfil |
| 32 | 2026-10-21 20:13 | ✅ subido | feature/ios-build24 | `5d96646d` | Voice over como el Figma; keyframes tipo CapCut; Post to TikTok |
| 31 | 2026-10-21 12:39 | ✅ subido | feature/ios-build24 | `382668b7` | Plan · Goal v2; menú ••• en Edits |
| 30 | 2026-10-20 21:50 | ✅ subido | feature/ios-build24 | `77f3dd22` | Plan my week (Meta / Semana / Día) con recordatorios |
| 29 | 2026-10-20 16:33 | ✅ subido | feature/ios-build24 | `b418c31d` | Mix nuevo; Mix y Bulk pasan a Plan |
| 28 | 2026-10-20 11:22 | ✅ subido | feature/ios-build24 | `02899145` | Edits · Products (marcas y productos) |
| 27 | 2026-10-19 13:28 | ✅ subido | feature/ios-build24 | `ae01af99` | Exportar con mensajes claros y "Try again" |
| 26 | 2026-10-19 12:21 | ✅ subido | feature/ios-build24 | `894eca38` | Bulk manda datos de grupo al servidor |
| 25 | 2026-10-18 18:50 | ✅ subido | feature/ios-build24 | `b5eeb02b` | Sin "Step x of 8" en Procesando |
| 24 | 2026-10-18 17:56 | ✅ subido | feature/ios-build24 | `ba4b4e01` | Procesar en segundo plano; Edits · Processing |
| 23 | 2026-10-18 14:05 | ✅ subido | feature/ios-build23 | `ba4cc251` | "Audio primero" e "iCloud liviano" |
| 22 | 2026-10-18 12:24 | ✅ subido | feature/ios-build22 | `1c9dc50b` | Subidas que terminan aunque la app se cierre |
| 21 | 2026-10-16 21:54 | ✅ subido | feature/figma-part1 | `8e67e7ec` | Subidas lentas y en segundo plano que terminan |
| 20 | 2026-10-16 21:21 | ✅ subido | feature/figma-part1 | `8ed95b4c` | Partes del video por una conexión normal |
| 19 | 2026-10-16 21:00 | ✅ subido | feature/figma-part1 | `8cd2f22c` | Ya no se queda en "Uploading 0 %" |
| 18 | 2026-10-16 17:02 | ✅ subido | feature/figma-part1 | `04cfff45` | Arreglos de lo que falló en el iPhone con el build 17 |
| 17 | 2026-10-16 11:36 | ✅ subido | feature/figma-part1 | `64e107b4` | Sonidos, volumen por clip, limpiar voz, keyframes |
| 16 | 2026-10-15 16:06 | ✅ subido | feature/figma-part1 | `30e8f9c3` | Editor v5: Audio, Volumen, Velocidad; privacidad y términos |
| 15 | 2026-10-14 17:11 | ✅ subido | feature/figma-part1 | `8040c98b` | Overlays y B-roll reales; arrastrar clips; exportar mejorado |
| 14 | 2026-10-14 12:51 | ✅ subido | feature/figma-part1 | `b517efe3` | Recortar clips por los bordes; voice over mejorado |
| 13 | 2026-10-13 20:52 | ✅ subido | feature/ios-captions-v2 | `993ebbb2` | Editor más rápido y una sola forma de guardar |
| 12 | 2026-10-13 19:39 | ✅ subido | feature/ios-captions-v2 | `9b3aae67` | Rediseño del Editor v4; Espejo en la cámara |
| 11 | 2026-10-13 18:07 | ✅ subido | feature/ios-captions-v2 | `d3a7ee26` | Editor v4 del Figma |
| 10 | 2026-10-13 12:55 | ✅ subido | feature/ios-captions-v2 | `92d0e94c` | B-roll visible en la pista de video |
| 9 | 2026-10-13 03:02 | ✅ subido | feature/ios-captions-v2 | `b3a4572c` | Editor v2 del Figma reemplaza el editor viejo |
| 8 | 2026-10-12 20:29 | ✅ subido | feature/ios-captions-v2 | `2356d0c2` | Arreglos del editor tras la revisión en el iPhone |
| 7 | 2026-10-12 19:49 | ✅ subido | feature/ios-captions-v2 | `83d0c51e` | El editor por fin carga el borrador |
| 6 | 2026-10-12 19:06 | ✅ subido | feature/ios-captions-v2 | `528e70d2` | Home Hub, Procesando con progreso real, pantalla Ready |
| 5 | 2026-10-12 18:14 | ✅ subido | feature/ios-captions-v2 | `f113e707` | Arreglo: los videos no se subían desde el iPhone |
| 4 | 2026-10-12 16:42 | ✅ subido | feature/ios-captions-v2 | `f2bcf167` | Diseño Figma parte 1: pestañas, cámara, galería, Edits, You |
| 3 | 2026-10-12 13:04 | ✅ subido | feature/ios-captions-v2 | `3e3c6814` | Primer build que llegó a TestFlight |
| 2 | 2026-10-12 13:03 | ❌ falló | feature/ios-captions-v2 | `d90bfc50` | Segundo intento, no llegó a TestFlight |
| 1 | 2026-10-12 13:01 | ❌ falló | feature/ios-captions-v2 | `d90bfc50` | Primer intento, no llegó a TestFlight |

---

## Build 78 — 2026-11-09
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/clon-ar-autopausa  ·  **Commit de la app:** `71ae403e`  ·  **Commit del pedido a TestFlight:** `42720ba5`
- **Qué cambió:**
  - En Grabar, la usuaria elige el modo con chips: **Talk** o **B-roll** (se quitó la detección de cara / persona).
  - Talk = auto-pausa por voz (los silencios de más de 0,7 s no se guardan).
  - B-roll = se guarda todo lo grabado, sin auto-pausa ni parada automática, y el clip queda marcado como B-roll (en el archivo y como tipo de clip "broll").
  - En la galería, cada clip tiene su interruptor "B-roll".
  - La cámara nunca es más alta que la pantalla: la hoja "Choose your style" vuelve a subir completa; el aviso de B-roll flota.
  - El interruptor de Auto-pausa vuelve al azul #1769FF.
  - Pruebas automáticas nuevas: modos a/b/c y que la cámara quepa en un iPhone Pro Max y en un SE.
- **Pantallas nuevas o cambiadas:** Cámara (Grabar), Choose your style, Galería, Mix, pantalla de subida.
- **Pendiente o con problemas:**
  - El motor del servidor todavía no usa la marca de B-roll: sigue cortando silencios en los clips B-roll y no los pone encima de la voz (el motor no se cambió).
  - Falta revisar Clon, auto-pausa y B-roll en un iPhone real.
  - La auditoría de diseño completa de este build está en docs/design-audit/build-78/.

## Build 77 — 2026-11-08
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/clon-ar-autopausa  ·  **Commit de la app:** `e5895e3a`  ·  **Commit del pedido a TestFlight:** `e58560bd`
- **Qué cambió:**
  - Se quitó el "Split clone" de 2 tomas.
  - Nuevo **Clon AR** (beta): hasta 4 clones, usa ARKit y el video se arma en el mismo iPhone.
  - Auto-pausa por voz en Grabar.
  - Botón de B-roll con detección de cara / persona; los tramos de B-roll quedan marcados en el video.
  - Suscripción cancelada: aviso con fecha "todo se va a borrar y no se puede recuperar" en Create y en You (solo aparece si el servidor dice que la suscripción está cancelada).
  - Si la cuenta se va (borrada, suscripción terminada o sesión cerrada) se borran del iPhone todos los recordatorios del Week Plan.
  - Política de privacidad: cuando termina una suscripción cancelada, se borra todo.
  - 22 pruebas automáticas en el simulador, con datos de prueba.
- **Pantallas nuevas o cambiadas:** Cámara (Grabar: Clon AR, Auto-pausa, B-roll), inicio de Create (aviso), You (aviso), Política de privacidad.
- **Pendiente o con problemas:**
  - En el iPhone, la hoja "Choose your style" no subía completa (el botón Create quedaba escondido) porque la columna derecha de la cámara quedó muy alta. Se arregló en el build 78.
  - Los interruptores de Auto-pausa / B-roll salían turquesa en vez del azul #1769FF. Se arregló en el build 78.
  - Falta revisar en un iPhone real los tiempos del Clon y de la auto-pausa.

## Build 76 — 2026-11-07
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/apple-signup  ·  **Commit de la app:** `a2bb7fd0`  ·  **Commit del pedido a TestFlight:** `fb61e60e`
- **Qué cambió:**
  - Sign in with Apple para todos: no se crean cuentas de invitado nuevas.
  - Los teléfonos que seguían con cuenta de invitado inician sesión con Apple una vez después de actualizar, y sus videos pasan a esa cuenta.
  - Proceso de build: las herramientas de revisión de TestFlight corren aparte (por eso falló el 75); la llave de App Store solo se usa en los dos pasos que la necesitan.
  - Según la sesión de la Product Owner, este build también trae el aviso de borrado de cuenta con fecha, la cancelación de recordatorios al cerrar sesión y el texto de privacidad sobre el fin de la suscripción. Según el historial de git, esos tres cambios se guardaron el 8 de noviembre y entraron en el build 77.
- **Pantallas nuevas o cambiadas:** Bienvenida / inicio de sesión con Apple.
- **Pendiente o con problemas:** sin datos

## Build 75 — 2026-11-07
- **Resultado:** falló (no llegó a TestFlight)
- **Rama (según el historial de git):** build/apple-signup  ·  **Commit de la app:** `f55d379c`  ·  **Commit del pedido a TestFlight:** `7c55ee3d`
- **Qué cambió:**
  - Falló la subida, no llegó a TestFlight.
  - Se intentaba subir: los teléfonos con cuenta de invitado deben iniciar sesión con Apple una vez y sus videos pasan a la cuenta.
  - También cambios de seguridad y de pruebas del proceso de build.
- **Pantallas nuevas o cambiadas:** ninguna (no llegó a TestFlight).
- **Pendiente o con problemas:** la subida falló; 3 minutos después salió el build 76 con el mismo código y un arreglo de las herramientas del proceso de build.

## Build 74 — 2026-11-06
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/apple-signup  ·  **Commit de la app:** `ec168b3c`  ·  **Commit del pedido a TestFlight:** `f609bce6`
- **Qué cambió:**
  - La primera vez que se abre, la app pide Sign in with Apple (no más cuentas de invitado nuevas).
  - Pantalla de bienvenida con la marca.
  - Las sesiones de invitado ya guardadas siguen funcionando.
  - Sign in with Apple funcionando (comprobado en el iPhone de la Product Owner).
- **Pantallas nuevas o cambiadas:** Bienvenida / inicio de sesión con Apple.
- **Pendiente o con problemas:** los teléfonos que seguían como invitados todavía no tenían que iniciar sesión; eso llegó en el build 76.

## Build 73 — 2026-11-06
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/privacy-create  ·  **Commit de la app:** `04c49670`  ·  **Commit del pedido a TestFlight:** `c9c76e62`
- **Qué cambió:**
  - Estilo 3D "suave" para las burbujas de Create y los botones Grabar / Subir.
  - Texto de privacidad de la IA corregido para que diga la verdad (solo palabras, sin imagen; el producto solo se sugiere si no se eligió).
  - Al borrar la cuenta, la app manda el código de Apple para que también se cancele el inicio de sesión con Apple; mensaje amable para reintentar si el borrado no terminó.
- **Pantallas nuevas o cambiadas:** inicio de Create, Aviso de IA, You (borrar cuenta), Política de privacidad.
- **Pendiente o con problemas:** sin datos

## Build 72 — 2026-11-04
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `96b960a1`  ·  **Commit del pedido a TestFlight:** `bc3fd890`
- **Qué cambió:**
  - Estilo 3D suave (relieve bajo) en toda la app, donde quedaba el diseño viejo plano (pedido de la Product Owner: "busca todas las pantallas con el diseño viejo").
  - Plan: el botón del Week Plan, los días y las tarjetas con relieve más bajo (estaba muy alto).
  - This week y la hoja del video con tarjetas suaves en vez de bordes grises.
  - Botones compartidos, tarjetas del Perfil, barra del editor y hoja de Fonts en el mismo estilo.
  - También: Export, Clon, alertas, guía, Procesando, selector de producto, Products y TikTok.
- **Pantallas nuevas o cambiadas:** Plan, This week, Week Plan, Export, Clon, alertas, guía de uso, Procesando, Products, selector de producto, Editor, Perfil, Post to TikTok.
- **Pendiente o con problemas:** las burbujas grandes del inicio de Create quedaron con su 3D fuerte "por ahora"; pasaron al estilo suave en el build 73.

## Build 71 — 2026-11-04
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `980368ae`  ·  **Commit del pedido a TestFlight:** `c55f5dab`
- **Qué cambió:**
  - "Plan my week" se llama **Week Plan** en todos lados (en español: "Plan de la semana").
  - Desde Edits, Week Plan muestra solo los videos elegidos en una tira (1, 2, 3...), no todos otra vez.
  - Botones en 3D suave en Week Plan y en Edits › Add to Plan.
- **Pantallas nuevas o cambiadas:** Plan, Week Plan, inicio de Create, guía de uso, Export, Edits.
- **Pendiente o con problemas:** sin datos

## Build 70 — 2026-11-04
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `f8fd8824`  ·  **Commit del pedido a TestFlight:** `d5e5111b`
- **Qué cambió:**
  - Armar el plan en un solo pedido al servidor (antes era uno por video, ~11 s cada uno, y se cortaba al bloquear el iPhone); sigue guardando aunque se bloquee el iPhone.
  - Plan rediseñado: sin "In progress", Week Plan primero como un botón 3D grande, días y videos como tarjetas 3D flotantes, un solo enlace "See the whole week".
  - Texto sobre el video: ahora sí se mueve y cambia de tamaño con los dedos (queda encima de fotos y subtítulos).
- **Pantallas nuevas o cambiadas:** Plan, Week Plan, This week, Editor (texto sobre el video).
- **Pendiente o con problemas:** sin datos

## Build 69 — 2026-10-30
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `f68f6c28`
- **Qué cambió:**
  - Cambios pedidos tras probar el build 68:
  - Al tocar "Listo" en el teclado, el texto nuevo aparece en el video al momento (seleccionado, se puede mover y cambiar de estilo).
  - Hoja Fonts / Style: arrastrar hacia arriba la agranda, hacia abajo la achica o la cierra; mientras se escribe solo se ve el campo de texto sobre el teclado.
- **Pantallas nuevas o cambiadas:** Editor (Texto, hoja Fonts / Style).
- **Pendiente o con problemas:** el 31 de octubre la Product Owner reportó que el texto no se movía ni cambiaba de tamaño con los dedos; se arregló en el build 70.

## Build 68 — 2026-10-30
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `32e1f587`
- **Qué cambió:**
  - Barra del editor nueva: Edit · Audio · Text · Captions · Overlay · Style.
  - Edit abre Split, Delete, Duplicate (nuevo), Mirror, Rotate; Audio abre Voice over, Music, Volume, Speed.
  - Herramienta Texto nueva: tocar para elegir, tocar otra vez para cambiar el estilo, arrastrar y pellizcar.
  - Hoja Fonts / Style para subtítulos y textos: 33 letras, 10 estilos "Aa", colores de texto, resaltado, borde, fondo, sombra y espaciado (24 letras nuevas, las mismas que usa el servidor).
  - Hoja "What are you promoting?": una fila por marca (A-Z) que abre sus productos; agregar producto o marca en un paso; Products en orden A-Z.
- **Pantallas nuevas o cambiadas:** Editor (barra, Texto, Fonts / Style), hoja "What are you promoting?", Products.
- **Pendiente o con problemas:** la Product Owner pidió cambios del teclado y de la hoja de estilos al probarlo; llegaron en el build 69.

## Build 67 — 2026-10-30
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `1b839550`
- **Qué cambió:**
  - Editor: aviso "Almost no voice in this video. Check the cut." cuando el servidor cortó un video con muy poca voz (se muestra una vez, 6 s, al abrir).
- **Pantallas nuevas o cambiadas:** Editor (aviso).
- **Pendiente o con problemas:** el aviso depende de un cambio del servidor (rama fix/short-speech-cut); el historial no dice si ya está publicado.

## Build 66 — 2026-10-29
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `0e8a9663`
- **Qué cambió:**
  - Un video de la galería ya no falla con "no speech": si la voz sacada en el iPhone sale muda, se prueba con el original y si no, se sube el video entero.
  - Un video sin voz lo dice claro: "We couldn't hear anyone talking" (en español también), en vez de un "Error".
  - Clon: el video unido ya no sale más claro que el original (colores corregidos).
- **Pantallas nuevas o cambiadas:** Procesando (mensaje de error), Clon.
- **Pendiente o con problemas:** sin datos

## Build 65 — 2026-10-29
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `6b1ff40b`  ·  **Commit del pedido a TestFlight:** `3d8cac1c`
- **Qué cambió:**
  - Solo cambios de pruebas automáticas (la prueba del Clon compara los colores de otra forma).
- **Pantallas nuevas o cambiadas:** ninguna
- **Pendiente o con problemas:** sin datos

## Build 64 — 2026-10-29
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/clon-and-style  ·  **Commit de la app:** `3df55022`  ·  **Commit del pedido a TestFlight:** `6b1ff40b`
- **Qué cambió:**
  - "Split" pasa a ser **Clon (beta)** con el método de silueta: en la cámara aparece "Clon" con la etiqueta BETA.
  - Pantalla de inicio, "Creating your clone...", resultado con "Retake" / "Use this video", en español.
  - Se usa todo el segundo tramo y solo la persona del primero, con borde suave y colores igualados.
  - Si el recorte falla o el teléfono se movió mucho, se usan dos mitades con un centro suave y un aviso.
  - Según el documento del Clon, este build incluye también el estilo 3D y el ícono grande.
- **Pantallas nuevas o cambiadas:** Cámara, Clon.
- **Pendiente o con problemas:**
  - El video unido salía más claro; se corrigió en el build 66.
  - Límites conocidos (documento del Clon): la alineación solo corrige desplazamiento (no giro ni zoom); con audífonos Bluetooth graba el micrófono del teléfono.

## Build 63 — 2026-10-29
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/home-bubbly  ·  **Commit de la app:** `96140e74`
- **Qué cambió:**
  - Ícono de la app: el mismo, con la bolsa mucho más grande (unos 3/4 del ícono en vez de la mitad), fondo negro igual.
- **Pantallas nuevas o cambiadas:** ícono de la app.
- **Pendiente o con problemas:** sin datos

## Build 62 — 2026-10-29
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/home-bubbly  ·  **Commit de la app:** `8283de25`
- **Qué cambió:**
  - Todas las pantallas en el mismo estilo 3D "bubbly" de Create: fondo azul suave, botones, chips, selector y tarjetas en 3D.
  - Editor en 3D: botones de atrás, calidad y Export, y cada herramienta en una burbujita 3D (la abierta en azul).
  - Fondo del editor con los mismos brillos rosa / lila / azul de Create.
  - Perfil: tarjeta "This month" y secciones en 3D.
- **Pantallas nuevas o cambiadas:** Editor, Perfil y sus páginas, Plan, Week Plan, Export, Procesando.
- **Pendiente o con problemas:**
  - Dos pruebas automáticas fallan por razones de la prueba, no del estilo (según el documento del estilo 3D).
  - El 4 de noviembre la Product Owner dijo que el relieve estaba muy alto; se bajó en los builds 71 y 72.

## Build 61 — 2026-10-29
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/home-bubbly  ·  **Commit de la app:** `0fa60d58`
- **Qué cambió:**
  - Nuevo inicio de Create en 3D "bubbly": el logo de CutSell como protagonista (negro mate, rebota, letras que saltan; quieto si está activado "Reducir movimiento").
  - Botones Grabar y Subir en forma de píldora, burbujas redondas, hoja de recientes y barra de pestañas tipo cápsula.
  - Tarjeta "Stats" (Your results): reloj 3D azul que gira mientras cuenta el tiempo ahorrado y burbuja rosa con los videos editados (solo números reales).
  - Velocidad: un video 4K o pesado empieza su corte desde la voz y se aligera después; "listo" se ve antes; el corte terminado queda en el teléfono y el editor abre al momento.
- **Pantallas nuevas o cambiadas:** inicio de Create, barra de pestañas.
- **Pendiente o con problemas:** sin datos

## Build 60 — 2026-10-28
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/split-and-speed  ·  **Commit de la app:** `670442d0`  ·  **Commit del pedido a TestFlight:** `bc145e83`
- **Qué cambió:**
  - Ningún cambio nuevo: se volvió a subir el mismo código del build 59.
- **Pantallas nuevas o cambiadas:** ninguna
- **Pendiente o con problemas:** sin datos

## Build 59 — 2026-10-27
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** build/split-and-speed  ·  **Commit de la app:** `670442d0`
- **Qué cambió:**
  - Nuevo **Split clone** en Grabar (Record › Split): se graban dos partes y la app las une (versión básica y Pro).
  - Guía debajo del título, etiqueta de cada lado, en español.
  - Prueba automática con video de prueba para revisar la alineación.
- **Pantallas nuevas o cambiadas:** Cámara (entrada Split), pantallas del Split clone.
- **Pendiente o con problemas:**
  - La cámara misma (grabar, bloqueo, audífonos) necesita revisarse en un iPhone real.
  - Límites conocidos: la alineación solo corrige desplazamiento; con audífonos Bluetooth graba el micrófono del teléfono; Pro es lento en el simulador.
  - Reemplazado por el Clon con silueta en el build 64.

## Build 58 — 2026-10-27
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `d26267d9`
- **Qué cambió:**
  - Silenciar / volumen por parte ya funciona de verdad: antes una parte silenciada subía poco a poco hasta volumen completo. Arreglado en la vista previa y en la exportación del iPhone.
  - El resto son pruebas automáticas.
- **Pantallas nuevas o cambiadas:** Editor (Volumen).
- **Pendiente o con problemas:** sin datos

## Build 57 — 2026-10-27
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `fa43632e`
- **Qué cambió:**
  - Los botones del editor ya no se cortan ("Unmute" salía como "Unmu…").
  - Volumen dice qué parte cambia ("Selected part · 0:04–0:09") y cómo elegir una; la parte elegida tiene un borde azul más grueso.
- **Pantallas nuevas o cambiadas:** Editor (Volumen).
- **Pendiente o con problemas:** una parte silenciada subía hasta volumen completo; se arregló en el build 58.

## Build 56 — 2026-10-26
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `1475a61f`
- **Qué cambió:**
  - "Your results" más vivo: selector Semana / Mes / Total, números que van contando, un anillo de 24 horas que se llena con el tiempo ahorrado, barras de las últimas 8 semanas y la próxima meta.
- **Pantallas nuevas o cambiadas:** Perfil › Your results.
- **Pendiente o con problemas:** sin datos

## Build 55 — 2026-10-26
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `5850059a`
- **Qué cambió:**
  - "Your results" por Total, Este mes y Esta semana: horas ahorradas, jornadas de 8 horas, videos editados y tiempo de video.
  - Menos de una hora se muestra en minutos.
- **Pantallas nuevas o cambiadas:** Perfil › Your results.
- **Pendiente o con problemas:** sin datos

## Build 54 — 2026-10-26
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `d7a1e05b`
- **Qué cambió:**
  - Nueva sección Perfil › Your results: videos editados, tiempo total de video, horas ahorradas (20 min por video) y lo de este mes.
- **Pantallas nuevas o cambiadas:** Perfil › Your results (nueva).
- **Pendiente o con problemas:** sin datos

## Build 53 — 2026-10-26
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `e75e9a7b`
- **Qué cambió:**
  - Una sola puerta al Plan: Edits › Select, Products › Select y Export › Add to Plan abren Plan my week con los videos elegidos.
  - Plan my week permite elegir la red social ("Post on").
- **Pantallas nuevas o cambiadas:** Edits, Products, Export, Plan my week.
- **Pendiente o con problemas:** sin datos

## Build 52 — 2026-10-26
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `19a27279`
- **Qué cambió:**
  - Nueva sección Edits › Posted (por mes); se puede deshacer a mano ("Not posted", vuelve a Ready).
  - Los videos publicados o ya programados no se ofrecen en Plan my week.
  - Today: "Share to TikTok" pide confirmar; al decir que sí, lo marca como publicado, para el recordatorio y abre TikTok.
  - Plan y Edits cargan lo nuevo sin tener que deslizar para refrescar.
- **Pantallas nuevas o cambiadas:** Edits › Posted, Plan · Today, Plan my week.
- **Pendiente o con problemas:** sin datos

## Build 51 — 2026-10-26
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `96f72a49`
- **Qué cambió:**
  - Plan "opción B": la fila de días muestra los próximos 7 días desde hoy.
  - "Coming up" empieza después del día que se ve en la tarjeta (nada repetido).
- **Pantallas nuevas o cambiadas:** Plan.
- **Pendiente o con problemas:** la opción B quedó "para revisar" por la Product Owner.

## Build 50 — 2026-10-25
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `82387ee8`
- **Qué cambió:**
  - Inicio del Plan: tocar un día para verlo; los videos del día con foto, marca, producto y hora; tira "Coming up" con fotos grandes.
  - Si hoy no hay nada, se muestra primero el próximo día con videos.
  - Products: "Rename" pone marca y producto.
  - Edits y producto: elegir varios videos › Add to Plan (uno: hoja de horario; varios: Plan my week en ese orden).
- **Pantallas nuevas o cambiadas:** Plan, Products, Edits.
- **Pendiente o con problemas:** sin datos

## Build 49 — 2026-10-24
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `97ffa9d3`
- **Qué cambió:**
  - Plan my week · Between: filas "From" / "To" que caben en la pantalla.
- **Pantallas nuevas o cambiadas:** Plan my week.
- **Pendiente o con problemas:** sin datos

## Build 48 — 2026-10-24
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `3dc96de1`
- **Qué cambió:**
  - This week: dos videos por fila, como el Figma.
  - El selector "Between" cabe.
- **Pantallas nuevas o cambiadas:** This week, Plan my week.
- **Pendiente o con problemas:** las filas de "Between" se terminaron de acomodar en el build 49.

## Build 47 — 2026-10-24
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `2c84b832`
- **Qué cambió:**
  - Plan my week en una sola pantalla: tocar los videos en orden, Por día / Entre / Días / Empieza, mezclar productos.
  - This week: cada video abre una hoja para cambiarlo, moverlo de día u hora, o quitarlo.
  - Settings = recordatorios.
  - Export pregunta "WHERE TO?": guardar en Fotos o agregar al Plan (con hoja de horario).
  - Aviso "Time to post on {red social}". Nada se publica solo.
- **Pantallas nuevas o cambiadas:** Plan my week, This week, Export, hoja de horario.
- **Pendiente o con problemas:** el selector "Between" no cabía bien; se arregló en los builds 48 y 49.

## Build 46 — 2026-10-24
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `415f1987`
- **Qué cambió:**
  - Plan · Today: un solo botón azul "Share to TikTok", más "Save to Photos" y "Mark as posted".
  - El aviso de This week en colores claros.
- **Pantallas nuevas o cambiadas:** Plan · Today, This week.
- **Pendiente o con problemas:** sin datos

## Build 45 — 2026-10-24
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `6f79bd27`
- **Qué cambió:**
  - Un mismo anillo de colores para todas las esperas (carga del editor, envío de Sales Mix, exportar).
  - Pantallas nuevas en White Indigo: En cola, Export falló, Needs attention (Plan / Bulk), fila neutra para videos sin voz.
  - "Retry processing" en azul; saludo "Hi {nombre}".
- **Pantallas nuevas o cambiadas:** Procesando, Needs attention, Export, inicio de Create, Editor (carga), Products.
- **Pendiente o con problemas:** sin datos

## Build 44 — 2026-10-23
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `6a1ec805`
- **Qué cambió:**
  - Editor en White Indigo: paneles blancos, línea de tiempo gris, íconos negros, hoja de voice over.
  - Tarjeta de permiso del micrófono, íconos de overlay, botón de atrás chico, azul sólido.
  - "Retry processing" en la pantalla de error, también en español.
- **Pantallas nuevas o cambiadas:** Editor, Procesando (error).
- **Pendiente o con problemas:** sin datos

## Build 43 — 2026-10-23
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `939d8204`
- **Qué cambió:**
  - Botón de atrás White Indigo en las páginas del Perfil y en el editor; títulos en negrita.
  - Cámara sin logo; controles y anillo de grabar en el color de acento.
  - "Choose your style" (y todas las hojas de abajo) siempre en White Indigo, también sobre la cámara oscura.
  - Letra Plus Jakarta Sans en Perfil y listas; marco de vista previa en Export.
- **Pantallas nuevas o cambiadas:** Cámara, Choose your style, Perfil, Editor, Export, intro.
- **Pendiente o con problemas:** sin datos

## Build 42 — 2026-10-23
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `d21b4f2d`
- **Qué cambió:**
  - Hoja nueva "What are you promoting?" antes de Single / Bulk / Combine / Record / Faceless / Sales Mix: el producto viaja con los videos.
  - Edits Ready, detalle de producto y asignar en White Indigo.
  - Las pantallas del Plan y de productos mantienen la barra de abajo; botones azules de TikTok.
- **Pantallas nuevas o cambiadas:** hoja "What are you promoting?", inicio de Create, Edits, Products, Plan.
- **Pendiente o con problemas:** sin datos

## Build 41 — 2026-10-23
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `a29a6e0c`
- **Qué cambió:**
  - Nuevo Create White Indigo: solo logo, saludo, cuadrícula 2x2 (Record, Faceless, Sales Mix, Plan my week), zona punteada para subir con Single / Bulk / Combine, Recientes; barra de pestañas más delgada.
  - Se sube la versión que se ve en Fotos: un video editado (con canción, recorte o filtro) ya no se sube como el original mudo.
  - Si el servidor está ocupado, ya no es un error: el video espera "En cola" y se reenvía solo cada minuto.
  - Avisos push de Apple registrados con el servidor.
- **Pantallas nuevas o cambiadas:** inicio de Create, barra de pestañas.
- **Pendiente o con problemas:** sin datos

## Build 40 — 2026-10-23
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/white-indigo  ·  **Commit de la app:** `707ac939`
- **Qué cambió:**
  - Primer build con el diseño **White Indigo**: blanco en todos lados, letra Plus Jakarta Sans, logo nuevo, botones tipo píldora, hojas claras.
  - "Mix" pasa a llamarse "Sales Mix".
  - Barra de pestañas, Create, Edits, Procesando, Plan, Perfil, Ready, errores, permisos, anillo de exportar, intro y guía en blanco.
  - Voice over y teleprompter mantienen su diseño.
- **Pantallas nuevas o cambiadas:** casi todas: barra de pestañas, Create, Edits, Procesando, Ready, Plan, Perfil, permisos, errores, intro y guía.
- **Pendiente o con problemas:**
  - Faltaban las pantallas de error En cola, Export falló y Needs attention (llegaron en el build 45).
  - Las pantallas de "video muy largo / archivo muy grande" esperan una decisión de la Product Owner.
  - Esta rama no incluye arreglos posteriores de feat/ios-build24 que esperaban aprobación (según el documento White Indigo).

## Build 39 — 2026-10-23
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `61f37247`
- **Qué cambió:**
  - Bulk / Mix sin tiempos: desde el inicio dice "We'll notify you when your videos are ready. You can keep using CutSell."
  - Filas de Bulk: En cola / Procesando / Listo, con barra de grupo "3 of 10 ready".
  - UN solo aviso por Bulk ("Your 10 videos are ready ✨" o "9 ready, 1 needs attention").
- **Pantallas nuevas o cambiadas:** Procesando (Bulk / Mix), Edits.
- **Pendiente o con problemas:** sin datos

## Build 38 — 2026-10-23
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `39a80d21`
- **Qué cambió:**
  - Procesando: un video que lleva más de 8 minutos dice "Taking a little longer, we'll notify you" (Bulk / Mix).
- **Pantallas nuevas o cambiadas:** Procesando.
- **Pendiente o con problemas:** sin datos

## Build 37 — 2026-10-22
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `568af596`
- **Qué cambió:**
  - Exportar en el iPhone desde el original de Fotos: 4K por defecto si el original es 4K (nunca se agranda), mismos cortes y elementos que la vista previa; el servidor queda de respaldo.
  - Nueva página Perfil › Preferences › Storage.
  - Plan my week: la exportación de noche y "Post to TikTok" también hacen el video en el iPhone.
  - "Restoring video…": si el servidor borró su copia por falta de uso, se rehace desde el original en Fotos (el plazo quedó en 3 días por pedido de la Product Owner).
  - Video donde nadie habla: "We couldn't hear anyone talking in this video" con "Use in Mix" (no "Retry").
  - Copia liviana con calidad exacta (10 Mb/s, menos de 950 MB).
- **Pantallas nuevas o cambiadas:** Export, Preferences › Storage, Procesando, Plan, Mix.
- **Pendiente o con problemas:** sin datos

## Build 36 — 2026-10-22
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `97acf237`
- **Qué cambió:**
  - Un video pesado (por ejemplo, iPhone 1080p a 60) se aligera antes de subir, también los de iCloud.
  - Máximo 4 pedidos a la vez al servidor para las partes (80 a la vez fallaban en videos de 5 minutos).
  - El botón "Retry / Reintentar" crece en vez de cortar la palabra.
- **Pantallas nuevas o cambiadas:** Procesando (botón Retry).
- **Pendiente o con problemas:** sin datos

## Build 35 — 2026-10-22
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `9c162aa2`
- **Qué cambió:**
  - Solo modo oscuro por ahora: el modo día se apagó y se escondió la opción "Appearance".
- **Pantallas nuevas o cambiadas:** Perfil (sin "Appearance"), Plan.
- **Pendiente o con problemas:** el modo día quedó apagado "por ahora".

## Build 34 — 2026-10-22
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `277e759c`
- **Qué cambió:**
  - Intro de 4 pantallas de bienvenida y guía "How to use CutSell" con 9 guías y "Try now"; se puede repetir la intro desde Ayuda.
  - En la intro aparece el avatar de IA de la creadora.
  - Teleprompter en la cámara: guion en la barra lateral, se desplaza solo al grabar, velocidad, tamaño, editar, inglés / español.
  - Modo día: Appearance Automático / Oscuro / Claro (editor, cámara, procesando e intro siguen oscuros).
- **Pantallas nuevas o cambiadas:** Intro, guía de uso, Cámara (teleprompter), Perfil (Appearance), Plan.
- **Pendiente o con problemas:** el modo día se apagó en el build 35.

## Build 33 — 2026-10-22
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `8eefff75`
- **Qué cambió:**
  - Toda la app en español: Language (Automatic · English · Español), cambia al instante; días, horas y fechas en el idioma de la app.
  - Pestañas en español: Edits → "Editar", You → "Perfil".
  - Menú You como el Figma: resumen (nombre, ediciones del mes), Cuenta y seguridad, Plan y uso, Privacidad, Preferencias, Conexiones, Ayuda y legal.
  - Preferencias: idioma, apariencia, vibración, cámara y exportación, avisos ("Time to post" y "Processing complete").
  - Sign in with Apple pide el nombre.
  - Editor: ‹ › solo aparecen cuando el clip tiene keyframes.
- **Pantallas nuevas o cambiadas:** You / Perfil y todas sus páginas; textos de todas las pantallas.
- **Pendiente o con problemas:** "Connections" quedó como "SOON"; Appearance se guarda pero sigue oscuro hasta que exista la paleta clara.

## Build 32 — 2026-10-21
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `5d96646d`
- **Qué cambió:**
  - Voice over exactamente como el Figma: hoja debajo del video, cuenta regresiva 3·2·1, onda de sonido; se cierra con X, deslizando o tocando afuera; pregunta "Discard recording?" si hay algo sin guardar.
  - Panel de la voz elegida rehecho: bajar video, limpiar voz, volver a grabar, volumen 0 / 100 / 200 %, borrar.
  - Keyframes tipo CapCut: movimiento suave en todo el video, curvas por cada ◇, ‹ › para ir al anterior / siguiente, Auto Zoom AI.
  - Post to TikTok (desde Export y desde el día del Plan): el video va a Fotos, se copia el texto, se abre TikTok; al volver pregunta "Did you post it?".
  - El "+" de las filas de voz / overlay ya no choca con el borde de recorte.
- **Pantallas nuevas o cambiadas:** Editor (Voice over, keyframes, línea de tiempo), Export, Plan · Día.
- **Pendiente o con problemas:** el envío directo con TikTok Share Kit solo funciona cuando estén puestos los datos de TikTok; la fila "Faceless video" queda escondida hasta que se construya.

## Build 31 — 2026-10-21
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `382668b7`
- **Qué cambió:**
  - Plan · Goal v2: cuántos videos (−/+), horarios para publicar (hasta 8), días, inicio, recordatorio en cada horario, resumen en vivo, "Build my plan".
  - Create = Home Hub · Mix: chips Single · Bulk · Combine y fila de Mix.
  - Menú ••• en Edits: "Personal · no planear" / "Usar en el Plan"; tocar ••• ya no abre el video.
- **Pantallas nuevas o cambiadas:** Plan · Goal, inicio de Create, Edits (menú •••).
- **Pendiente o con problemas:** sin datos

## Build 30 — 2026-10-20
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `77f3dd22`
- **Qué cambió:**
  - Nuevo **Plan my week** (Meta / Semana / Día): This week real, editar y copiar textos, Share to TikTok, guardar, marcar como publicado.
  - Recordatorios "Time to post" que abren su día.
  - Los videos de mañana se exportan de noche con WiFi y cargando.
- **Pantallas nuevas o cambiadas:** Plan, Plan · Semana, Plan · Día.
- **Pendiente o con problemas:** sin datos

## Build 29 — 2026-10-20
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `b418c31d`
- **Qué cambió:**
  - Mix nuevo: Create → configurar Mix, cuántos, procesando en grupo, resultados con código MIX, exportar todos; la voz se sube primero en un solo trabajo.
  - Mix y Bulk pasan a la pestaña Plan (Nuevo Mix, Bulk, En proceso); Create queda para un solo video.
- **Pantallas nuevas o cambiadas:** Plan (inicio), Mix, inicio de Create, Edits · Processing.
- **Pendiente o con problemas:** "This week" quedó como "SOON"; llegó en el build 30.

## Build 28 — 2026-10-20
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `02899145`
- **Qué cambió:**
  - Nueva sección Edits · Products: filtros deslizables, marcas → productos, vista de producto (filtros, cuadrícula, renombrar, elegir, editar varios).
  - Hoja "Which product is this?" y sugerencia de álbum de Fotos.
- **Pantallas nuevas o cambiadas:** Edits · Products, hoja "Which product is this?".
- **Pendiente o con problemas:** sin datos

## Build 27 — 2026-10-19
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `ae01af99`
- **Qué cambió:**
  - Exportar: mensaje claro y "Try again" si no puede empezar, "Retry" en la descarga, texto "Preparing your export…".
  - Se registra cada paso de la exportación para encontrar fallas; ya no hay falsa alarma por la letra de subtítulos.
- **Pantallas nuevas o cambiadas:** Export.
- **Pendiente o con problemas:** sin datos

## Build 26 — 2026-10-19
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `894eca38`
- **Qué cambió:**
  - Cada video de un envío Bulk lleva datos del grupo, para que el servidor ahorre trabajo en ese envío.
- **Pantallas nuevas o cambiadas:** ninguna
- **Pendiente o con problemas:** sin datos

## Build 25 — 2026-10-18
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `b5eeb02b`
- **Qué cambió:**
  - Procesando ya no muestra "Step x of 8" ni puntos (Single, Bulk, Combine).
  - Bulk ya no abre la pantalla de un solo video; solo la hoja del Figma.
- **Pantallas nuevas o cambiadas:** Procesando.
- **Pendiente o con problemas:** sin datos

## Build 24 — 2026-10-18
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build24  ·  **Commit de la app:** `ba4b4e01`
- **Qué cambió:**
  - Procesar en segundo plano: "Continue in background" en cada paso; la subida y el corte siguen en cualquier pestaña.
  - Los trabajos elegidos mientras otro sube esperan su turno ("En cola").
  - Bulk: hoja "Your N videos are being edited" con "Go to Edits" / "Stay here".
  - Edits: Processing · Ready · Drafts · All con números; cada fila con estado y "Retry"; un corte terminado pasa a Ready con "<nombre> is ready · View".
  - Al salir de la app sigue unos 30 s más; al volver, continúa las subidas paradas (solo lo que falta).
  - El chip Bulk ya no abre la galería en modo Single.
- **Pantallas nuevas o cambiadas:** Edits · Processing, inicio de Create (Home Hub), Galería, hoja de Bulk.
- **Pendiente o con problemas:** Bulk todavía mostraba "Step 1 of 8"; se quitó en el build 25.

## Build 23 — 2026-10-18
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build23  ·  **Commit de la app:** `ba4cc251`
- **Qué cambió:**
  - "Audio primero": la voz se saca en el iPhone y se sube primero, el corte empieza con ella y los videos suben después.
  - "iCloud liviano": un video que solo está en iCloud empieza con una copia chica y rápida; el original baja y sube en segundo plano.
  - Máximo 3 partes a la vez (antes se ponían todas y las que esperaban vencían).
  - Diagnóstico con el tiempo de cada paso del servidor.
- **Pantallas nuevas o cambiadas:** Procesando (textos de progreso).
- **Pendiente o con problemas:** "Audio primero" solo se usa si el servidor lo permite; ese cambio del servidor no estaba publicado (rama feat/audio-first).

## Build 22 — 2026-10-18
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-build22  ·  **Commit de la app:** `1c9dc50b`
- **Qué cambió:**
  - Subidas que terminan: con el build 21 la subida se completaba pero el corte nunca empezaba porque iOS dormía la app; ahora termina al reabrir o en segundo plano.
  - "Try again" ya no vuelve a subir todo desde 0.
  - En Combine, cada clip es su propia subida; un clip que falla ya no para a los demás.
  - Se copia el archivo original de Fotos; los códigos reales quedan en Diagnóstico.
- **Pantallas nuevas o cambiadas:** ninguna (cambios internos de la subida).
- **Pendiente o con problemas:** copiar el original hacía que un video editado en Fotos (con canción, recorte o filtro) se subiera sin esos cambios; se arregló en el build 41.

## Build 21 — 2026-10-16
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `8e67e7ec`
- **Qué cambió:**
  - Subidas que terminan aunque vayan lentas (~1 MB/s) y con CutSell en segundo plano.
  - Cada parte se reintenta sola hasta 3 veces; si pasan 90 s sin avanzar sale un aviso con "Try again", sin cancelar nada.
  - Preparación: videos de menos de 4K se suben tal cual; los otros se achican solo si quedan más livianos.
- **Pantallas nuevas o cambiadas:** Procesando (aviso de subida detenida).
- **Pendiente o con problemas:**
  - Con este build la subida se completaba pero el corte no empezaba si iOS dormía la app; se arregló en el build 22.
  - Había un cambio del servidor preparado y no publicado (necesitaba permiso).

## Build 20 — 2026-10-16
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `8ed95b4c`
- **Qué cambió:**
  - Las partes del video se suben por una conexión normal con progreso real (el servicio de subida en segundo plano de iOS fallaba cada parte).
- **Pantallas nuevas o cambiadas:** ninguna
- **Pendiente o con problemas:** sin datos

## Build 19 — 2026-10-16
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `8cd2f22c`
- **Qué cambió:**
  - Un corte nuevo ya no se queda en "Uploading 0 %": el aviso de IA quedaba escondido debajo de la pantalla de subida; ahora sale arriba y antes.
  - Porcentaje real de iCloud, preparación y subida; 3 intentos por pedido y por parte; aviso con el error real y "Try again".
  - Después del cambio de almacenamiento a R2, ninguna parte salía del iPhone (nombres de archivo demasiado largos); arreglado.
  - Diagnóstico de cada paso; el sonido de la app se mezcla con otras apps.
- **Pantallas nuevas o cambiadas:** Aviso de IA, pantalla de subida / Procesando.
- **Pendiente o con problemas:**
  - El servicio de subida en segundo plano fallaba cada parte; se cambió en el build 20.
  - En el iPhone de la Product Owner, con subida lenta (~1 MB/s) la app marcaba falla y cancelaba partes que seguían subiendo; y una grabación de pantalla se agrandaba al prepararla (72 MB → 103 MB). Se arregló en el build 21.

## Build 18 — 2026-10-16
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `04cfff45`
- **Qué cambió:**
  - Arreglos de lo que falló en un iPhone real con el build 17:
  - Los sonidos ya suenan en la vista previa del iPhone.
  - "Whole video" cambia solo el sonido del video (antes silenciaba todo).
  - Pellizcar, girar y arrastrar para los keyframes ya funciona en el iPhone; arrastrar gana sobre tocar en la línea de tiempo.
  - "Photo" abre las fotos; se usa el selector de fotos del sistema (la galería anterior leía máximo 300).
  - Un overlay recién subido sigue seleccionado; la fila elegida siempre se ve.
  - Diagnóstico escondido con "Send report".
- **Pantallas nuevas o cambiadas:** Editor (línea de tiempo, keyframes, overlays, volumen), Galería, You (Diagnóstico).
- **Pendiente o con problemas:** después del cambio a R2 ninguna parte de video salía del iPhone; se arregló en el build 19.

## Build 17 — 2026-10-16
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `64e107b4`
- **Qué cambió:**
  - Sonidos de CutSell: se eligen, mueven, cortan, recortan y borran; nunca se agregan mudos.
  - Panel de Volumen por clip o para todo el video, con Silenciar e íconos en la línea de tiempo.
  - "Clean up voice" (limpiar voz) Sí / No con porcentaje.
  - Keyframes tipo CapCut: pellizcar, arrastrar o girar el clip, ◇ para agregar o borrar; Auto Zoom AI.
  - El aviso de IA y la Política de privacidad nombran los servicios de IA usados y el almacenamiento.
- **Pantallas nuevas o cambiadas:** Editor (Sonidos, Volumen, Clean up voice, keyframes), Aviso de IA, Política de privacidad.
- **Pendiente o con problemas:**
  - En un iPhone real los sonidos no se oían en la vista previa, los gestos de keyframes no hacían nada, "Whole video" silenciaba todo y la galería mostraba máximo 300 fotos. Se arregló en el build 18.
  - Faltaba la revisión de ver y escuchar en un iPhone real por la Product Owner.

## Build 16 — 2026-10-15
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `30e8f9c3`
- **Qué cambió:**
  - Editor v5: barra con Audio, Volumen y Velocidad; paneles arriba de la barra.
  - Overlays con capas, diseños, borde y giro.
  - Voice overs y sonidos de CutSell como clips (cortar, recortar, mover, volumen, bajar el sonido del video, limpiar voz).
  - Velocidad por clip; voice over en 2 pasos con cuenta regresiva; música desde Archivos.
  - Aviso de IA, Política de privacidad y Términos dentro de la app.
- **Pantallas nuevas o cambiadas:** Editor (Audio, Volumen, Velocidad, Overlay, Voice over), Aviso de IA, Política de privacidad, Términos, You.
- **Pendiente o con problemas:** el cambio del servidor para el Editor v5 esperaba aprobación para publicarse (según el documento del Editor v5).

## Build 15 — 2026-10-14
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `8040c98b`
- **Qué cambió:**
  - Arreglos de lo reportado en el build 14:
  - Un solo encabezado en el editor (antes salía doble).
  - Overlay abre directo las fotos y videos del teléfono; se ve al momento, con % de subida y reintento.
  - B-roll: el "+" de la fila de video abre la galería y lo pone en la pista principal.
  - Mantener presionado y arrastrar clips, voces y overlays, con imanes y vibración; un solo deshacer.
  - Las miniaturas ya no desaparecen.
  - Exportar empieza al tocar, espera lo pendiente, muestra el % real y avisa cuando está listo; si nada cambió, reusa la última exportación.
  - Anillo de Procesando nuevo y nombre del corte en vez del nombre del archivo.
- **Pantallas nuevas o cambiadas:** Editor, Procesando, Export.
- **Pendiente o con problemas:** había cambios del servidor esperando publicarse; uno de ellos: un B-roll mudo o quieto hacía fallar la exportación.

## Build 14 — 2026-10-14
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/figma-part1  ·  **Commit de la app:** `b517efe3`
- **Qué cambió:**
  - Recortar clips por los bordes con manijas blancas (hacia afuera recupera material, hacia adentro acorta); los subtítulos siguen a las palabras.
  - Voice over: aparece al instante al guardar, se sube en segundo plano y se reintenta; una voz nueva va justo después de la que está en el cursor.
  - Las etiquetas "Voiceover N" siempre se leen.
  - Panel de sonido del video debajo de la línea de tiempo.
- **Pantallas nuevas o cambiadas:** Editor (línea de tiempo, Voice over).
- **Pendiente o con problemas:** reportes de este build (encabezado doble, overlay, B-roll, arrastrar, miniaturas que desaparecen, exportar) se arreglaron en el build 15.

## Build 13 — 2026-10-13
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `993ebbb2`
- **Qué cambió:**
  - Editor más rápido: el video se guarda en el teléfono, abre al momento, la primera imagen sale enseguida.
  - El reproductor se detiene al final del corte.
  - Una sola forma de guardar cada cambio; si hay conflicto reintenta solo y muestra un error sencillo solo tras 2 fallas.
  - Los cortes en Edits quedan precargados para abrir al instante.
- **Pantallas nuevas o cambiadas:** Editor, Edits.
- **Pendiente o con problemas:** sin datos

## Build 12 — 2026-10-13
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `9b3aae67`
- **Qué cambió:**
  - Rediseño del Editor v4: barra y controles con contorno, menú de calidad, línea de tiempo con íconos por fila.
  - Auto Zoom con keyframes ◇; hoja de voice over combinada; Style = solo Auto Zoom.
  - "Choose your style" manda Overlap y Auto Zoom.
  - Espejo: interruptor en la cámara (apagado al inicio, se recuerda) y espejo por clip en el editor.
  - El menú de subtítulos reemplaza los controles, como el Figma.
- **Pantallas nuevas o cambiadas:** Editor, Cámara, Choose your style.
- **Pendiente o con problemas:** sin datos

## Build 11 — 2026-10-13
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `d3a7ee26`
- **Qué cambió:**
  - Editor v4 del Figma (aprobado el 13 de octubre): encabezado con el nombre del corte, video más grande, cursor fijo en el medio, zoom con dos dedos.
  - Pista de video con imágenes reales y subtítulos adentro; pista de voice over siempre visible; pista de overlay.
  - Barra: Split, Delete, Mirror, Rotate | Overlay, Voice Over, Captions, Style.
  - Voice over con onda en vivo y "Enhance voice".
  - Arreglos: ya hay sonido al reproducir (el interruptor de silencio lo apagaba); la vista previa ya no queda negra.
- **Pantallas nuevas o cambiadas:** Editor.
- **Pendiente o con problemas:** el panel Style decía que todavía no se aplicaba; en el build 12 pasó a ser solo Auto Zoom.

## Build 10 — 2026-10-13
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `92d0e94c`
- **Qué cambió:**
  - El B-roll se ve como bloques con nombre en la pista de video; se puede tocar, dividir y borrar.
- **Pantallas nuevas o cambiadas:** Editor.
- **Pendiente o con problemas:** sin datos

## Build 9 — 2026-10-13
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `b3a4572c`
- **Qué cambió:**
  - Editor v2 del Figma reemplaza el editor viejo en lista: barra de arriba, video con borde de marca, pistas Video / Voiceover / Overlay, barra Split · Delete · Overlay · Voice Over · Captions.
  - Nombres de corte legibles y renombrables ("Cut 13 Oct") en vez de nombres de archivo.
  - Arreglos de diseño (fila de tiempo, etiquetas en una línea, avisos a todo lo ancho).
- **Pantallas nuevas o cambiadas:** Editor.
- **Pendiente o con problemas:** sin datos

## Build 8 — 2026-10-12
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `2356d0c2`
- **Qué cambió:**
  - Arreglos tras la revisión de la Product Owner en su iPhone: los botones del editor se salían de la pantalla (ahora la barra se desliza); una sola barra de tiempo; sin Deshacer / Rehacer flotantes.
  - Los clips muestran una imagen real del video.
  - Se usa el nombre del video en vez de "Untitled Cut".
  - Edits y Recent cuts siguen los cortes hasta que están listos (punto verde) y muestran miniatura real.
  - Home Hub: contador de cortes del mes / 30.
- **Pantallas nuevas o cambiadas:** Editor, Edits, inicio de Create (Home Hub).
- **Pendiente o con problemas:** el límite de 30 por mes solo se muestra; no se aplica.

## Build 7 — 2026-10-12
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `83d0c51e`
- **Qué cambió:**
  - El editor por fin carga el borrador (en los builds 5 y 6 la carga se cortaba y se quedaba en "Loading timeline…").
  - Ready: "YOU JUST SAVED" y el tiempo total de edición ahorrado por la cuenta.
- **Pantallas nuevas o cambiadas:** Editor, Ready.
- **Pendiente o con problemas:** sin datos

## Build 6 — 2026-10-12
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `528e70d2`
- **Qué cambió:**
  - Create abre en el Home Hub: subir videos (Single / Bulk / Combine), grabar con CutSell, proyecto Voice Over (próximamente), cortes recientes.
  - Procesando con el progreso real del servidor; "Continue in background" y aviso cuando el corte está listo.
  - Pantalla Ready: primera imagen, metraje, clips usados, "Review & edit" y "Export video".
  - Arreglos de revisión: mientras sube no se puede salir; se limpian errores viejos.
- **Pantallas nuevas o cambiadas:** inicio de Create (Home Hub), Procesando, Ready, Edits.
- **Pendiente o con problemas:**
  - El editor no cargaba el borrador; se arregló en el build 7.
  - "Voice Over project" quedó como próximamente; el aviso de listo sale dentro de CutSell (todavía no había avisos del servidor).

## Build 5 — 2026-10-12
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `f113e707`
- **Qué cambió:**
  - Arreglo: en los builds 3 y 4 ningún video se subía desde el iPhone (el almacenamiento rechazaba la subida), así que no había corte ni nada que abrir.
  - Un proyecto cuya subida no terminó dice "This video didn't finish uploading" con "Resume upload".
- **Pantallas nuevas o cambiadas:** Edits (proyecto sin terminar de subir).
- **Pendiente o con problemas:** el editor no cargaba el borrador; se arregló en el build 7.

## Build 4 — 2026-10-12
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `f2bcf167`
- **Qué cambió:**
  - Diseño Figma parte 1: colores, letra Inter y componentes.
  - Pestañas Create · Edits · Plan · You; la app abre en Create (cámara).
  - Cámara, galería Single / Bulk / Combine y hoja de estilo.
  - Pantallas de permisos y de problemas; cuadrícula de Edits con filtros All / Ready / Drafts.
  - You con el botón real de Sign in with Apple.
  - Arreglos de la revisión independiente (no decir "guardado" si no se guardó, temporizador de grabar, permisos).
- **Pantallas nuevas o cambiadas:** barra de pestañas, Cámara, Galería, Choose your style, permisos, problemas, Edits, You.
- **Pendiente o con problemas:**
  - Ningún video se subía desde el iPhone; se arregló en el build 5.
  - Los límites de 10 minutos / 1 GB del Figma esperan a la Product Owner.

## Build 3 — 2026-10-12
- **Resultado:** subido a TestFlight
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `3e3c6814`  ·  **Commit del pedido a TestFlight:** `9572ff80`
- **Qué cambió:**
  - Primer build que llegó a TestFlight. Incluye todo lo hecho antes:
  - Subtítulos v2: se ven sobre el video al editar; panel con Classic / Highlight / Box / Yellow, 9 letras y colores; arrastrar para mover y pellizcar para cambiar el tamaño.
  - Sign in with Apple (guarda los proyectos de invitado), cerrar sesión, borrar cuenta y pantalla de bienvenida.
  - Voice over grabado con el micrófono.
  - Textos de permisos (micrófono, fotos, cámara) en inglés y español.
  - Proceso de subida manual a TestFlight (solo pruebas internas).
- **Pantallas nuevas o cambiadas:** Editor (panel de subtítulos), Voice over, menú de perfil, Bienvenida.
- **Pendiente o con problemas:** ningún video se subía desde el iPhone; se arregló en el build 5.

## Build 2 — 2026-10-12
- **Resultado:** falló (no llegó a TestFlight)
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `d90bfc50`  ·  **Commit del pedido a TestFlight:** `3e3c6814`
- **Qué cambió:**
  - Segundo intento con el mismo código del build 1, con un cambio en la firma del proceso de build.
- **Pantallas nuevas o cambiadas:** ninguna (no llegó a TestFlight).
- **Pendiente o con problemas:** la subida falló; el build 3 (un minuto después) sí llegó, con otro ajuste de la firma.

## Build 1 — 2026-10-12
- **Resultado:** falló (no llegó a TestFlight)
- **Rama (según el historial de git):** feature/ios-captions-v2  ·  **Commit de la app:** `d90bfc50`
- **Qué cambió:**
  - Primer intento del proceso de subida a TestFlight (solo cambios del proceso de build y textos de permisos).
- **Pantallas nuevas o cambiadas:** ninguna (no llegó a TestFlight).
- **Pendiente o con problemas:** la subida falló; también falló el build 2 y el primero que llegó fue el build 3.
