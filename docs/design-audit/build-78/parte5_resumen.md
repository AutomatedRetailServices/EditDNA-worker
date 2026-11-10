# Assessment de diseño · Build 78 · Parte 5 · Resumen

- **Build revisado:** 78 (TestFlight, 9 nov 2026)
- **Rama:** `feature/clon-ar-autopausa` · **Commit de la app:** `71ae403`
- **Cómo se hizo:** solo lectura del código real (vistas SwiftUI en `mobile/ios/CutSell/`) comparado con Figma
  `rXfUuJQgwZIZJTuONvU1Z7` ("CutSell.ai — App White Indigo"), páginas 00–08 y 09 (propuesta Split).
  No se cambió código ni Figma.
- **Detalle por pantalla** (textos exactos con su español, colores hex, fuentes, tamaños, íconos y estados):
  - [Parte 1 · Record y Upload](parte1_record_upload.md)
  - [Parte 2 · Plan](parte2_plan.md)
  - [Parte 3 · Edits, Editor, Export](parte3_edits_editor_export.md)
  - [Parte 4 · Inicio de la app, Profile, errores y avisos](parte4_profile_auth_avisos.md)

## Totales

Varias pantallas salen en más de una parte (barra de abajo, permisos, avisos de "listo", errores). Sin repetirlas:

| Estado | Pantallas |
|---|---|
| ✅ igual a Figma | 5 |
| 🟠 existe en Figma pero en la app está distinta | ~36 |
| 🔴 no existe en Figma | ~90 |
| **Total** | **~131** |

Contando por parte (con repetidas): Parte 1 ✅0 🟠5 🔴31 · Parte 2 ✅0 🟠5 🔴15 · Parte 3 ✅5 🟠15 🔴29 ·
Parte 4 ✅0 🟠12 🔴33 → 150 en total.

Las 5 ✅: aviso "X is ready · View" (`1213:185`), hoja "Which product is this?", voz en off grabar (`1222:379`),
guardar (`1222:476`) y seleccionada (`1222:669`).

**Diferencia que más se repite** en casi todas las 🟠:
- la app usa botones y tarjetas **3D con sombra**; Figma los tiene **planos con borde**;
- el azul de la app es un degradado #2F7BFF → #0A55F0;
- el fondo de la app es un **degradado azul claro con brillos**; el de Figma es **blanco liso**;
- en la barra de abajo, la pestaña activa lleva **burbuja azul clara** en la app y **punto azul** en Figma.

Actualizar en Figma esos 4 componentes (botón, tarjeta, fondo, barra de abajo) resuelve gran parte de las 🟠.

## Para pasar a Figma, en orden de flujo

### 1. Inicio y Record
- 🟠 Inicio (logo animado, píldoras Record / Upload, 5 burbujas, tarjeta Stats)
- 🟠 Barra de abajo
- 🟠 Hoja "What are you promoting?" (agrupa por marca)
- 🔴 Formulario de producto nuevo
- 🔴 Aviso de borrado de cuenta
- 🔴 Alerta Faceless "Coming soon"
- 🔴 Aviso de IA "Before your first cut"
- 🔴 Permisos de cámara y micrófono
- 🔴 **Cámara antes de grabar** (Talk / B-roll, Auto-pause, Clon BETA, Script, Speed, duraciones)
- 🔴 **Cámara grabando** ("● REC" / "Waiting for voice…")
- 🔴 Aviso sin voz
- 🔴 Error de grabación
- 🔴 Hoja del guion, guion sobre la cámara y cuenta 3·2·1
- 🔴 **Clon**: inicio, grabar, vista previa, "Creating your clones…", resultado, falló
- 🔴 **"Choose your style"**
- 🟠 Processing
- 🟠 Queued
- 🟠 Processing issue
- 🟠 Ready
- 🟠 "Upload paused" / "Couldn’t start the cut"
- 🔴 Aviso de "listo" de grupo
- 🔴 Banner de arriba

### 2. Upload, Faceless y Sales Mix
- 🔴 **Galería** Single / Bulk / Combine (con el chip B-roll)
- 🔴 Acceso limitado a Fotos
- 🔴 Galería sin permiso
- 🟠 Unsupported media
- 🟠 Clip limit
- 🟠 Couldn’t add this clip
- 🔴 Supported video files
- 🔴 Hoja de Bulk "Your N videos are being edited"
- 🔴 **Sales Mix**
- 🔴 "How many videos?"
- 🔴 Hoja de producto del Mix

### 3. Plan
- 🟠 Plan
- 🟠 Needs attention
- 🟠 Week Plan
- 🔴 Week Plan con videos elegidos
- 🔴 Hojas Per day / Between / Days / Starts
- 🔴 Error del plan
- 🟠 This week
- 🟠 Acciones del video
- 🔴 Swap
- 🔴 Move
- 🔴 Menú al mantener pulsado
- 🔴 Settings
- 🟠 Día
- 🔴 "Post on TikTok now?"
- 🔴 "Edit caption"
- 🔴 Recordatorio "Time to post"

### 4. Edits y Editor
- 🟠 Edits Ready
- 🟠 Edits Processing
- 🔴 Drafts
- 🔴 Posted
- 🔴 Selección + "Add to Plan"
- 🔴 Buscador
- 🔴 Menú "•••"
- 🟠 Products y detalle de producto
- 🔴 Subida sin terminar
- 🔴 "Opening your cut…"
- 🟠 **Editor base** (barra nueva Edit · Audio · Text · Captions · Overlay · Style)
- 🔴 Opciones de Edit y de Audio
- 🔴 Rename
- 🔴 Menú de calidad
- 🔴 Pantalla completa
- 🔴 Keyframes y Curve
- 🟠 Panel de subtítulos
- 🔴 **Hoja Fuentes / Estilo** (Captions y Text)
- 🔴 Style / Auto Zoom
- 🟠 Add overlay
- 🟠 Galería del editor
- 🟠 Overlay seleccionado
- 🟠 Volume
- 🟠 Speed
- 🟠 CutSell sounds
- 🔴 Music from Files
- 🔴 Sonido seleccionado
- 🟠 Permiso de micrófono
- 🟠 Avisos con "Undo"
- 🔴 Recortar el borde de un clip

### 5. Export y publicar
- 🟠 Ajustes de exportación
- 🔴 Exportando
- 🔴 "Saved to Photos"
- 🔴 "Add to Plan to post"
- 🟠 Export failed
- 🔴 Exportar en el servidor
- 🔴 Publicar en TikTok / Compartir / Guardar
- 🔴 "Did you post it?"
- 🔴 Notificaciones

### 6. Entrar a la app y Profile
- 🔴 Splash
- 🔴 Error de arranque
- 🔴 **Bienvenida / Sign in with Apple**
- 🔴 "Signed in"
- 🔴 **Intro** (4 pantallas)
- 🟠 Profile ("THIS MONTH · N of 30", no "CUTSELL PRO")
- 🔴 Your results
- 🔴 Account & Security
- 🔴 Plan & Usage
- 🔴 Privacy & Data
- 🔴 Preferences
- 🔴 App Language
- 🔴 Storage
- 🔴 Connections
- 🔴 Privacy Policy / Terms
- 🔴 Help & Support
- 🔴 Guías y su detalle
- 🔴 Diagnostics
- 🔴 Alertas de iOS dentro de la app

## Frames de Figma que ya no existen en la app
- **Inicio:**
  - caja "Upload videos" con Single / Bulk / Combine (`1205:107`)
  - las 4 tarjetas con subtítulo (`1164:40`, `1195:115`, `1195:103`, `1164:47`)
  - logo fijo arriba a la izquierda (`1164:77`)
- **Plan:**
  - sección "In progress" (dentro de `1166:107`)
  - tarjeta "Today: post 1 CeraVe and 1 Dior video" (`1166:174`)
  - "Programar publicación" (`1259:3199`): el código existe, pero ningún botón lo abre
- **Editor:**
  - barra plana vieja Split · Delete · Mirror · Rotate · Overlay · Voice Over · Audio (`1218:2`)
  - galería propia (`1220:533`)
  - Volume: "Level" y marcas 0/100/200 (`1222:2781`)
  - CutSell sounds: "My music" y "Add Typing" (`1222:3563`)
  - Speed: "0:08→0:04" (`1222:3179`)
- **Profile:** "CUTSELL PRO · 62 of 250" (`1174:26`)
- **Errores:** "Video muy largo / muy pesado" (`1176:2888`): la app nunca la muestra porque los límites están apagados
- **Página "09 · Record Split" completa** (`3006:3`, `3006:25`, `3006:41`, `3006:60`, `3006:76`): el Split se quitó de la app

## Pantallas en diseño viejo oscuro
- **Oscuras de verdad:**
  - hoja del guion ("Your script") y sus ajustes
  - exportar en el servidor (`ExportSheet`)
  - Publicar en TikTok / Compartir / Guardar
  - un Week Plan viejo (`PlanGoalView`) que ya no se abre
- **Escritas con colores oscuros viejos y aclaradas solas, sin rediseñar** (algunas con textos que no se ven):
  - Sales Mix, "How many videos?" y la hoja de producto del Mix
  - hoja de Bulk
  - aviso de IA
  - This week, Día y Move de Plan
  - avisos del editor
- **Colores viejos sueltos:**
  - cyan #19E6FF en Mirror y Upload de la cámara y en el número de la galería
  - rojos #FF4D67, #FF2E5E y #FF8A8A en vez de #E5484D
  - letra del sistema en vez de Plus Jakarta Sans en Bienvenida, Diagnostics y los paneles del editor

## Problemas encontrados de paso (no arreglados)
1. Hoja de Bulk: el título "Ready" es blanco sobre fondo claro (no se ve).
2. "How many videos?": las opciones no elegidas tienen texto blanco (no se ven).
3. Galería: la rueda de carga es blanca sobre blanco, y el aviso de acceso limitado no tiene fondo visible.
4. Duraciones de la cámara en orden raro: "10m · 60s · 3m".
5. Cerrar "Choose your style" después de grabar borra la grabación sin preguntar.
6. En Ready, "Review & edit" y "Export video" hacen lo mismo.
7. Editor: varios avisos con texto blanco sobre fondo casi blanco (deducido del código, no visto en el teléfono).
8. Textos sin traducir al español: "Sales Mix", "Faceless", "Flash", "Split", "Use in Plan", "Tonight · …",
   "Later · …", las letras de los días (M T W T F S S) y "Your original video is safe. Nothing was lost.", entre otros.

## Regla de mantenimiento
Si un build cambia pantallas, se actualiza esta carpeta (o se crea `docs/design-audit/build-NN/` con las pantallas
cambiadas) y se anota en `docs/BITACORA.md`, en el mismo commit del build.
