# CutSell.ai — estado al 8 de octubre de 2026

Pega o adjunta este archivo al iniciar cualquier sesión nueva. Reemplaza al del 7 de octubre.
También está guardado en GitHub, en la rama `feat/ios-captions-v2`: `docs/estado/CutSell_estado_2026-10-08.md`.

## Qué está encendido hoy (esta sesión no lo cambió)
- **Servidor (Render, `cutsell-api`, Starter):** en vivo en la rama `feat/simple-engine-v1`, commit `6dbb7147`. https://cutsell-api.onrender.com. Tiene las dos variables del aviso (`CUTSELL_WORKER_WAKE_URL`, `CUTSELL_WORKER_WAKE_TOKEN`). Auto-deploy apagado a propósito.
- **Trabajador (Modal, app `cutsell-worker`, cuenta `automatedretailservices`):** sin GPU, duerme y despierta con el aviso; revisión de respaldo cada 5 minutos. Secreto `cutsell-worker` con 8 valores (las llaves de Amazon son las mismas de Render; las del archivo `aws_clave` NO sirven para este almacenamiento). Aviso: https://automatedretailservices--cutsell-worker-wake.modal.run
- **Motor:** el nuevo (v1.0, `CUTSELL_ENGINE=simple`) con entrega directa (`technical_qc`).
- `main` sin tocar. Nada en TestFlight.

## Automatizar los despliegues
- **GitHub:** las sesiones abiertas con el repositorio `AutomatedRetailServices/EditDNA-worker` ya pueden escribir (comprobado el 7-oct). Claude sube los cambios; Swanny no pega comandos.
- **Modal:** el robot (`.github/workflows/cutsell-modal-worker-deploy.yml`, commit `2746c47` en `feat/simple-engine-v1`) está subido y corrió bien el 7-oct a las 8:16 pm: corre las pruebas y hace `modal deploy` cuando cambia el trabajador en esa rama. Los dos secretos de Modal ya están en GitHub.
- **Render:** Claude lo actualiza con su conexión de Render (servicio `srv-d9rdlsvavr4c73912uo0`). Siempre con permiso de Swanny.

## Ramas
- `feat/simple-engine-v1` (`2746c47`): servidor y trabajador. Lo que está en vivo.
- `feat/ios-testflight-prep` (`0e1be0a`): puesta al día con `feat/simple-engine-v1` el 7-oct. Sin choques.
- `feat/ios-captions-v2` (sale de `feat/ios-testflight-prep`): **la pantalla de Captions nueva de la app.** Commits:
  - `248d758` entrega 1: subtítulos sobre el video mientras editas
  - `51a35b3` entrega 2: panel de Captions de Figma debajo del video
  - `3163118` entrega 3: mover y agrandar con los dedos
  - `4b521de` arreglo de las 5 pruebas viejas de "partir clip" (solo pruebas)
  - (este archivo de estado)
- Todavía NO se unió `feat/ios-captions-v2` a `feat/ios-testflight-prep`. Hay que hacerlo antes de compilar para TestFlight (con permiso de Swanny).

## App iPhone: Captions del Editor v2 (hecho el 7 y 8 de octubre)
Las tres entregas compilan: la revisión de iPhone en GitHub (`cutsell-ios-ci.yml`) salió en verde en cada una. La app se construye, tiene las 9 letras dentro y arranca en el simulador.
1. **Subtítulos sobre el video mientras editas.** Máximo 3 palabras, con las mismas reglas del servidor (código copiado regla por regla). Los cuatro estilos, la palabra pintada de Highlight, las 9 letras (son los mismos archivos del servidor, `cutsell_worker/fonts`) y la posición y el tamaño guardados. Nunca se sale del cuadro. Cambiar un ajuste ya no reinicia el video.
2. **Panel de Figma debajo del video** (filas "2 · Captions" y "2b · Captions"). On/Off, Done, Classic / Highlight / Box / Yellow, fila de letras deslizable, 3 colores de palabra (verde, rojo, azul), caja negra o blanca. Se ve al instante y se guarda en el servidor, un cambio detrás de otro. Tocar el subtítulo del video abre "Fix words" para corregir las palabras de ese clip.
3. **Mover y agrandar con los dedos.** Arrastrar y pellizcar el subtítulo con el panel abierto. Una sola posición y tamaño para todo el video, como CapCut. Se usa la regla del servidor en cada movimiento, así que nunca se sale del cuadro ni salta al soltar.

Cosas que funcionan así a propósito:
- Si se corrigen palabras a mano, ese clip pasa a un solo subtítulo pequeño para todo el clip y no se puede mover. Así lo exporta hoy el servidor.
- Mientras el panel de Captions está abierto, la página no se desplaza (para que el dedo mueva el subtítulo). "Done" lo cierra.
- La revisión de iPhone ahora falla si faltan las letras dentro de la app.

## Pruebas
- Las 169 pruebas de la app de iPhone (`tests/test_cutsell_ios_*.py`) pasan en `feat/ios-captions-v2`.
- Las 5 que fallaban desde el arreglo de "partir clip" de TestFlight ya buscan la línea nueva (solo se cambiaron las pruebas, no la app).
- Pruebas nuevas en `tests/test_cutsell_ios_captions_v2_preview.py`: comprueban que la app usa los mismos números que el servidor (3 palabras, tamaños de letra, posición, colores).

## Para revisar en TestFlight
1. En un iPhone pequeño, que el panel de Captions no quede tapado abajo.
2. Que arrastrar y pellizcar el subtítulo se sienta bien con los dedos (sobre todo pellizcar un subtítulo pequeño).
3. Que los subtítulos se vean igual en la app que en el video exportado (frases, momento, letra, tamaño y lugar).

## Lo que falta, en orden
1. **Apple:** esperar el correo de activación (pagado, documentos en revisión). Luego: Team ID → compilar y firmar en Xcode → TestFlight.
2. Unir `feat/ios-captions-v2` a `feat/ios-testflight-prep` (con permiso).
3. **App iPhone, resto del Editor v2:** el resto del editor sigue con el aspecto viejo. Falta duplicar, rotar, keyframes, grabar voz y estirar para recuperar. El voice over actual no llega al video final.
4. Antes de unir a `main`: quitar los disparadores temporales de la revisión de iOS en `.github/workflows/cutsell-ios-ci.yml` (ahora son dos ramas: `feat/ios-testflight-prep` y `feat/ios-captions-v2`).
5. Pendientes de producto: export siempre 1080×1920 a 30 cuadros; cuentas anónimas (falta Sign in with Apple); Android.
6. Motor: frase de toma anterior antes del gancho (V01, V05), tomas repetidas (video 07, V03), paso inestable (Y08).

## Subtítulos en el servidor (sin cambios)
- Tercio de abajo, máximo 3 palabras. Estilos (`caption_preset`): classic, yellow, highlight_green / highlight_red / highlight_blue, box (caja negra), box_light (caja blanca).
- Letras (`caption_font`, por defecto montserrat): montserrat, poppins, roboto, oswald, anton, luckiest_guy, bebas_neue, inter, bangers.
- Posición y tamaño para todo el video (`caption_x`, `caption_y` de 0 a 1; `caption_scale` de 0.5 a 2).
- Diferencias aceptadas con el primer Figma: no es la letra de Apple, lleva borde oscuro fino, la caja tiene esquinas rectas.

## Figma (archivo `PZnh8DuFcnhMUqPScHHL5q`, página "04_Editor")
Editor v2 tiene 19 pantallas; las de Captions son "Captions · On", "Captions · Off" y la fila "2b · Captions". Las 47 originales no se tocaron.

## Reglas de Swanny
Explicar simple, en español, sin tecnicismos y un paso a la vez. Preguntar antes de subir cada cambio. No gastar dinero nuevo, no unir a `main` ni desplegar sin su permiso. No inventar precios ni planes. No tocar sus 47 pantallas originales de Figma. Nunca escribir llaves en mensajes ni archivos compartidos; las escribe ella en Modal y Render.
