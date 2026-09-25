# MOV — prueba Medium + WhisperX + Watch & Listen

Run 36105019864. SHA probado 762f44bd1f8a348556f3a886c216450638b00c02.
Archivo original: 68,730,266 bytes, 97.433333 s; SHA256 5ae3cffb9034cc9aebbe4f645c4f1f34538f9113f05990291215c86b7163b681.

## Resultado real

WhisperX 3.8.6 ejecutado en L4, inglés detectado, 252 palabras en 18 segmentos. Validación estructural de alineación passed, sin interpolación ni fallback. Esto no prueba exactitud perceptual de cada palabra; por ejemplo, hay un score CTC 0.0 para «now» en el primer segmento.
El motor creó una selección de cinco segmentos (20.418 s previstos) pero NO produjo video. Freeze bloqueado antes de pacing/boundaries/render: freeze_blocked_no_render. Cero intentos de render. El workflow success y engine_ok=true significan ejecución concluida, no éxito editorial.

## Dos fallos editoriales observados

1. Conserva fragmentos de preparación: «now how am i supposed to say», «pull it yeah there you go i just needed a pep talk», «strap other side so let's flip it not so good» y «you can get a flip it i'm». También conserva la toma final completa de producto (84.44–96.756 s). El clasificador marcó dos de esos fragmentos como failed y recomendó eliminar, pero applied_delete quedó false; no había reemplazo aceptado.
2. El validador bloquea por tres UNIQUE_FACT_LOST / REAL_CONTENT_LOSS después de eliminar intentos/preparación. Los textos incluyen «oh too many people ready set these are the», «already gonna mess it up…» y «I've moved my finger again». En los tres consta omission_permit_denied_reason=missing_identity y no llega clasificación/propiedad del contenido al control final. Esto identifica una inconsistencia de integración entre limpieza y preservación, no una causa de fallo de WhisperX.

## Capas

Contexto global preparado para clasificador: seis de seis regiones. BestTake V2 y protecciones evaluaron dos casos, sin cambios de ganador. Prosodia procesó dos candidatos de una familia; evidencia insuficiente para el árbitro. No se aplicó bypass al bloqueo ni se confeccionó una edición manual como si fuera salida del motor.

## Límite y siguiente trabajo

No hay MP4 de esta prueba para comparar visualmente. La prueba anterior del MOV precedía la integración automática: no es un A/B aislado de WhisperX. Corresponde rastrear la evidencia de error/BTS y su identidad desde la clasificación hasta la eliminación y el validador, con reglas generales y reproducción offline de este resultado, antes de otro render pagado. No se modificó la lógica del motor en este turno ni se desplegó producción.
