# V28/V29: proveedor, recuperación 08 y defecto 09

| Versión | Run / SHA | RAW08 | RAW09 |
| --- | --- | --- | --- |
| V28 | 36443271393 / 5e0ebbb6 | Fallo de selector: respuesta sin candidates tras retry | QC PASS; 58.860/73 s Gold; 11.480 s DELETE retenidos; no demo 97–104 |
| V29 | 36444903026 / c4181717 | QC PASS; 25.199/27 s Gold; 0 s DELETE retenidos | Fallo Watch+Listen AV_TIME_RANGE 113–20 s tras retry |

Inspección de V28 en runs 36445663383, 36445891795 y 36445985170: source09 tenía una región AV mixta 68–113.5, visual «recoge una tapa, luego muestra un bote», sin probe focal pese al silencio medido alrededor de 98.435–106.764. V28 descartó la frase anterior 73.78–77.07 mientras retuvo 77.07–90.59, con índices de orden invertidos 9 vs 5; la comprobación focal no se abrió por ese conflicto. V29 restauró 08 por política de reintentos basada en reclamos materiales de la misma fuente. V30 propone dividir AV largo en ventanas locales, nominar regiones mixed sólo para investigar y verificar continuidad fuente antes de reconciliar orden. Se deben medir 08 y 09, costo, QC y preview, luego 01–10 si estable. Ni V28 ni V29 autorizan entrega comercial.
