"""Self-contained Spanish report; embeds the exact JSON evidence, never keys."""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
import re
import zipfile

from compare_gpt_whisperx_stability import build_comparison

LABELS = {
    "abandoned_stomach_attempt_absent": "Elimina el intento fallido del estómago",
    "full_gynecologist_take_selected": "Conserva la toma completa de la ginecóloga",
    "percentage_restatement_absent": "Elimina la repetición del porcentaje",
    "pimples_bad_monolith_absent": "Elimina la toma larga fallida de las espinillas",
    "pimples_bad_source_absent": "Evita el tramo fallido de las espinillas",
    "pimples_later_source_selected": "Conserva la toma posterior de las espinillas",
    "sonography_opening_complete": "Conserva completa la entrada de la sonografía",
    "same_take_negation_with_sentence": "Conserva la negación en su frase",
    "closing_exhortation_not_reopened": "No vuelve a iniciar el cierre",
    "closing_instruction_preserved": "Conserva el consejo final",
    "closing_care_exhortation_exactly_once": "Dice «cuídate» una sola vez",
}
esc = html.escape


def clock(value):
    if value is None:
        return "—"
    return f"{int(value) // 60}:{value % 60:04.1f}"


def cell(value):
    return "<td>" + esc(str(value)) + "</td>"


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(cell(x) for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def render(artifacts: Path, output: Path):
    data, results = build_comparison(artifacts)
    a, trials = data["aggregate"], data["trials"]
    available = [s for s in trials if s.get("transcript_sha256")]
    batch = json.loads((artifacts / "batch.json").read_text()) if (artifacts / "batch.json").exists() else {}
    evidence = io.BytesIO()
    with zipfile.ZipFile(evidence, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(artifacts.rglob("*.json")):
            raw = path.read_bytes()
            if re.search(rb'(?:sk-(?:proj-|admin-)?[A-Za-z0-9_\-]{28,}|AIza[0-9A-Za-z_\-]{30,}|AKIA[A-Z0-9]{16})', raw):
                raise RuntimeError("Credential-like content found; evidence export blocked")
            archive.writestr(path.relative_to(artifacts).as_posix(), raw)
        archive.writestr("comparison.json", json.dumps(data, ensure_ascii=False, indent=2))
    summary_rows = [[s["trial"], clock(s.get("output_duration_sec")), s.get("selected_count", "—"),
                     s.get("word_count", "—"), "Pasa" if s.get("technical_qc") == "PASS" else ("Bloqueado" if s.get("technical_qc") else "Sin dato"),
                     f'{s["editorial_passed"]}/11' if "editorial_passed" in s else "Sin resultado",
                     clock(s.get("engine_elapsed_sec"))] for s in trials]
    failure_rows = [[LABELS.get(check, check),
                     *["Falla" if any(f["id"] == check for f in s.get("editorial_failed", [])) else
                       ("Pasa" if s.get("transcript_sha256") else "Sin dato") for s in trials]]
                    for check in LABELS]
    pair_rows = [[f'{p["trial_a"]} ↔ {p["trial_b"]}', p["lexical_token_edit_distance"],
                  f'{p["lexical_difference_percent"]:.2f} %',
                  f'{p["selected_source_coverage_iou_percent"]:.2f} %',
                  f'{1000*p["matching_boundary_median_drift_sec"]:.1f} ms',
                  f'{1000*p["matching_boundary_p95_drift_sec"]:.1f} ms'] for p in data["pairwise"]]
    verdict = ("Mantener como opción experimental: la edición todavía falla criterios de aceptación."
               if a["editorial_passes"] < 5 else
               "Las cinco pruebas pasan los criterios editoriales automáticos; queda la revisión humana.")
    body = f'''<div class="eyebrow">CUTSELL · VIDEO00 · CINCO PRUEBAS COMPLETAS</div>
<h1>GPT Transcribe + WhisperX</h1>
<p class="lead">{a["results_available"]} resultados del motor completo, con el mismo original y la misma versión.</p>
<div class="verdict"><strong>{esc(verdict)}</strong></div>
<div class="cards"><div><b>{a["technical_passes"]}/5</b><span>Render técnico aprobado</span></div>
<div><b>{a["editorial_passes"]}/5</b><span>Edición aprobada por todos los criterios</span></div>
<div><b>{a["unique_selections"]}</b><span>Selecciones distintas, por texto y tiempos</span></div></div>
<h2>Resultado de cada prueba</h2>'''
    body += table(["Prueba", "Video", "Clips", "Palabras", "Control técnico", "Criterios editoriales", "Tiempo del motor"], summary_rows)
    body += '''<p class="note">El tiempo del motor no incluye toda la preparación, limpieza posterior y transferencia. Los cinco MP4 originales se entregan por separado. Aprobar el render no significa que todas las decisiones editoriales sean correctas.</p><h2>¿Qué se repite y qué cambia?</h2>'''
    body += f'''<p>Se observaron <strong>{a["unique_exact_transcripts"]} transcripciones de texto distintas</strong> ({a["unique_lexical_transcripts"]} al ignorar mayúsculas y puntuación), {a["unique_timed_transcripts"]} conjuntos distintos de palabras con tiempos y {a["unique_selection_texts"]} selecciones distintas de texto. La variación no demuestra por sí sola cuál versión es correcta.</p>'''
    body += table(["Criterio editorial", "Prueba 1", "Prueba 2", "Prueba 3", "Prueba 4", "Prueba 5"], failure_rows)
    body += '<h2>Consistencia entre cada par de ejecuciones</h2>'
    body += table(["Pruebas", "Cambios de palabras", "Diferencia de texto", "Solapamiento del original elegido", "Desvío temporal mediano", "Desvío temporal P95"], pair_rows)
    body += '''<p class="note">Diferencia de texto: distancia mínima de inserciones, borrados y sustituciones de tokens, dividida entre la longitud mayor. No es una tasa de error contra una transcripción humana. Solapamiento: intersección/unión del tiempo de original seleccionado; 100 % significa que se eligen los mismos intervalos. Los desvíos temporales se calculan solo en palabras que coinciden por secuencia.</p><h2>Alineación y reutilización dentro de cada prueba</h2>'''
    body += table(["Prueba", "Duración cero", "Tiempos inválidos", "Palabras >1.5 s", "Palabras <25 ms", "Peticiones GPT registradas", "Reutilizaciones internas"],
                  [[s["trial"], s["zero_duration_words"], s["invalid_word_count"], len(s["long_words_over_1_5s"]),
                    s["short_words_under_25ms"], s["recorded_gpt_requests"], s["cache_hit_count"]] for s in available])
    body += f'''<p>La evidencia contiene {a["recorded_gpt_requests"]} peticiones GPT y {a["distinct_request_ids"]} identificadores de petición distintos; el motor reutilizó el resultado dentro del mismo trabajo {a["total_cache_hits"]} veces. No se compartieron transcripciones entre pruebas. No se consultó la facturación, por lo que estos conteos no equivalen a un precio en dólares.</p>
<p class="note">WhisperX aporta alineación, no una segunda transcripción. Su confianza CTC no es confianza léxica de GPT. Las duraciones extremas son alertas de revisión, no errores confirmados. La limpieza posterior conserva el Medium existente; esta tanda cambia el ASR principal.</p>'''
    for s in available:
        index = s["trial"]
        result = results[index]
        body += f'<details><summary>Prueba {index}: selección, texto completo y alertas</summary>'
        body += f'<p>Congelación del plan: <strong>{esc(str(s["freeze_status"]))}</strong>. Coincide con el plan revisado: {s["freeze_matches_reviewed_plan"]}. Revisión perceptual: {esc(str(s["perceptual_status"]))}; {s["perceptual_findings"]} hallazgos.</p>'
        contact = artifacts.parent / "five-trial-review" / f"trial{index}-contact.jpg"
        if contact.exists():
            body += '<img style="max-width:100%;height:auto" alt="Fotogramas de muestra de la prueba ' + str(index) + '" src="data:image/jpeg;base64,' + base64.b64encode(contact.read_bytes()).decode() + '"><p class="note">Muestras visuales extraídas del MP4 de esta prueba; no representan una revisión humana completa.</p>'
        body += table(["#", "Tiempo en el original", "Texto seleccionado"],
                      [[i, f'{clock(c["start"])}–{clock(c["end"])}', c["text"]] for i, c in enumerate(result["selected"], 1)])
        body += '<h3>Transcripción completa alineada</h3>'
        body += ''.join(f'<p><span class="time">{clock(c["start"])}–{clock(c["end"])}</span> {esc(c["text"])}</p>'
                        for c in result["timed_asr_replay_evidence"]["raw_segments"])
        body += '<h3>Palabras con duración superior a 1.5 segundos</h3>'
        body += table(["Palabra", "Inicio original", "Fin original", "Duración"],
                      [[w["text"], clock(w["start"]), clock(w["end"]), w["duration"]] for w in s["long_words_over_1_5s"]])
        body += f'<p class="note">SHA-256 del resultado original: <code>{s["raw_result_sha256"]}</code></p></details>'
    body += '<details><summary>Diferencias de transcripción entre pruebas</summary>'
    for pair in data["pairwise"]:
        body += f'<h3>Prueba {pair["trial_a"]} frente a prueba {pair["trial_b"]}</h3>'
        body += table(["Tiempo original A", "Texto A", "Texto B"],
                      [[clock(d["source_time_a"]), d["trial_a_text"] or "∅", d["trial_b_text"] or "∅"] for d in pair["text_differences"]])
    body += '</details><details><summary>Identidad de ejecución y evidencia descargable</summary>'
    if any(s.get("harness_error_type") for s in trials):
        body += '<p class="note">El ejecutor de pruebas no tenía disponible la herramienta adicional de inspección de medios: la recolección registró FileNotFoundError después de producir y guardar los resultados. Se recuperaron los archivos exactos y se verificaron localmente su integridad y decodificación, sin repetir el motor. El ZIP conserva los informes originales y esta verificación adicional por separado.</p>'
    body += table(["Control de igualdad", "Resultado"],
                  [[label, "Verificado" if a[key] else "No verificado"] for key, label in (
                      ("same_source_all_five", "Mismo archivo original en las cinco"),
                      ("same_code_all_five", "Misma revisión de código"),
                      ("same_package_all_five", "Mismo contenido del paquete del motor"),
                      ("same_asr_config_all_five", "Misma configuración ASR"),
                      ("same_alignment_packages_all_five", "Mismas versiones de alineación"))])
    body += f'<p>Run: <a href="https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/{batch.get("run_id", "")}">{batch.get("run_id", "")}</a>. Commit: <code>{esc(batch.get("build_sha", ""))}</code>. Original: <code>{esc(batch.get("expected_source_sha256", ""))}</code>.</p>'
    body += '<a class="button" download="Video00_GPT_WhisperX_5_Evidencias.zip" href="data:application/zip;base64,' + base64.b64encode(evidence.getvalue()).decode() + '">Descargar resultados y controles originales (ZIP)</a></details>'
    body += '''<footer>No se realizó escucha humana en esta sesión. Los controles automáticos no certifican todos los fonemas ni todas las decisiones de edición. Las cinco pruebas cubren un solo video. No se cambió producción ni se autorizó una sexta prueba.</footer>'''
    css = '''body{margin:0;background:#f2f5f8;color:#152d41;font:16px/1.6 system-ui,sans-serif}main{max-width:1140px;margin:28px auto;padding:32px;background:#fff;border-radius:16px}h1{font-size:36px;line-height:1.2;margin:10px 0}h2{margin-top:34px;font-size:23px}h3{font-size:18px}.eyebrow,.note,footer{color:#526778;font-size:13px}.lead{font-size:19px}.verdict{border-left:5px solid #b0751c;background:#fff4e2;padding:16px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.cards>div{background:#edf5fa;padding:16px;border-radius:10px}.cards b{display:block;font-size:30px}.cards span{font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dde6ed;text-align:left;vertical-align:top;padding:11px}th{background:#edf3f7}.scroll{overflow-x:auto}code{overflow-wrap:anywhere;font-size:12px}details{border:1px solid #dce5ec;border-radius:8px;margin:20px 0;padding:16px}summary{cursor:pointer;font-weight:650}.time{font-size:12px;white-space:nowrap;color:#597087}.button{display:inline-block;border:1px solid #829bad;padding:10px 15px;border-radius:7px;color:#17628d;text-decoration:none}footer{border-top:1px solid #dde6ed;margin-top:30px;padding-top:20px}@media(max-width:650px){main{padding:18px;margin:0;border-radius:0}.cards{grid-template-columns:1fr}h1{font-size:29px}td,th{padding:8px}}'''
    output.write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Video00 · Cinco pruebas GPT + WhisperX</title><style>' + css + '</style><main>' + body + '</main></html>')
    print(json.dumps({"report": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.artifacts, args.output)
