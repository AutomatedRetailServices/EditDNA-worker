"""CPU-only descriptive report for an uploaded-source provider comparison."""
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import sys
import zipfile

from compare_gpt_whisperx_stability import compare_pair


def report(folder: Path, destination: Path):
    records, raw_results = [], {}
    for provider in ("medium", "gpt-whisperx"):
        p = folder / provider
        compact = json.loads((p / "compact.json").read_text())
        row = {"provider": provider, "engine_ok": compact.get("ok"),
               "error": compact.get("error"), "error_type": compact.get("error_type"),
               "deliverable": compact.get("deliverable"), "delivery_status": compact.get("delivery_status")}
        if (p / "result.json").exists():
            r = json.loads((p / "result.json").read_text())
            raw_results[provider] = r
            words = [w for s in r["timed_asr_replay_evidence"]["raw_segments"] for w in s["words"]]
            audit = r.get("asr_provider_audit") or {}
            qc = r.get("live_render_qc") or {}
            freeze = r.get("diagnostics", {}).get("selection_boundary_contract", {})
            row.update({"model": r.get("models", {}).get("asr"), "words": len(words),
                        "word_durations_zero": sum(w["end"] == w["start"] for w in words),
                        "words_over_1_5_sec": [w for w in words if w["end"] - w["start"] > 1.5],
                        "source_duration_sec": r.get("source_duration_sec"),
                        "source_sha256": r.get("source_media_sha256"),
                        "build_sha": r.get("active_path_identity", {}).get("build_git_sha"),
                        "package_sha256": r.get("active_path_identity", {}).get("package", {}).get("sha256"),
                        "selected_count": r.get("selected_count"), "discarded_count": r.get("discarded_count"),
                        "output_duration_sec": r.get("output_duration_sec"), "elapsed_sec": r.get("elapsed_sec"),
                        "qc": qc.get("status"), "freeze_status": freeze.get("status"),
                        "freeze_matches_reviewed_plan": freeze.get("matches_reviewed_plan"),
                        "freeze_diagnostics": r.get("diagnostics", {}).get("selection_freeze_diagnostics"),
                        "reviewer": r.get("diagnostics", {}).get("final_edit_reviewer"),
                        "qc_findings": [f for a in qc.get("attempts", []) for f in a.get("findings", [])],
                        "watch_listen": r.get("perceptual_watch_listen"),
                        "gpt_request_count": len(audit.get("chunks") or []),
                        "cache_hit_count": audit.get("cache_hit_count"),
                        "selected": r.get("selected"), "discarded": r.get("discarded"),
                        "transcript": [{"start": s["start"], "end": s["end"], "text": s["text"]}
                                       for s in r["timed_asr_replay_evidence"]["raw_segments"]]})
        records.append(row)
    data = {"results": records, "limitations": ["One full-engine call per provider, not a stability battery.",
            "No human audio listening or ground-truth transcript; no WER/accuracy claim.",
            "Video00-specific editorial criteria do not apply to this source."]}
    if len(raw_results) == 2:
        data["pairwise"] = compare_pair(1, raw_results["medium"], 2, raw_results["gpt-whisperx"])
    destination.mkdir(exist_ok=True)
    (destination / "NuevoVideo_Comparacion.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    esc = html.escape
    def table(headers, rows):
        return '<div class="scroll"><table><tr>' + ''.join('<th>'+esc(str(h))+'</th>' for h in headers) + '</tr>' + ''.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in row)+'</tr>' for row in rows) + '</table></div>'
    body = '<h1>Video recibido: Medium frente a GPT + WhisperX</h1><p>Dos ejecuciones completas del mismo archivo de 55.401 segundos, en la misma revisión del motor. Comparación exploratoria: una ejecución por proveedor.</p>'
    body += '<p><strong>Veredicto: mantener Medium como base por ahora.</strong> En esta prueba, GPT + WhisperX transcribió y alineó 157 palabras, pero la selección del motor conservó solo 3.66 segundos y perdió contenido; el control de coherencia impidió generar el video. Medium generó 16 segundos, pero su control perceptual bloqueó la entrega por contenido repetido. Ninguno produjo una edición aprobada.</p><p>GPT tuvo cero palabras de duración cero frente a diez en Medium. Esto mejora esa propiedad temporal; no demuestra por sí solo mejor exactitud ni mejor edición. Los 73 s de GPT no son comparables como velocidad final con los 100 s de Medium, porque GPT no llegó al render.</p>'
    body += table(["Proveedor", "Palabras", "Duración cero", "Clips elegidos", "Salida (s)", "Control técnico", "Entrega", "Motor (s)"],
                  [[r["provider"],r.get("words","—"),r.get("word_durations_zero","—"),r.get("selected_count","—"),r.get("output_duration_sec","—"),r.get("qc","Sin resultado"),r.get("delivery_status","—"),r.get("elapsed_sec","—")] for r in records])
    body += '<p>Los videos marcados DIAGNÓSTICO fueron bloqueados por el motor. Un PASS técnico sigue sujeto a la revisión humana de contenido. Los criterios particulares de Video00 no se utilizaron en este video.</p>'
    for row in records:
        body += '<h2>'+esc(row["provider"])+'</h2>'
        if row.get("error"):
            body += '<p>'+esc(row["error"][:1500])+'</p>'
        if row.get("transcript"):
            body += '<p>Modelo: '+esc(str(row["model"]))+'. Contrato del plan: '+esc(str(row["freeze_status"]))+'.</p>'
            body += '<h3>Texto del plan seleccionado y tiempos en el original (puede ser bloqueado antes del render)</h3>'
            body += table(["Inicio", "Fin", "Texto"],[[round(s["start"],3),round(s["end"],3),s["text"]] for s in row["selected"]])
            body += '<details><summary>Transcripción completa</summary>'
            body += table(["Inicio", "Fin", "Texto"],[[round(s["start"],3),round(s["end"],3),s["text"]] for s in row["transcript"]])+'</details>'
            body += '<details><summary>Descartes y hallazgos automáticos</summary><pre>'+esc(json.dumps({"discarded":row["discarded"],"qc_findings":row["qc_findings"], "freeze_diagnostics":row["freeze_diagnostics"], "reviewer":row["reviewer"], "watch_listen":row["watch_listen"]},ensure_ascii=False,indent=2))+'</pre></details>'
    if "pairwise" in data:
        pair=data["pairwise"]
        body += '<h2>Diferencias de transcripción</h2><p>Diferencia léxica entre transcripciones: '+str(pair["lexical_difference_percent"])+' %. Esto mide desacuerdo entre proveedores, no exactitud contra una transcripción humana. La tabla conserva puntuación y por eso también muestra cambios superficiales que no cuentan como errores léxicos.</p>'
        body += table(["Tiempo original", "Medium", "GPT + WhisperX"],[[d["source_time_a"],d["trial_a_text"] or "∅",d["trial_b_text"] or "∅"] for d in pair["text_differences"]])
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*.json")):
            archive.writestr(path.relative_to(folder).as_posix(),path.read_bytes())
    body += '<p><a download="NuevoVideo_Evidencias.zip" href="data:application/zip;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">Descargar evidencia original (ZIP)</a></p><p class="note">Sin escucha humana en esta sesión. La confianza de WhisperX describe alineación CTC; no es directamente comparable con la confianza de transcripción de Medium. No se cambió el proveedor de producción.</p>'
    css='body{font:16px/1.6 system-ui,sans-serif;color:#183247;background:#f2f5f8;margin:0}main{max-width:1080px;background:white;margin:25px auto;padding:30px;border-radius:14px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #dde5ee}th{background:#edf3f8}.scroll{overflow:auto}details{padding:14px;border:1px solid #d7e2eb;margin:16px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.note{font-size:13px;color:#587084}h2{margin-top:32px}@media(max-width:650px){main{margin:0;padding:16px}}'
    (destination / "NuevoVideo_Comparacion.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Comparación del video recibido</title><style>'+css+'</style><main>'+body+'</main></html>')
    print(json.dumps([{k:r.get(k) for k in ("provider","model","words","word_durations_zero","selected_count","output_duration_sec","qc","elapsed_sec","error")} for r in records],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    report(Path(sys.argv[1]),Path(sys.argv[2]))
