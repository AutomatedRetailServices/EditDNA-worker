"""CPU-only side-by-side report of existing five-run provider batches."""
import base64
import hashlib
import html
import io
import itertools
import json
from pathlib import Path
import statistics
import sys
import zipfile
from compare_gpt_whisperx_stability import compare_pair, lexical


def read_batch(folder):
    rows, results = [], {}
    for i in range(1,6):
        p=folder/f'trial-{i}'
        if not (p/'result.json').exists():
            rows.append({'trial':i,'missing_result':True})
            continue
        r=json.loads((p/'result.json').read_text());c=json.loads((p/'compact.json').read_text())
        e=json.loads((p/'editorial-qa.json').read_text());a=r.get('asr_provider_audit') or {}
        words=[w for s in r['timed_asr_replay_evidence']['raw_segments'] for w in s['words']]
        request_ids=[x['request_id'] for x in a.get('chunks',[]) if x.get('request_id')]
        if a.get('request_id'):request_ids.append(a['request_id'])
        rows.append({'trial':i,'engine_ok':c.get('ok'),'source_sha':r.get('source_media_sha256'),
            'build_sha':r.get('active_path_identity',{}).get('build_git_sha'),
            'words':len(words),'zero_duration_words':sum(w['end']==w['start'] for w in words),
            'overlapping_word_pairs':a.get('overlapping_word_pairs'),
            'output_sec':r.get('output_duration_sec'),'engine_sec':r.get('elapsed_sec'),
            'selected_count':len(r.get('selected',[])),'qc':r.get('live_render_qc',{}).get('status'),
            'delivery_status':c.get('delivery_status'),'editorial_passed':e['passed_check_count'],
            'editorial_failed':e['failed_checks'],'request_ids':request_ids,
            'cache_hits':a.get('cache_hit_count'),'selected':r.get('selected'),
            'lexical_hash':hashlib.sha256(json.dumps(lexical(' '.join(w['text'] for w in words))).encode()).hexdigest(),
            'qc_findings':[f for x in r.get('live_render_qc',{}).get('attempts',[]) for f in x.get('findings',[])]})
        results[i]=r
    pairs=[compare_pair(i,a,j,b) for (i,a),(j,b) in itertools.combinations(results.items(),2)]
    valid=[r for r in rows if not r.get('missing_result')]
    counts={}
    for row in valid:
        for failure in row['editorial_failed']:
            counts[failure['id']]=counts.get(failure['id'],0)+1
    stats={'available':len(valid),'all_source_same':len({r['source_sha'] for r in valid})==1,
        'distinct_requests':len({x for r in valid for x in r['request_ids']}),
        'unique_lexical_transcripts':len({r['lexical_hash'] for r in valid}),
        'failure_frequency':counts,
        'editorial_mean':statistics.mean(r['editorial_passed'] for r in valid) if valid else None,
        'editorial_complete':sum(r['editorial_passed']==11 for r in valid),
        'technical_passes':sum(r['qc']=='PASS' for r in valid),
        'output_min_sec':min((r['output_sec'] for r in valid if r['output_sec'] is not None),default=None),
        'output_max_sec':max((r['output_sec'] for r in valid if r['output_sec'] is not None),default=None),
        'selection_iou_min_percent':min((p['selected_source_coverage_iou_percent'] for p in pairs),default=None),
        'selection_iou_max_percent':max((p['selected_source_coverage_iou_percent'] for p in pairs),default=None),
        'lexical_disagreement_max_percent':max((p['lexical_difference_percent'] for p in pairs),default=None)}
    return {'trials':rows,'statistics':stats,'pairs':pairs}


def report(deepgram, gpt, output):
    data={'deepgram':read_batch(deepgram),'gpt_whisperx':read_batch(gpt),'limitations':[
        'One source only; no human listening or reference transcript.',
        'Provider, timing and segmentation differ; no causal attribution to word recognition alone.',
        'Editorial source-range checks are not ASR accuracy or full audiovisual acceptance.',
        'Prior GPT batch and new Deepgram batch share editorial code, but not identical whole package or runtime snapshots.',
        'No provider billing queries; request count is not dollar cost.']}
    (output/'Video00_Deepgram_vs_GPT_5_Resultados.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    esc=lambda x:html.escape(str(x))
    def table(headers,rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    body='<h1>Video00: cinco pruebas Deepgram frente a cinco GPT + WhisperX</h1><p>Cinco ejecuciones nuevas de Deepgram, sin reutilizar transcripciones entre pruebas. Comparadas con las cinco ejecuciones previas de GPT. Se conservan todos los resultados, incluidos los bloqueados. No se modificaron las reglas editoriales para esta tanda.</p>'
    body+='<p><strong>Valoración de la creadora:</strong> la prueba individual anterior de Deepgram (2:23, 10/11) fue considerada un Clean Cut útil. No forma parte de estas cinco ejecuciones. Los criterios puntuales y las etiquetas técnicas se presentan aparte de esa valoración; no equivalen a declarar que todo el video esté mal. La capa comercial del funnel permanece integrada en la dirección del mismo editor, sin haberse modificado para esta tanda.</p>'
    body+=table(['Prueba','Deepgram: s','GPT: s','Deepgram: criterios /11','GPT: criterios /11','Deepgram: QC','GPT: QC'],[[i+1,data['deepgram']['trials'][i].get('output_sec'),data['gpt_whisperx']['trials'][i].get('output_sec'),data['deepgram']['trials'][i].get('editorial_passed'),data['gpt_whisperx']['trials'][i].get('editorial_passed'),data['deepgram']['trials'][i].get('qc'),data['gpt_whisperx']['trials'][i].get('qc')] for i in range(5)])
    body+='<p>Los números de prueba son índices de ejecución; no significan pares controlados uno a uno. Los archivos DIAGNÓSTICO/INVALIDADO no son entregables aprobados.</p>'
    body+=table(['Medida','Deepgram','GPT + WhisperX'],[[k,data['deepgram']['statistics'][k],data['gpt_whisperx']['statistics'][k]] for k in ['editorial_mean','editorial_complete','technical_passes','distinct_requests','unique_lexical_transcripts','selection_iou_min_percent','selection_iou_max_percent','lexical_disagreement_max_percent']])
    body+='<p>IoU mide cuánto coinciden los intervalos seleccionados entre pruebas: mayor significa más estabilidad, no necesariamente mejor edición. La diferencia léxica compara proveedores consigo mismos entre ejecuciones; no es WER frente a una transcripción humana.</p>'
    body+='<h2>Fallos editoriales por frecuencia</h2>'+table(['Criterio','Deepgram /5','GPT /5'],[[k,data['deepgram']['statistics']['failure_frequency'].get(k,0),data['gpt_whisperx']['statistics']['failure_frequency'].get(k,0)] for k in sorted(set(data['deepgram']['statistics']['failure_frequency'])|set(data['gpt_whisperx']['statistics']['failure_frequency']))])
    for key,title in [('deepgram','Deepgram'),('gpt_whisperx','GPT + WhisperX')]:
        for row in data[key]['trials']:
            body+='<details><summary>'+title+' — prueba '+str(row['trial'])+'</summary>'
            body+='<p>Estado de entrega: '+esc(row.get('delivery_status'))+'</p>'
            body+=table(['Inicio original','Fin original','Texto'],[[s['start'],s['end'],s['text']] for s in row.get('selected',[])])
            body+='<pre>'+esc(json.dumps({'failed':row.get('editorial_failed'),'qc_findings':row.get('qc_findings')},ensure_ascii=False,indent=2))+'</pre></details>'
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for key,folder in [('deepgram',deepgram),('gpt',gpt)]:
            for p in sorted(folder.rglob('*.json')):z.writestr(key+'/'+p.relative_to(folder).as_posix(),p.read_bytes())
    body+='<p><a download="Video00_10_Pruebas_Evidencias.zip" href="data:application/zip;base64,'+base64.b64encode(b.getvalue()).decode()+'">Descargar evidencia original de las diez pruebas</a></p><p>Sin escucha humana. Sin cambio de proveedor de producción. Los cinco ensayos sobre un video no establecen resultados generales para otros idiomas o fuentes.</p>'
    (output/'Video00_Deepgram_vs_GPT_5_Informe.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Video00: Deepgram vs GPT, cinco pruebas</title><style>body{font:16px/1.6 system-ui;background:#f3f5f8;color:#183247;margin:0}main{max-width:1100px;margin:24px auto;background:white;padding:24px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}table{width:100%;font-size:14px}.scroll{overflow:auto}pre{white-space:pre-wrap;font-size:12px}details{padding:12px;border:1px solid #ddd;margin:12px 0}summary{cursor:pointer;font-weight:bold}</style><main>'+body+'</main></html>')
    print(json.dumps({k:v['statistics'] for k,v in data.items() if k!='limitations'},indent=2))


if __name__=='__main__':
    report(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
