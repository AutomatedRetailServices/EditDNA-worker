"""Pure AV response contract and offline replay. No provider or media calls."""
import json
import math
from types import SimpleNamespace


def response_schema(duration):
    text = {"type": "string", "minLength": 1}
    region = {"type": "object", "additionalProperties": False, "properties": {
        "start": {"type": "number", "minimum": 0, "maximum": duration},
        "end": {"type": "number", "minimum": 0, "maximum": duration},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "role": {"type": "string", "enum": ["audience", "mixed", "recording_only", "uncertain"]},
        **{key: text for key in ("audio_observation", "visual_observation", "reason")},
    }}
    region["required"] = list(region["properties"])
    return {"type": "object", "additionalProperties": False, "properties": {
        **{key: text for key in ("summary", "creator_intent", "story_logic")},
        "regions": {"type": "array", "maxItems": 12, "items": region},
    }, "required": ["summary", "creator_intent", "story_logic", "regions"]}


def captured_response(raw):
    # Retain visible response only; never credentials, media bytes or model thinking.
    return {"usageMetadata": raw.get("usageMetadata", {}), "candidates": [
        {"finishReason": c.get("finishReason"), "content": {"parts": [
            {"text": p["text"]} for p in c.get("content", {}).get("parts", [])
            if isinstance(p.get("text"), str) and not p.get("thought")
        ]}} for c in raw.get("candidates", [])
    ]}


def replay(record):
    if record.get("contract_version") != "cutsell.av.v1":
        raise ValueError("unsupported AV replay contract")
    return parse_response(record["response"], record["source_duration_sec"], record["prepared_duration_sec"])


def parse_response(raw, source_duration, duration):
    if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (source_duration, duration)):
        raise ValueError("AV_DURATION_INVALID")
    if abs(duration-source_duration) > .3:
        raise ValueError("AV_DURATION_MISMATCH")
    source = SimpleNamespace(duration_sec=source_duration)
    candidates=raw.get('candidates') or []
    if len(candidates)!=1 or candidates[0].get('finishReason')!='STOP':
        raise ValueError('AV response incomplete or blocked')
    text=''.join(p.get('text','') for p in candidates[0].get('content',{}).get('parts',[]) if not p.get('thought'))
    data=json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("AV_RESPONSE_TYPE")
    regions=data.get('regions')
    if not isinstance(regions,list) or len(regions)>12:
        raise ValueError('AV region contract invalid')
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f'AV_REGION_TYPE: region={index}')
        region = dict(region)
        regions[index] = region
        start,end,confidence=(region.get(k) for k in ('start','end','confidence'))
        if not all(type(v) in (int,float) and math.isfinite(v) for v in (start,end,confidence)):
            raise ValueError('AV region numeric evidence invalid')
        # Encoder padding is bounded above by the already-checked .3s
        # duration tolerance. Intersect advisory regions with real source;
        # never accept a wholly out-of-source region or invent cut times.
        if 0 <= start < source.duration_sec < end <= duration:
            region['encoder_padding_trimmed_sec'] = end - source.duration_sec
            region['end'] = end = source.duration_sec
        if not 0<=confidence<=1:
            raise ValueError(f'AV_CONFIDENCE_RANGE: region={index}, confidence={confidence}')
        if not 0<=start<end<=source.duration_sec:
            raise ValueError(f'AV_TIME_RANGE: region={index}, start={start}, end={end}, source_end={source.duration_sec}')
        if region.get('role') not in {'audience','mixed','recording_only','uncertain'}:
            raise ValueError('AV role invalid')
        region['observation_id'] = f'av_{index}'
        for key in ('audio_observation','visual_observation','reason'):
            if not isinstance(region.get(key),str) or not region[key].strip():
                raise ValueError('AV region missing modality evidence')
            region[key]=region[key][:240]
    for key in ('summary','creator_intent','story_logic'):
        if not isinstance(data.get(key),str) or not data[key].strip():
            raise ValueError('AV whole-source understanding missing')
    return data
