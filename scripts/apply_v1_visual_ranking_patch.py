"""One-shot V1 patch: visual models rank/flag valid takes; they do not delete them."""
from pathlib import Path

path = Path("worker/pipeline.py")
text = path.read_text(encoding="utf-8")

old_bad = '''        if verdict is Verdict.BAD:\n            c["meta"]["keep"] = False\n            c["llm_reason"] = (c.get("llm_reason") or "") + " | Removed for visual bad-take."\n'''
new_bad = '''        if verdict is Verdict.BAD:\n            # Acting/visual quality belongs to Best Take ranking, not Clean Cut\n            # deletion. Keep the valid take available for editor swap/restore.\n            c["meta"]["visual_bad_take"] = True\n            c["meta"]["visual_quality_flag"] = "bad_take"\n            c["llm_reason"] = (c.get("llm_reason") or "") + " | Flagged as lower-quality visual take."\n'''
if old_bad in text:
    text = text.replace(old_bad, new_bad, 1)
elif 'c["meta"]["visual_quality_flag"] = "bad_take"' not in text:
    raise SystemExit("Unexpected visual bad-take baseline")

old_all_bad = '''        # Caso extremo: todo malo → matar el clip completo.\n        if head_bad and mid_bad and tail_bad:\n            c["meta"]["keep"] = False\n            c["llm_reason"] = (c.get("llm_reason") or "") + " | Removed by boundary refiner: full take visually bad."\n            changed_any = True\n            continue\n'''
new_all_bad = '''        # A visually weak full take remains an editable alternate. Boundary\n        # refinement may flag quality but is not deletion authority.\n        if head_bad and mid_bad and tail_bad:\n            c["meta"]["boundary_refiner_quality_flag"] = "all_frames_bad"\n            c["llm_reason"] = (c.get("llm_reason") or "") + " | Boundary refiner flagged lower visual quality."\n            changed_any = True\n            continue\n'''
if old_all_bad in text:
    text = text.replace(old_all_bad, new_all_bad, 1)
elif 'boundary_refiner_quality_flag"] = "all_frames_bad"' not in text:
    raise SystemExit("Unexpected boundary all-bad baseline")

old_short = '''        # Si después del recorte queda ridículamente corto, lo matamos\n        if new_end <= new_start or (new_end - new_start) < max(0.5, duration * 0.25):\n            c["meta"]["keep"] = False\n            c["llm_reason"] = (c.get("llm_reason") or "") + " | Removed by boundary refiner: too short after trim."\n            changed_any = True\n            continue\n'''
new_short = '''        # If a proposed visual trim would destroy the spoken take, preserve the\n        # original boundaries and expose the quality flag instead.\n        if new_end <= new_start or (new_end - new_start) < max(0.5, duration * 0.25):\n            c["meta"]["boundary_refiner_quality_flag"] = "trim_rejected_too_short"\n            c["llm_reason"] = (c.get("llm_reason") or "") + " | Unsafe visual trim rejected; original take preserved."\n            changed_any = True\n            continue\n'''
if old_short in text:
    text = text.replace(old_short, new_short, 1)
elif 'boundary_refiner_quality_flag"] = "trim_rejected_too_short"' not in text:
    raise SystemExit("Unexpected boundary short-trim baseline")

path.write_text(text, encoding="utf-8")
print("Visual quality converted from deletion authority to ranking flags")
