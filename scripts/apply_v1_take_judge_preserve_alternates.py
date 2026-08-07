"""One-shot V1 patch: Take Judge ranks takes but never deletes valid alternates."""
from pathlib import Path

path = Path("worker/pipeline.py")
text = path.read_text(encoding="utf-8")

old = '''            clip_item["meta"]["take_judge_execution_status"] = (\n                "candidate_winner" if candidate_id == result.winner_id else "candidate_loser"\n            )\n            if candidate_id != result.winner_id and clip_item["meta"].get("keep", True):\n                clip_item["meta"]["keep"] = False\n                clip_item["llm_reason"] = (clip_item.get("llm_reason") or "") + (\n                    " | Removed by TakeJudgeAI (better take exists)."\n                )\n'''
new = '''            clip_item["meta"]["take_judge_execution_status"] = (\n                "candidate_winner" if candidate_id == result.winner_id else "candidate_loser"\n            )\n            # Best Take is ranking authority, not Clean Cut deletion authority.\n            # Losers remain valid alternates for swap/restore in the editable draft.\n            clip_item["meta"]["take_judge_selected"] = candidate_id == result.winner_id\n'''
if old in text:
    text = text.replace(old, new, 1)
elif 'clip_item["meta"]["take_judge_selected"] = candidate_id == result.winner_id' not in text:
    raise SystemExit("Unexpected Take Judge winner baseline")

old_rank = '''            sem1 = safe_float(c1.get("semantic_score", 0.0))\n            sem2 = safe_float(c2.get("semantic_score", 0.0))\n            len1 = content1.effective_semantic_units\n            len2 = content2.effective_semantic_units\n\n            if sem2 > sem1 or (sem2 == sem1 and len2 >= len1):\n                c1["meta"]["keep"] = False\n                break\n            else:\n                c2["meta"]["keep"] = False\n'''
new_rank = '''            sem1 = safe_float(c1.get("semantic_score", 0.0))\n            sem2 = safe_float(c2.get("semantic_score", 0.0))\n            len1 = content1.effective_semantic_units\n            len2 = content2.effective_semantic_units\n\n            # Composer operates on copies, so it may suppress an alternate from\n            # this render while the original candidate remains editable. Prefer\n            # an explicit Take Judge winner before ordinary semantic tie-breakers.\n            rank1 = (\n                1 if c1.get("meta", {}).get("take_judge_selected") else 0,\n                safe_float(c1.get("meta", {}).get("take_judge_score", 0.0)),\n                safe_float(c1.get("score", sem1)), sem1, len1,\n            )\n            rank2 = (\n                1 if c2.get("meta", {}).get("take_judge_selected") else 0,\n                safe_float(c2.get("meta", {}).get("take_judge_score", 0.0)),\n                safe_float(c2.get("score", sem2)), sem2, len2,\n            )\n            if rank2 >= rank1:\n                c1["meta"]["keep"] = False\n                break\n            c2["meta"]["keep"] = False\n'''
if old_rank in text:
    text = text.replace(old_rank, new_rank, 1)
elif 'rank1 = (' not in text or 'take_judge_selected' not in text:
    raise SystemExit("Unexpected duplicate ranking baseline")

path.write_text(text, encoding="utf-8")
print("Take Judge now preserves alternates and ranks composer copies")
