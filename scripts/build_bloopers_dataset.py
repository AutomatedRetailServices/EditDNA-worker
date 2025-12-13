import json
from pathlib import Path

# Folder where all your Bloppers*.json files are stored
BLOOPERS_DIR = Path("data/bloopers_json")

# Output dataset file (one JSON per line)
OUTPUT_JSONL = Path("data/datasets/bloopers_v1.jsonl")


def iter_bloopers():
    """
    Iterate over all BloppersXX.json files and yield one row per clip.
    """
    if not BLOOPERS_DIR.exists():
        raise SystemExit(f"Bloopers folder not found: {BLOOPERS_DIR.resolve()}")

    # You can adjust this pattern if your filenames are different
    json_paths = sorted(BLOOPERS_DIR.glob("Bloppers*.json"))

    if not json_paths:
        raise SystemExit(f"No Bloppers JSON files found in {BLOOPERS_DIR.resolve()}")

    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            job = json.load(f)

        job_id = job.get("job_id") or job.get("id")
        result = job.get("result") or {}
        session_id = result.get("session_id")
        duration = result.get("duration_sec")
        clips = result.get("clips", [])

        for clip in clips:
            meta = clip.get("meta") or {}

            # Basic label:
            #   keep == True  → label_blooper = 0 (good / usable)
            #   keep == False → label_blooper = 1 (blooper / bad)
            keep_flag = bool(meta.get("keep", False))
            label_blooper = 0 if keep_flag else 1

            row = {
                "job_id": job_id,
                "session_id": session_id,
                "video_duration_sec": duration,
                "clip_id": clip.get("id"),
                "slot": clip.get("slot"),
                "start": clip.get("start"),
                "end": clip.get("end"),
                "text": clip.get("text"),
                "llm_reason": clip.get("llm_reason"),

                "score": clip.get("score"),
                "semantic_score": clip.get("semantic_score"),
                "visual_score": clip.get("visual_score"),
                "face_q": clip.get("face_q"),
                "scene_q": clip.get("scene_q"),
                "vtx_sim": clip.get("vtx_sim"),

                # From meta
                "keep": meta.get("keep"),
                "take_judge_score": meta.get("take_judge_score"),
                "take_judge_verdict": meta.get("take_judge_verdict"),

                # Global labels / tags
                "label_blooper": label_blooper,  # 1 = bad/blooper, 0 = good
                "source": "bloopers_v1",
            }

            yield row


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUTPUT_JSONL.open("w", encoding="utf-8") as out_f:
        for row in iter_bloopers():
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Written {count} rows to {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()
