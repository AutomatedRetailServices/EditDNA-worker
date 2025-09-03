from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os, tempfile, json
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def classify_transcript(transcript: str, features_csv: str, tone: str):
    system = (
        "You are an ad pre-editor. From the transcript, extract ONLY text that "
        "already exists in the transcript (no rewriting). Find at most 1–3 lines for each:\n"
        "hook, feature_lines, proof_lines, cta_lines. Keep them short and punchy, "
        "verbatim from the transcript. Return strict JSON."
    )
    user = (
        f"Tone: {tone}\n"
        f"Key features to prioritize: {features_csv}\n\n"
        f"Transcript:\n{transcript}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
        max_tokens=400
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        # Ensure all keys exist
        return {
            "hook": data.get("hook", ""),
            "feature_lines": data.get("feature_lines", []),
            "proof_lines": data.get("proof_lines", []),
            "cta_lines": data.get("cta_lines", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    # --- keep your existing transcription code above this line ---
    # (we assume you already computed `transcript` and `data` bytes)

    # save file -> transcribe -> set transcript variable ...
    data = await video.read()
    await video.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json"
            )
        transcript = tr.text if hasattr(tr, "text") else tr.get("text", "")
    finally:
        try: os.remove(tmp_path)
        except: pass

    # NEW: classify into Hook / Feature / Proof / CTA
    buckets = classify_transcript(transcript, features_csv, tone)

    return JSONResponse({
        "ok": True,
        "bytes": len(data),
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
        "transcript": transcript[:500],
        "buckets": buckets
    })
