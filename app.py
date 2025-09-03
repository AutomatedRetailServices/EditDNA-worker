from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    # DEBUG: read the file and report size, nothing else
    data = await video.read()
    await video.seek(0)  # reset pointer for later use (when we restore real logic)
    print("=== /process hit ===")
    print("tone:", tone)
    print("features_csv:", features_csv)
    print("product_link:", product_link)
    print("video filename:", video.filename, "bytes:", len(data))

    return {"ok": True, "bytes": len(data)}
