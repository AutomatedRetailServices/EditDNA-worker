"""Two authorized full-engine calls for one uploaded source; transport only."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.run_video00_gpt_whisperx_stability import OVERLAYS, require_media_tools, s3_client, write_json

EXPECTED_SHA = os.environ.get("UPLOAD_EXPECTED_SHA", "c9d892629f19d49058246e79bb57eb0af10bdf123416122307aa0bcd5c91cb14")
EXPECTED_BYTES = int(os.environ.get("UPLOAD_EXPECTED_BYTES", "12429383"))
PRIVATE = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "cutsell-upload-comparison"
OUT = ROOT / "uploaded-comparison-artifacts"


def prepare(existing_source_key=None, single_provider=False):
    import requests
    selected_provider = os.environ.get("UPLOAD_SINGLE_PROVIDER", "gpt-whisperx")
    if selected_provider not in {"gpt-whisperx", "deepgram"}:
        raise ValueError("Unsupported single provider")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from modal_gpu_config import require_modal_token_env
    require_media_tools()
    require_modal_token_env(os.environ)
    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise RuntimeError("No automatic paid reruns")
    PRIVATE.mkdir(mode=0o700, exist_ok=False)
    OUT.mkdir(exist_ok=True)
    response = requests.get("https://rest.runpod.io/v1/templates", headers={
        "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]}, timeout=(15, 60))
    if response.status_code != 200:
        raise RuntimeError(f"Template preflight HTTP {response.status_code}")
    templates = [t for t in response.json() if t.get("name") == "EditDNA-Worker-2"]
    if len(templates) != 1:
        raise RuntimeError("Canonical template is ambiguous or absent")
    env = {str(k): str(v) for k, v in templates[0]["env"].items()}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    env.update(OVERLAYS, CUTSELL_BUILD_GIT_SHA=head, CUTSELL_ASR_MODEL="medium")
    required_key = "DEEPGRAM_API_KEY" if selected_provider == "deepgram" else "OPENAI_API_KEY"
    if not env.get(required_key) or env[required_key].startswith("sk-admin-"):
        raise RuntimeError("Required inference key missing; no GPU started")
    for k, v in env.items():
        if v and (len(v) >= 12 or any(s in k for s in ("KEY", "TOKEN", "SECRET"))) and k not in OVERLAYS and k != "CUTSELL_BUILD_GIT_SHA":
            print("::add-mask::" + v.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))
    write_json(PRIVATE / "environment.json", env)
    (PRIVATE / "environment.json").chmod(0o600)
    if existing_source_key:
        key = existing_source_key
    else:
        key = f"cutsell/benchmark-inputs/upload-{os.environ['GITHUB_RUN_ID']}/{EXPECTED_SHA}.mp4"
        # Only a short-lived, one-object PUT capability is returned, encrypted to
        # the requester's ephemeral public key. AWS/inference keys never leave CI.
        url = s3_client(env).generate_presigned_url("put_object", Params={
            "Bucket": env["S3_BUCKET"], "Key": key, "ContentType": "video/mp4",
            "ContentLength": EXPECTED_BYTES}, ExpiresIn=1200, HttpMethod="PUT")
        public = serialization.load_pem_public_key((ROOT / "benchmarks/upload_comparison_public.pem").read_bytes())
        aes_key, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
        ciphertext = AESGCM(aes_key).encrypt(nonce, json.dumps({"url": url, "source_key": key}).encode(), None)
        wrapped = public.encrypt(aes_key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
        write_json(OUT / "upload-response.enc.json", {"wrapped_key": base64.b64encode(wrapped).decode(),
            "nonce": base64.b64encode(nonce).decode(), "ciphertext": base64.b64encode(ciphertext).decode()})
    write_json(PRIVATE / "source.json", {"key": key})
    write_json(OUT / "manifest.json", {"source_sha256": EXPECTED_SHA, "source_bytes": EXPECTED_BYTES,
        "build_sha": head, "run_id": os.environ["GITHUB_RUN_ID"], "authorized_runs": 1 if existing_source_key or single_provider else 2,
        "providers": ["deepgram-nova-3-multi" if selected_provider == "deepgram" else "gpt-transcribe-whisperx"] if existing_source_key or single_provider else ["faster-whisper-medium", "gpt-transcribe-whisperx"], "retries": 0,
        "same_template_snapshot": True, "auto_speech_visual_microtrim": True})
    if existing_source_key or single_provider:
        write_json(PRIVATE / "single-provider.json", {"provider": selected_provider})
    print("Source preflight prepared; no GPU call yet")


def await_upload():
    from botocore.exceptions import ClientError
    env = json.loads((PRIVATE / "environment.json").read_text())
    key = json.loads((PRIVATE / "source.json").read_text())["key"]
    client = s3_client(env)
    for _ in range(90):
        try:
            obj = client.head_object(Bucket=env["S3_BUCKET"], Key=key)
            if obj["ContentLength"] == EXPECTED_BYTES:
                body = client.get_object(Bucket=env["S3_BUCKET"], Key=key)["Body"]
                digest = hashlib.sha256()
                for chunk in iter(lambda: body.read(1024 * 1024), b""):
                    digest.update(chunk)
                if digest.hexdigest() != EXPECTED_SHA:
                    raise RuntimeError("Uploaded source hash mismatch; no GPU started")
                (PRIVATE / "source.verified").touch()
                print("Uploaded source size and SHA-256 verified")
                return
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        time.sleep(10)
    raise RuntimeError("Upload deadline reached; no GPU started")


def run(provider):
    if provider not in {"medium", "gpt-whisperx", "deepgram"} or os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise RuntimeError("Unsupported provider or repeat attempt")
    if not (PRIVATE / "source.verified").exists() or (PRIVATE / "STOP").exists():
        raise RuntimeError("Source or prior terminal-state preflight failed")
    single = PRIVATE / "single-provider.json"
    if provider == "deepgram" and not single.exists():
        raise RuntimeError("Deepgram requires explicit single-provider mode")
    if single.exists() and provider != json.loads(single.read_text())["provider"]:
        raise RuntimeError("Provider outside single-run authorization")
    if provider == "gpt-whisperx" and not single.exists() and not (PRIVATE / "medium.terminal").exists():
        raise RuntimeError("Sequential calls required")
    with (PRIVATE / f"{provider}.claimed").open("x") as f:
        f.write("No retry")
    env = json.loads((PRIVATE / "environment.json").read_text())
    env["CUTSELL_VALIDATION_ASR_PROVIDER"] = {"medium": "faster-whisper", "gpt-whisperx": "gpt-transcribe-whisperx", "deepgram": "deepgram-nova-3-multi"}[provider]
    config = PRIVATE / f"{provider}.json"
    write_json(config, env)
    key = json.loads((PRIVATE / "source.json").read_text())["key"]
    bid = f"uploaded-asr-{os.environ['GITHUB_RUN_ID']}-{provider}"
    output = OUT / provider
    reports = output / "reports"
    reports.mkdir(parents=True)
    child = dict(os.environ, CUTSELL_VALIDATION_ASR_PROVIDER=env["CUTSELL_VALIDATION_ASR_PROVIDER"],
                 CUTSELL_ENV_JSON_PATH=str(config), CUTSELL_BENCHMARK_PAYLOAD_JSON=json.dumps({
                     "op": "focused", "source_key": key, "benchmark_id": bid,
                     "auto_speech_visual_microtrim": True}))
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_SESSION_TOKEN", "S3_BUCKET"):
        if k in env:
            child[k] = env[k]
    compact_path = ROOT / "modal-video00-result.json"
    if compact_path.exists():
        raise RuntimeError("Unexpected previous uncollected result")
    began = time.monotonic()
    with (reports / "modal.log").open("w") as log:
        call = subprocess.run(["modal", "run", "modal_video00_full_benchmark.py"], env=child,
                              stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if call.returncode or not compact_path.exists():
        (PRIVATE / "STOP").touch()
        raise RuntimeError("No confirmed terminal Modal response; no subsequent GPU call")
    compact_path.replace(reports / "compact.json")
    compact = json.loads((reports / "compact.json").read_text())
    if compact.get("terminal_state") == "local_wrapper_exception":
        (PRIVATE / "STOP").touch()
    (PRIVATE / f"{provider}.terminal").touch()
    client = s3_client(env)
    files = [("result_uri", reports / "result.json"), ("preview_uri", output / f"{provider}.mp4"),
             ("diagnostic_preview_uri", output / f"{provider}-DIAGNOSTICO.mp4")]
    for field, dest in files:
        if compact.get(field):
            uri = urlparse(compact[field])
            if uri.scheme != "s3" or uri.netloc != env["S3_BUCKET"]:
                raise RuntimeError("Unexpected artifact location")
            client.download_file(uri.netloc, uri.path.lstrip("/"), str(dest))
    for video in output.glob("*.mp4"):
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], text=True)
        write_json(reports / "media.json", json.loads(probe))
        pieces = []
        digest = hashlib.sha256()
        with video.open("rb") as source:
            for n, letter in enumerate("ABCDEFGHIJKL"):
                block = source.read(20 * 1024 * 1024)
                if not block:
                    break
                dest = output / letter
                dest.mkdir()
                name = video.name + f".part{n}"
                (dest / name).write_bytes(block)
                digest.update(block)
                pieces.append({"filename": name, "sha256": hashlib.sha256(block).hexdigest()})
            if source.read(1):
                raise RuntimeError("Artifact exceeds part bound")
        write_json(reports / "parts.json", {"filename": video.name, "sha256": digest.hexdigest(), "parts": pieces})
    write_json(reports / "summary.json", {"provider": provider, "engine_ok": compact.get("ok"),
        "wall_elapsed_sec": round(time.monotonic() - began, 3), "source_match": compact.get("source_media_sha256") == EXPECTED_SHA,
        "deliverable": compact.get("deliverable"), "qc": compact.get("live_render_qc_status"),
        "selected_count": compact.get("selected_count"), "output_duration_sec": compact.get("output_duration_sec")})
    print("Completed", provider, "QC", compact.get("live_render_qc_status"))


if __name__ == "__main__":
    {"prepare-single": lambda: prepare(single_provider=True), "prepare-existing": lambda: prepare(sys.argv[2]), "prepare": prepare, "await-upload": await_upload, "run": lambda: run(sys.argv[2])}[sys.argv[1]]()
