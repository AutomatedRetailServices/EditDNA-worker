"""Read-only artifact retrieval and lossless 20 MiB parts for workspace transfer.

Does not import or invoke Modal, the ASR or any editor. It only downloads
already-produced GitHub artifacts using the workflow's read-only token.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = "AutomatedRetailServices/EditDNA-worker"
run_id = os.environ["SOURCE_RUN_ID"]
index = int(sys.argv[1])
assert run_id.isdigit() and index in range(1, 6)
root = Path("collected") / f"trial-{index}"
reports = root / "reports"
reports.mkdir(parents=True, exist_ok=True)
name = f"video00-gptwx-trial-{index}"
for _ in range(120):
    raw = subprocess.check_output(["gh", "api", f"repos/{REPO}/actions/runs/{run_id}/artifacts"], text=True)
    artifacts = json.loads(raw)["artifacts"]
    if any(a["name"] == name and not a["expired"] for a in artifacts):
        break
    parent = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/actions/runs/{run_id}"], text=True))
    if parent["status"] == "completed":
        raise SystemExit(f"Parent completed without trial {index} artifact; no re-execution")
    time.sleep(30)
else:
    raise SystemExit("Artifact polling bound exhausted; no re-execution")
subprocess.run(["gh", "run", "download", run_id, "--repo", REPO, "--name", name,
                "--dir", str(root / "original")], check=True)
for path in (root / "original").rglob("*"):
    if not path.is_file():
        continue
    if path.suffix != ".mp4":
        shutil.copy2(path, reports / path.name)
        continue
    pieces = []
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for number, letter in enumerate("ABC"):
            block = source.read(20 * 1024 * 1024)
            if not block:
                break
            dest = root / "parts" / letter
            dest.mkdir(parents=True, exist_ok=True)
            filename = path.name + f".part{number}"
            (dest / filename).write_bytes(block)
            digest.update(block)
            pieces.append({"filename": filename, "part": letter, "bytes": len(block),
                           "sha256": hashlib.sha256(block).hexdigest()})
        if source.read(1):
            raise RuntimeError("Video exceeds the three-part transfer bound")
    (reports / "video-parts.json").write_text(json.dumps({"filename": path.name,
        "sha256": digest.hexdigest(), "bytes": path.stat().st_size, "parts": pieces}, indent=2))
print(f"Trial {index} collected; exact bytes split for transfer, no GPU/provider call")
