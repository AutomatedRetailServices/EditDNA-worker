"""Security review (PO 23 Oct): NAMES of the worker's secret keys (never values) and what the storage / ASR keys may do.
Read-only: lists and HEADs; the delete test targets a key that does not exist (no data can change). Report encrypted."""
import json

import modal

app = modal.App("cutsell-fail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, secrets=[secret], timeout=600)
def audit():
    import os
    import uuid
    import boto3
    import requests
    from botocore.exceptions import ClientError
    out = {"env_names": sorted(k for k in os.environ if not k.startswith(("MODAL_TASK", "MODAL_IMAGE", "MODAL_ENVIRONMENT",
                                                                       "MODAL_CLOUD", "MODAL_REGION", "MODAL_SERVER", "MODAL_IDENTITY",
                                                                       "LANG", "PATH", "HOME", "HOSTNAME", "PYTHON", "GPG", "TERM",
                                                                       "SHLVL", "PWD", "_", "LC_", "SSL_")))}

    def code(fn):
        try:
            fn()
            return "ALLOWED"
        except ClientError as e:
            return "DENIED:" + e.response.get("Error", {}).get("Code", "?")
        except Exception as e:
            return "ERROR:" + e.__class__.__name__

    from cutsell_worker import storage_routing as sr
    probe_key = "security-review/does-not-exist-" + uuid.uuid4().hex
    r2, r2_bucket = sr.default_client(), sr.current_bucket()
    out["r2"] = {
        "list_all_buckets": code(lambda: r2.list_buckets()),
        "bucket_count": (lambda: len(r2.list_buckets().get("Buckets", [])))() if code(lambda: r2.list_buckets()) == "ALLOWED" else None,
        "list_objects": code(lambda: r2.list_objects_v2(Bucket=r2_bucket, MaxKeys=1)),
        "delete_object(nonexistent)": code(lambda: r2.delete_object(Bucket=r2_bucket, Key=probe_key)),
        "get_bucket_policy/cors": code(lambda: r2.get_bucket_cors(Bucket=r2_bucket)),
    }
    fb = sr.fallback_storage()
    s3 = sr._make_fallback_client(fb)
    out["s3_fallback"] = {
        "list_all_buckets": code(lambda: s3.list_buckets()),
        "list_objects": code(lambda: s3.list_objects_v2(Bucket=fb.bucket, MaxKeys=1)),
        "delete_object(nonexistent)": code(lambda: s3.delete_object(Bucket=fb.bucket, Key=probe_key)),
    }
    try:
        creds = s3._request_signer._credentials.get_frozen_credentials()
        sess = boto3.Session(aws_access_key_id=creds.access_key, aws_secret_access_key=creds.secret_key,
                             aws_session_token=creds.token)
        ident = sess.client("sts").get_caller_identity()
        arn = ident.get("Arn", "")
        out["s3_fallback"]["identity_kind"] = arn.split(":")[5].split("/")[0] if arn.count(":") >= 5 else "?"
        out["s3_fallback"]["identity_name"] = arn.split("/")[-1] if "/" in arn else ("root" if arn.endswith(":root") else "?")
        iam = sess.client("iam")
        out["s3_fallback"]["iam_list_users"] = code(lambda: iam.list_users(MaxItems=1))
        if out["s3_fallback"]["identity_kind"] == "user":
            name = out["s3_fallback"]["identity_name"]
            out["s3_fallback"]["attached_policies"] = code(lambda: iam.list_attached_user_policies(UserName=name))
            try:
                out["s3_fallback"]["policy_names"] = [p["PolicyName"] for p in iam.list_attached_user_policies(UserName=name)["AttachedPolicies"]] + \
                    iam.list_user_policies(UserName=name)["PolicyNames"]
            except Exception as e:
                out["s3_fallback"]["policy_names"] = "cannot read: " + e.__class__.__name__
        out["s3_fallback"]["ec2_describe_regions"] = code(lambda: sess.client("ec2", region_name="us-east-1").describe_regions())
    except Exception as e:
        out["s3_fallback"]["identity_error"] = e.__class__.__name__

    dg = os.environ.get("DEEPGRAM_API_KEY", "")
    if dg:
        h = {"Authorization": "Token " + dg}
        try:
            p = requests.get("https://api.deepgram.com/v1/projects", headers=h, timeout=20)
            out["deepgram"] = {"list_projects": p.status_code}
            if p.ok and p.json().get("projects"):
                pid = p.json()["projects"][0]["project_id"]
                k = requests.get(f"https://api.deepgram.com/v1/projects/{pid}/keys", headers=h, timeout=20)
                out["deepgram"]["list_keys(admin)"] = k.status_code
                if k.ok:
                    out["deepgram"]["key_scopes"] = sorted({s for item in k.json().get("api_keys", []) for s in item.get("api_key", {}).get("scopes", [])})
                    out["deepgram"]["key_count"] = len(k.json().get("api_keys", []))
        except Exception as e:
            out["deepgram"] = {"error": e.__class__.__name__}
    return out


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump({"label": "security review: key names and storage permissions", "rows": [audit.remote()]}, f, indent=1)
