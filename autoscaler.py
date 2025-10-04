import os, time, json, requests, redis

RUNPOD_API_KEY     = os.getenv("RUNPOD_API_KEY")
RUNPOD_TEMPLATE_ID = os.getenv("RUNPOD_TEMPLATE_ID")
REDIS_URL          = os.getenv("REDIS_URL")
QUEUE_NAME         = os.getenv("QUEUE_NAME", "default")
SCALE_MIN          = int(os.getenv("SCALE_MIN", "0"))
SCALE_MAX          = int(os.getenv("SCALE_MAX", "1"))
SCALE_UP_THRESHOLD = int(os.getenv("SCALE_UP_THRESHOLD", "1"))
SCALE_DOWN_IDLE_S  = int(os.getenv("SCALE_DOWN_IDLE_S", "300"))

RP_API = "https://api.runpod.ai/v2"
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

r = redis.from_url(REDIS_URL)

def list_pods():
    resp = requests.get(f"{RP_API}/pods", headers=HEADERS, timeout=20)
    resp.raise_for_status()
    pods = resp.json().get("data", [])
    return [p for p in pods if p.get("templateId") == RUNPOD_TEMPLATE_ID and p.get("desiredStatus") == "RUNNING"]

def start_pod():
    body = {"templateId": RUNPOD_TEMPLATE_ID}
    resp = requests.post(f"{RP_API}/pods", headers=HEADERS, data=json.dumps(body), timeout=30)
    resp.raise_for_status()
    pod = resp.json()
    print("scale_up: started pod", pod.get("id"))
    return pod

def stop_pod(pod_id):
    requests.post(f"{RP_API}/pods/{pod_id}/stop", headers=HEADERS, timeout=20)
    print("scale_down: stopped pod", pod_id)

def main_loop():
    idle_since = None
    while True:
        try:
            qlen = r.llen(QUEUE_NAME)
            pods = list_pods()
            running = len(pods)
            print({"queue": qlen, "running": running})

            if qlen > SCALE_UP_THRESHOLD and running < SCALE_MAX:
                start_pod()
                idle_since = None

            if qlen == 0 and running > SCALE_MIN:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > SCALE_DOWN_IDLE_S:
                    stop_pod(pods[0]["id"])
                    idle_since = None
            else:
                idle_since = None

        except Exception as e:
            print("autoscaler_error:", e)

        time.sleep(30)

if __name__ == "__main__":
    main_loop()
