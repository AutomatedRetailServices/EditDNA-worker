import requests
import os

def process_urls(file_urls: list[str]):
    results = []
    for url in file_urls:
        # Example: just test download
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            results.append({"url": url, "status": "downloaded", "size": len(r.content)})
        else:
            results.append({"url": url, "status": f"error {r.status_code}"})
    return results
