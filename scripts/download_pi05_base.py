"""Download the public OpenPI pi05_base checkpoint to shared storage."""
import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

PREFIX = "checkpoints/pi05_base/"
ROOT = Path("/mnt/cfs/9wt59p/genalyu/robodojo-posttrain/models/pi05-base")
API = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o"
BUCKET = "https://storage.googleapis.com/openpi-assets/"


def list_objects():
    items = []
    token = None
    while True:
        params = {"prefix": PREFIX, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        response = requests.get(API, params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        items.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return items


def valid(path, item):
    if not path.is_file() or path.stat().st_size != int(item["size"]):
        return False
    expected = item.get("md5Hash")
    if not expected:
        return True
    h = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode() == expected


def download(item):
    rel = item["name"].removeprefix(PREFIX)
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid(path, item):
        return rel, "cached"
    part = path.with_name(path.name + ".part")
    total = int(item["size"])
    url = BUCKET + quote(item["name"], safe="/")
    for attempt in range(12):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with part.open(mode) as target:
                    for chunk in response.iter_content(8 << 20):
                        if chunk:
                            target.write(chunk)
            if part.stat().st_size == total:
                part.replace(path)
                if valid(path, item):
                    return rel, "downloaded"
                path.replace(part)
            time.sleep(min(2 ** attempt, 30))
        except Exception as exc:
            print(f"retry {rel}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed {rel}: {part.stat().st_size if part.exists() else 0}/{total}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    items = list_objects()
    print(f"files={len(items)} bytes={sum(int(x['size']) for x in items)}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, item) for item in items]
        for future in as_completed(futures):
            print(*future.result(), flush=True)
    (ROOT / "source_manifest.json").write_text(json.dumps([
        {"name": x["name"], "size": x["size"], "generation": x.get("generation"),
         "md5Hash": x.get("md5Hash")} for x in items
    ], indent=2))
    print("complete", flush=True)


if __name__ == "__main__":
    main()
