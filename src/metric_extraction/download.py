import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

URL = "https://huggingface.co/{model_id}/raw/main/README.md"
RETRY_STATUS = {429, 500, 502, 503, 504}
PERMANENT = [401, 403, 404]   # no tiene sentido reintentarlos


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _wait_seconds(resp, attempt: int) -> float:
    try:
        return float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return 2.0 ** attempt


def _download_one(model_id, out_dir, headers, max_retries=5):
    path = out_dir / f"{safe_name(model_id)}.md"
    if path.exists():
        return model_id, "cached", 200
    status = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(URL.format(model_id=model_id),
                                headers=headers, timeout=30)
        except requests.RequestException:
            status = -1                       # error de red
            time.sleep(2.0 ** attempt + random.random())
            continue
        status = resp.status_code
        if resp.ok:
            tmp = path.with_suffix(".tmp")    # evita archivos truncados
            tmp.write_text(resp.text, encoding="utf-8")
            os.replace(tmp, path)
            return model_id, "ok", status
        if status in RETRY_STATUS:
            time.sleep(_wait_seconds(resp, attempt) + random.random())
            continue
        break
    return model_id, "fail", status


def download_readmes(model_ids, out_dir, token=None, workers=4,
                     retry_permanent=False) -> pd.DataFrame:
    """Descarga los README que falten. Se puede interrumpir y volver a correr.

    Guarda ../readmes_status.csv junto a out_dir con el resultado por modelo.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir.parent / "readmes_status.csv"
    cols = ["model_id", "result", "http"]
    prev = (pd.read_csv(status_path) if status_path.exists()
            else pd.DataFrame(columns=cols))

    skip = set()
    if not retry_permanent:
        skip = set(prev.loc[prev["http"].isin(PERMANENT), "model_id"])
    todo = [m for m in dict.fromkeys(model_ids) if m not in skip]
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    new = []
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(_download_one, m, out_dir, headers) for m in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            new.append(fut.result())
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}")
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        print("Interrumpido; los archivos ya bajados se conservan.")
    finally:
        ex.shutdown(wait=True)
        status = (pd.concat([prev, pd.DataFrame(new, columns=cols)])
                  .drop_duplicates("model_id", keep="last"))
        status.to_csv(status_path, index=False)
    return status