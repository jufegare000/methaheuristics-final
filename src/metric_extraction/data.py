"""Carga de datos, referencia (model-index) y descarga de model cards."""
import json
import os
import re

import pandas as pd
import requests


def load_gold(path: str) -> pd.DataFrame:
    """Lee el CSV Gold exportado por el ETL (delimitador ';')."""
    return pd.read_csv(path, sep=";")


def ref_values(performance_metrics) -> list[float] | None:
    """Valores numéricos del model-index; None si no hay ninguno."""
    try:
        data = json.loads(performance_metrics)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    vals = [
        float(m["value"])
        for e in data if isinstance(e, dict)
        for r in e.get("results", [])
        for m in r.get("metrics", [])
        if isinstance(m.get("value"), (int, float))
    ]
    return vals or None


def fetch_readme(model_id: str, token: str | None = None) -> str | None:
    """Descarga el README crudo; None si falla."""
    token = token or os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(
        f"https://huggingface.co/{model_id}/raw/main/README.md",
        headers=headers, timeout=30,
    )
    return resp.text if resp.ok else None


def strip_front_matter(text: str) -> str:
    """Quita el bloque YAML inicial, donde vive el model-index."""
    return re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)


def value_in_text(value: float, text: str) -> bool:
    """True si el valor aparece en el texto (fracción o porcentaje)."""
    cands = {f"{value:.{d}f}" for d in (2, 3, 4)}
    cands |= {str(round(value * 100, d)) for d in (1, 2)}
    return any(
        re.search(rf"(?<![\d.]){re.escape(c)}(?!\d)", text) for c in cands
    )
