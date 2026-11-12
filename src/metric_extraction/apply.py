"""Aplicación de una configuración optimizada a fichas sin métricas.

Uso desde la terminal:
    python -m metric_extraction.apply \
        --config results/config_final.json \
        --csv data/HFCO2_2026.csv --readmes data/raw/readmes \
        --out data/processed/metrics_from_body.csv

La salida es una columna APARTE de ``performance_metrics``: los valores salen del
cuerpo del README y no son equivalentes a los del ``model-index`` (en fichas
AutoTrain ambos reportan mediciones distintas), así que no deben mezclarse.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .download import safe_name
from .evaluator import Configuration
from .evidence import read_body
from .gold import canon, gold_tuples
from .rules import build_catalog


# ------------------------------------------------------------ configuración
def save_config(cfg: Configuration, catalog, path, meta: dict | None = None) -> None:
    """Guarda la configuración con nombres de regla (legibles y estables)."""
    data = {"rules": [catalog[i].name for i in cfg.rules], "theta": cfg.theta,
            "meta": meta or {}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config(path):
    """Devuelve (reglas en orden de prioridad, theta, meta)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name = {r.name: r for r in build_catalog()}
    missing = [n for n in data["rules"] if n not in by_name]
    if missing:
        raise ValueError(f"reglas desconocidas en la configuración: {missing}")
    return [by_name[n] for n in data["rules"]], float(data["theta"]), data.get("meta", {})


# --------------------------------------------------------------- extracción
def extract_with_config(text: str, rules, theta: float) -> list[dict]:
    """Aplica la configuración al texto de una ficha (misma semántica que el evaluador).

    Para cada métrica normalizada, la primera regla en prioridad que la produce es su
    dueña y aporta todos sus valores; las siguientes reglas se descartan para ella.
    Los valores por encima de ``theta`` se rechazan.
    """
    owner, seen, out = {}, set(), []
    for rule in rules:
        for name, value in dict.fromkeys(rule.extract(text)):
            key = canon(name)
            if value > theta:
                continue
            if owner.setdefault(key, rule.name) != rule.name:
                continue
            if (key, value) in seen:
                continue
            seen.add((key, value))
            out.append({"metric": name, "key": key, "value": value, "rule": rule.name})
    return out


def is_empty(value) -> bool:
    """True si performance_metrics no trae valores numéricos utilizables."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    if str(value).strip() in ("", "[]", "{}", "null", "None", "nan"):
        return True
    try:
        return not gold_tuples(value)
    except Exception:       # JSON ilegible: no hay nada utilizable
        return True


def apply_to_models(df: pd.DataFrame, readmes_dir, rules, theta: float,
                    only_empty: bool = True) -> pd.DataFrame:
    """Una fila por modelo: estado, número de métricas y su JSON con procedencia."""
    readmes_dir = Path(readmes_dir)
    rows = []
    for _, r in df.iterrows():
        mid = r["modelId"]
        if only_empty and not is_empty(r.get("performance_metrics")):
            continue
        path = readmes_dir / f"{safe_name(mid)}.md"
        if not path.exists():
            rows.append((mid, r.get("pipeline_tag"), "sin_readme", 0, "[]", ""))
            continue
        hits = extract_with_config(read_body(path), rules, theta)
        estado = "con_metricas" if hits else "sin_metricas"
        rows.append((mid, r.get("pipeline_tag"), estado, len(hits),
                     json.dumps(hits, ensure_ascii=False),
                     ";".join(sorted({h["rule"] for h in hits}))))
    return pd.DataFrame(rows, columns=["modelId", "pipeline_tag", "estado", "n_metricas",
                                       "metrics_from_body", "reglas_usadas"])


def summary(res: pd.DataFrame) -> pd.DataFrame:
    """Cobertura por etiqueta de tarea."""
    g = res.groupby("pipeline_tag", dropna=False)
    s = g.agg(fichas=("modelId", "size"),
              con_metricas=("estado", lambda x: (x == "con_metricas").sum()),
              sin_readme=("estado", lambda x: (x == "sin_readme").sum()))
    s["cobertura"] = (s["con_metricas"] / s["fichas"]).round(3)
    return s.sort_values("fichas", ascending=False)


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--readmes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--all", action="store_true", help="incluir también modelos que ya tienen métricas")
    args = ap.parse_args(argv)

    rules, theta, meta = load_config(args.config)
    df = pd.read_csv(args.csv, sep=args.sep)
    res = apply_to_models(df, args.readmes, rules, theta, only_empty=not args.all)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"{len(res)} modelos procesados -> {args.out}")
    print(res["estado"].value_counts().to_string())
    print(summary(res).head(12).to_string())


if __name__ == "__main__":
    main()