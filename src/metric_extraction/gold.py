"""Tuplas de referencia (model-index) y comparación con lo extraído."""
import json
import re


# Campos que aparecen en el model-index pero no miden desempeño.
NON_METRIC = re.compile(
    r"active dims|sparsity ratio|flops|gen len|runtime|samples per second|steps per second",
    re.I)


def norm(name) -> str:
    """Normaliza un nombre de métrica: minúsculas y separadores a espacio."""
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower())


def gold_tuples(performance_metrics, exclude_non_metrics: bool = True) -> list[tuple[frozenset, float]]:
    """[(alias, valor)] únicos del model-index.

    alias = {type, name} normalizados; así 'cosine_accuracy' y 'Cosine Accuracy'
    se consideran la misma métrica. Se colapsan los pares repetidos entre datasets.
    """
    try:
        data = json.loads(performance_metrics)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    seen, out = set(), []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for res in entry.get("results") or []:
            for m in res.get("metrics") or []:
                v = m.get("value")
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                aliases = frozenset(norm(x) for x in (m.get("type"), m.get("name")) if x)
                key = (aliases, float(v))
                if exclude_non_metrics and any(NON_METRIC.search(a) for a in aliases):
                    continue
                if aliases and key not in seen:
                    seen.add(key)
                    out.append(key)
    return out


def value_close(x: float, ref: float, rel: float = 0.01) -> bool:
    """True si x iguala a ref dentro de una tolerancia relativa (también en %)."""
    tol = rel * abs(ref)
    return any(abs(c - ref) <= tol for c in (x, x / 100, x * 100))


def score_hits(hits, gold, rel: float = 0.01):
    """hits: [(nombre_normalizado, valor)].

    Devuelve (gold_recuperadas, n_gold, hits_correctas, n_hits):
    recall = gold_recuperadas / n_gold, precisión = hits_correctas / n_hits.
    """
    hits = list(dict.fromkeys(hits))
    rec = sum(any(n in al and value_close(v, rv, rel) for n, v in hits) for al, rv in gold)
    ok = sum(any(n in al and value_close(v, rv, rel) for al, rv in gold) for n, v in hits)
    return rec, len(gold), ok, len(hits)