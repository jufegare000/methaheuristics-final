"""Experimentos de la Entrega 4: GA vs GRASP vs líneas base, ajuste del GA y validación cruzada."""
import itertools
import time

import numpy as np
import pandas as pd

from .baselines import Objective, greedy_forward, random_search
from .genetic import GAParams, genetic
from .grasp import grasp

METHODS = ("Voraz", "Aleatoria", "GRASP", "GA")


def run_method(method, inst, K, budget, seed, tmax=None, ga=GAParams(), alpha=0.3, pmin=0.90):
    """Corre un método con caché nueva. Devuelve (config, objetivo, trazo, evaluaciones, segundos)."""
    n = len(inst.catalog)
    obj = Objective(inst, pmin=pmin, tmax=tmax)
    t0 = time.perf_counter()
    trace = []
    if method == "Voraz":
        cfg = greedy_forward(obj, n, K)
    elif method == "Aleatoria":
        cfg = random_search(obj, n, K, budget, seed=seed)
    elif method == "GRASP":
        cfg, trace = grasp(obj, n, K, budget, alpha=alpha, seed=seed)
    elif method == "GA":
        cfg, trace, _ = genetic(obj, n, K, budget, ga, seed=seed)
    else:
        raise ValueError(method)
    return cfg, obj(cfg), trace, obj.evals, time.perf_counter() - t0


def _row(method, cfg, f, evals, secs, inst, extra):
    m = inst.evaluate(cfg)
    return {"method": method, "f_train": f, "P": m["P"], "R": m["R"], "macro_F1": m["macro_F1"],
            "time_ms": m["time"] * 1000, "evals": evals, "secs": secs,
            "rules": " > ".join(inst.catalog[i].name for i in cfg.rules), "k": len(cfg.rules),
            "theta": cfg.theta, **extra}


def compare(inst, K, budget, seeds, tmax=None, ga=GAParams(), alpha=0.3, methods=METHODS, verbose=True):
    """Una fila por (método, semilla). El voraz es determinista: se corre una vez y se replica."""
    rows, traces = [], {}
    for method in methods:
        cache_row = None
        for s in seeds:
            if method == "Voraz" and cache_row is not None:
                rows.append({**cache_row, "seed": s}); continue
            cfg, f, tr, ev, secs = run_method(method, inst, K, budget, s, tmax, ga, alpha)
            r = _row(method, cfg, f, ev, secs, inst, {"K": K, "seed": s})
            rows.append(r); traces[(method, s)] = tr
            if method == "Voraz":
                cache_row = r
        if verbose:
            d = [r["macro_F1"] for r in rows if r["method"] == method]
            print(f"K={K} {method:10s} mediana={np.median(d):.4f} media={np.mean(d):.4f}")
    return pd.DataFrame(rows), traces


def tune_ga(inst, K, budget, seeds, tmax=None, grid=None, verbose=True):
    """Rejilla del GA: una fila por (configuración, semilla). Usar semillas distintas a las finales."""
    grid = grid or {"N": (30, 50, 100), "pc": (0.7, 0.9), "pm": (0.1, 0.2, 0.4)}
    keys = list(grid)
    rows = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(keys, vals))
        ga = GAParams(**p)
        for s in seeds:
            cfg, f, _, ev, secs = run_method("GA", inst, K, budget, s, tmax, ga)
            rows.append({**p, "seed": s, "f_train": f, "evals": ev, "secs": secs})
        if verbose:
            d = [r["f_train"] for r in rows[-len(seeds):]]
            print(p, f"mediana={np.median(d):.4f}")
    return pd.DataFrame(rows)


def group_folds(cards, k=5):
    """Pliegues por organización (prefijo del id), equilibrados por número de fichas."""
    size = {}
    for i, c in enumerate(cards):
        size.setdefault(c["id"].split("/")[0], []).append(i)
    folds = [[] for _ in range(k)]
    for org, idx in sorted(size.items(), key=lambda kv: -len(kv[1])):
        min(folds, key=len).extend(idx)
    return folds


def cross_validate(inst, K, budget, seeds, folds, tmax=None, ga=GAParams(), alpha=0.3,
                   methods=METHODS, verbose=True):
    """Optimiza en k-1 pliegues y mide en el pliegue retenido. Una fila por (pliegue, método, semilla)."""
    rows = []
    for fi, test_idx in enumerate(folds):
        if not test_idx:
            continue
        train_idx = [i for j, f in enumerate(folds) if j != fi for i in f]
        tr, te = inst.subset(train_idx), inst.subset(test_idx)
        for method in methods:
            for s in (seeds[:1] if method == "Voraz" else seeds):
                cfg, f, _, ev, secs = run_method(method, tr, K, budget, s, tmax, ga, alpha)
                m = te.evaluate(cfg)
                rows.append({"fold": fi, "method": method, "seed": s, "f_train": f,
                             "test_macro_F1": m["macro_F1"], "test_P": m["P"], "test_R": m["R"],
                             "n_test": len(test_idx), "evals": ev, "k": len(cfg.rules)})
        if verbose:
            print(f"pliegue {fi} listo ({len(test_idx)} fichas de prueba)")
    return pd.DataFrame(rows)


def mean_trace(traces, method, grid):
    """Mejor aptitud promedio (sobre semillas) en una malla de evaluaciones, para la figura de convergencia."""
    curves = []
    for (m, _), tr in traces.items():
        if m != method or not tr:
            continue
        xs, ys = zip(*tr)
        curves.append(np.interp(grid, xs, ys, left=np.nan))
    return np.nanmean(np.vstack(curves), axis=0) if curves else np.full(len(grid), np.nan)