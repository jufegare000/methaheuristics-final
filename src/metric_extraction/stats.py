"""Pruebas estadísticas no paramétricas para ajustar y comparar algoritmos (Entrega 4)."""
import itertools

import numpy as np
from scipy import stats


def holm(pvals):
    """Corrección de Holm-Bonferroni; devuelve p ajustados en el orden original."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cliffs_delta(x, y):
    """Tamaño de efecto sin supuestos de distribución: P(x>y) - P(x<y), en [-1, 1]."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def bootstrap_ci(d, stat=np.median, B=10000, level=0.95, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    s = [stat(rng.choice(d, len(d), replace=True)) for _ in range(B)]
    lo, hi = np.percentile(s, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def kruskal(groups: dict):
    """Kruskal-Wallis entre configuraciones independientes (ajuste de parámetros)."""
    h, p = stats.kruskal(*groups.values())
    return {"H": float(h), "p": float(p), "k": len(groups), "n": sum(len(v) for v in groups.values())}


def friedman(table):
    """Friedman: table[bloque][método]. Devuelve estadístico, p y rango medio por método."""
    a = np.asarray(table, float)
    chi, p = stats.friedmanchisquare(*a.T)
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, a)   # rango 1 = mejor
    return {"chi2": float(chi), "p": float(p), "mean_rank": ranks.mean(0).tolist(), "blocks": a.shape[0]}


def pairwise_wilcoxon(data: dict):
    """Wilcoxon pareado entre todos los pares de métodos, con Holm y efecto.

    data[método] = vector de aptitudes alineado por bloque (semilla o pliegue).
    """
    names = list(data)
    rows = []
    for a, b in itertools.combinations(names, 2):
        x, y = np.asarray(data[a], float), np.asarray(data[b], float)
        d = x - y
        if np.allclose(d, 0):
            p, w = 1.0, 0.0
        else:
            res = stats.wilcoxon(x, y, zero_method="wilcox")
            p, w = float(res.pvalue), float(res.statistic)
        lo, hi = bootstrap_ci(d) if len(d) > 1 else (np.nan, np.nan)
        rows.append({"A": a, "B": b, "mediana_dif": float(np.median(d)), "IC95_lo": lo, "IC95_hi": hi,
                     "W": w, "p": p, "cliffs_delta": cliffs_delta(x, y),
                     "A_gana": int((d > 0).sum()), "B_gana": int((d < 0).sum()), "empates": int((d == 0).sum())})
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
    return rows


def pairwise_mannwhitney(data: dict):
    """Mann-Whitney U entre réplicas INDEPENDIENTES (semillas distintas) con Holm y efecto de Cliff.

    Es la prueba adecuada para comparar métodos estocásticos en una misma instancia: la semilla
    i de un método no está emparejada con la semilla i de otro. Wilcoxon pareado se reserva para
    bloques reales (pliegues de validación cruzada).
    """
    rows = []
    for a, b in itertools.combinations(list(data), 2):
        x, y = np.asarray(data[a], float), np.asarray(data[b], float)
        if np.ptp(np.concatenate([x, y])) == 0:
            u, p = len(x) * len(y) / 2, 1.0
        else:
            res = stats.mannwhitneyu(x, y, alternative="two-sided")
            u, p = float(res.statistic), float(res.pvalue)
        rows.append({"A": a, "B": b, "mediana_A": float(np.median(x)), "mediana_B": float(np.median(y)),
                     "U": u, "p": p, "cliffs_delta": cliffs_delta(x, y)})
    for r, q in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(q)
    return rows