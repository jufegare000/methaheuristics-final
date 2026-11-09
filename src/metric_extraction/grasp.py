"""GRASP con búsqueda local y caché de evaluaciones (Entrega 3)."""
import random

from .evaluator import THETAS, Configuration


def neighbors(cfg: Configuration, n: int, K: int):
    """Vecindario: insertar, eliminar, intercambiar, reordenar y cambiar umbral."""
    r, th = cfg.rules, cfg.theta
    unused = [i for i in range(n) if i not in r]
    if len(r) < K:
        for i in unused:
            for p in range(len(r) + 1):
                yield Configuration(r[:p] + (i,) + r[p:], th)
    if len(r) > 1:
        for p in range(len(r)):
            yield Configuration(r[:p] + r[p + 1:], th)
    for p in range(len(r)):
        for i in unused:
            yield Configuration(r[:p] + (i,) + r[p + 1:], th)
    for a in range(len(r)):
        for b in range(a + 1, len(r)):
            s = list(r); s[a], s[b] = s[b], s[a]
            yield Configuration(tuple(s), th)
    for t in THETAS:
        if t != th:
            yield Configuration(r, t)


def construct(obj, n: int, K: int, alpha: float, rng: random.Random) -> Configuration:
    """Construcción voraz aleatorizada: cada paso elige al azar dentro de la lista restringida."""
    rules, th = (), rng.choice(THETAS)
    cur = float("-inf")
    for _ in range(K):
        cand = [(obj(Configuration(rules[:p] + (i,) + rules[p:], th)), rules[:p] + (i,) + rules[p:])
                for i in range(n) if i not in rules for p in range(len(rules) + 1)]
        fmax, fmin = max(c[0] for c in cand), min(c[0] for c in cand)
        if fmax <= cur:
            break
        rcl = [c for c in cand if c[0] >= fmax - alpha * (fmax - fmin)]
        cur, rules = rng.choice(rcl)
    return Configuration(rules, th)


def local_search(obj, cfg: Configuration, n: int, K: int, limit: int):
    """Mejor mejora hasta un óptimo local o agotar el presupuesto de evaluaciones."""
    f = obj(cfg)
    while obj.evals < limit:
        best, best_f = None, f
        for nb in neighbors(cfg, n, K):
            fn = obj(nb)
            if fn > best_f:
                best, best_f = nb, fn
        if best is None:
            break
        cfg, f = best, best_f
    return cfg, f


def grasp(obj, n: int, K: int, budget: int, alpha: float = 0.3, seed: int = 0):
    """Devuelve (mejor configuración, trazo [(evaluaciones, mejor f)])."""
    rng = random.Random(seed)
    start = obj.evals
    limit = start + budget
    best, best_f, trace = None, float("-inf"), []
    while obj.evals < limit:
        cfg = construct(obj, n, K, alpha, rng)
        cfg, f = local_search(obj, cfg, n, K, limit)
        if f > best_f:
            best, best_f = cfg, f
        trace.append((obj.evals - start, best_f))
    return best, trace