"""Función objetivo y líneas base: voraz, búsqueda aleatoria y enumeración exacta."""
import itertools
import random

from .evaluator import THETAS, Configuration


class Objective:
    """Macro-F1 por etiqueta con restricciones duras.

    Una configuración que viola la precisión mínima o el tiempo máximo vale siempre
    menos que cualquiera factible: f = -1 - violación (la violación orienta la búsqueda
    hacia la región factible). Guarda en caché cada configuración evaluada (memoria)
    y cuenta las evaluaciones reales.
    """

    def __init__(self, inst, pmin: float = 0.90, tmax: float | None = None):
        self.inst, self.pmin, self.tmax = inst, pmin, tmax
        self.cache: dict = {}
        self.evals = 0          # evaluaciones reales (sin contar aciertos de caché)
        self.calls = 0

    def metrics(self, cfg: Configuration) -> dict:
        self.calls += 1
        if cfg not in self.cache:
            self.evals += 1
            self.cache[cfg] = self.inst.evaluate(cfg)
        return self.cache[cfg]

    def violation(self, cfg: Configuration) -> float:
        m = self.metrics(cfg)
        v = max(0.0, self.pmin - m["P"])
        if self.tmax is not None:
            v += max(0.0, m["time"] / self.tmax - 1.0)
        return v

    def feasible(self, cfg: Configuration) -> bool:
        return self.violation(cfg) == 0.0

    def __call__(self, cfg: Configuration) -> float:
        v = self.violation(cfg)
        return self.metrics(cfg)["macro_F1"] if v == 0.0 else -1.0 - v


def _best_theta(obj, rules):
    return max(THETAS, key=lambda th: obj(Configuration(tuple(rules), th)))


def greedy_individual(obj, catalog_size: int, K: int):
    """Línea base voraz de la Entrega 1: reglas ordenadas por su F1 individual, las K mejores."""
    score = [obj(Configuration((i,), 100.0)) for i in range(catalog_size)]
    rules = tuple(sorted(range(catalog_size), key=lambda i: -score[i])[:K])
    th = _best_theta(obj, rules)
    return Configuration(rules, th)


def greedy_forward(obj, catalog_size: int, K: int):
    """Voraz constructivo: en cada paso inserta la regla (y posición) que más mejora."""
    rules, th = (), 100.0
    best = float("-inf")
    for _ in range(K):
        cand = [(obj(Configuration(r[:p] + (i,) + r[p:], th)), r[:p] + (i,) + r[p:])
                for r in [rules] for i in range(catalog_size) if i not in r
                for p in range(len(r) + 1)]
        f, new = max(cand)
        if f <= best:
            break
        best, rules = f, new
    return Configuration(rules, _best_theta(obj, rules))


def random_search(obj, catalog_size: int, K: int, n_evals: int, seed: int = 0):
    rng = random.Random(seed)
    best, best_f = None, float("-inf")
    for _ in range(n_evals):
        k = rng.randint(1, K)
        cfg = Configuration(tuple(rng.sample(range(catalog_size), k)), rng.choice(THETAS))
        f = obj(cfg)
        if f > best_f:
            best, best_f = cfg, f
    return best


def exact_enumeration(obj, rule_subset, K: int):
    """Enumera todas las ordenaciones de hasta K reglas del subconjunto, para cada umbral."""
    best, best_f = None, float("-inf")
    for k in range(1, K + 1):
        for perm in itertools.permutations(rule_subset, k):
            for th in THETAS:
                cfg = Configuration(perm, th)
                f = obj(cfg)
                if f > best_f:
                    best, best_f = cfg, f
    return best