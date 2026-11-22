"""Algoritmo genético con memoria (Entrega 4), según el diseño de la Entrega 2.

Representación, función de aptitud, caché y vecindario son los mismos que usa GRASP,
de modo que ambos métodos se comparan con el mismo presupuesto de evaluaciones reales.
"""
import random
from dataclasses import dataclass

from .evaluator import THETAS, Configuration
from .grasp import construct


@dataclass(frozen=True)
class GAParams:
    N: int = 50              # tamaño de población
    pc: float = 0.9          # probabilidad de cruce
    pm: float = 0.2          # probabilidad de mutación
    k_t: int = 3             # tamaño del torneo
    elite: int = 2           # elitismo
    rho: float = 0.2         # fracción de la población inicial construida con GRASP
    alpha: float = 0.3       # alfa de la construcción aleatorizada
    G: int = 15              # generaciones sin mejora para detener
    seed_share: float = 0.25  # tope del presupuesto que puede gastar la siembra GRASP


def random_individual(n: int, K: int, rng: random.Random) -> Configuration:
    k = rng.randint(1, K)
    return Configuration(tuple(rng.sample(range(n), k)), rng.choice(THETAS))


def random_move(cfg: Configuration, n: int, K: int, rng: random.Random) -> Configuration:
    """Un movimiento aleatorio del vecindario (insertar, eliminar, reemplazar, intercambiar, cambiar θ)."""
    r, th = cfg.rules, cfg.theta
    unused = [i for i in range(n) if i not in r]
    moves = ["theta"]
    if len(r) < K and unused:
        moves.append("insert")
    if len(r) > 1:
        moves.append("remove")
    if r and unused:
        moves.append("replace")
    if len(r) > 1:
        moves.append("swap")
    m = rng.choice(moves)
    if m == "insert":
        p = rng.randint(0, len(r))
        return Configuration(r[:p] + (rng.choice(unused),) + r[p:], th)
    if m == "remove":
        p = rng.randrange(len(r))
        return Configuration(r[:p] + r[p + 1:], th)
    if m == "replace":
        p = rng.randrange(len(r))
        return Configuration(r[:p] + (rng.choice(unused),) + r[p + 1:], th)
    if m == "swap":
        a, b = rng.sample(range(len(r)), 2)
        s = list(r); s[a], s[b] = s[b], s[a]
        return Configuration(tuple(s), th)
    return Configuration(r, rng.choice([t for t in THETAS if t != th]))


def order_crossover(a: Configuration, b: Configuration, K: int, rng: random.Random) -> Configuration:
    """Prefijo de a (corte aleatorio) completado con las reglas de b, en el orden de b."""
    cut = rng.randint(1, len(a.rules))
    child = list(a.rules[:cut])
    for g in b.rules:
        if len(child) >= K:
            break
        if g not in child:
            child.append(g)
    return Configuration(tuple(child), rng.choice((a.theta, b.theta)))


def tournament(pop, fit, k: int, rng: random.Random) -> Configuration:
    return max(rng.sample(pop, min(k, len(pop))), key=lambda c: fit[c])


def genetic(obj, n: int, K: int, budget: int, params: GAParams = GAParams(), seed: int = 0):
    """Devuelve (mejor configuración, trazo [(evaluaciones reales, mejor f)], generaciones)."""
    rng = random.Random(seed)
    P = params
    start = obj.evals
    limit = start + budget
    used = lambda: obj.evals - start

    # --- población inicial: rho sembrada con GRASP (con tope de presupuesto) + resto aleatoria
    pop, seen = [], set()
    n_seed = round(P.rho * P.N)
    for _ in range(n_seed):
        if used() >= P.seed_share * budget:
            break
        c = construct(obj, n, K, P.alpha, rng)
        if c not in seen:
            seen.add(c); pop.append(c)
    guard = 0
    while len(pop) < P.N and guard < 50 * P.N:
        guard += 1
        c = random_individual(n, K, rng)
        if c not in seen:
            seen.add(c); pop.append(c)

    fit, trace = {}, []
    best, best_f = None, float("-inf")

    def evaluate(c):
        nonlocal best, best_f
        if c not in fit:
            fit[c] = obj(c)
        if fit[c] > best_f:
            best, best_f = c, fit[c]
        return fit[c]

    for c in pop:
        if used() >= budget and c not in obj.cache:   # lo ya evaluado no cuesta
            break
        evaluate(c)
        trace.append((used(), best_f))
    pop = [c for c in pop if c in fit]
    if best is None:                       # presupuesto agotado por la siembra: al menos un individuo
        evaluate(pop[0] if pop else random_individual(n, K, rng))

    gens, stall, prev = 0, 0, best_f
    while used() < budget and stall < P.G and len(pop) >= 2:
        gens += 1
        ranked = sorted(pop, key=lambda c: -fit[c])
        new = ranked[:P.elite]
        members = set(new)
        attempts = 0
        while len(new) < P.N and used() < budget and attempts < 20 * P.N:
            attempts += 1
            a, b = tournament(pop, fit, P.k_t, rng), tournament(pop, fit, P.k_t, rng)
            h = order_crossover(a, b, K, rng) if rng.random() < P.pc else a
            if rng.random() < P.pm:
                h = random_move(h, n, K, rng)
            tries = 0
            while h in members and tries < 5:       # sin duplicados: mutar de nuevo
                h = random_move(h, n, K, rng); tries += 1
            if h in members:
                continue
            evaluate(h)
            trace.append((used(), best_f))
            members.add(h); new.append(h)
        pop = new
        stall = stall + 1 if best_f <= prev + 1e-12 else 0
        prev = best_f
    return best, trace, gens