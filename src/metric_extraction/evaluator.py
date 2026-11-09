"""Evaluación de configuraciones con salidas de reglas precalculadas (caché)."""
import time
from collections import defaultdict
from dataclasses import dataclass

from .gold import canon, same_metric, value_close
from .rules import family

THETAS = (1.0, 100.0, 1000.0)   # cota superior plausible de un valor aceptado


def prf1(true_positives, n_extracted, n_reference):
    """Devuelve (precisión, cobertura, F1)."""
    p = true_positives / n_extracted if n_extracted else 0.0
    r = true_positives / n_reference if n_reference else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


@dataclass(frozen=True)
class Configuration:
    rules: tuple[int, ...]   # índices de reglas activas, en orden de prioridad
    theta: float             # cota superior de validación


class Instance:
    """Fichas + salidas precalculadas de cada regla.

    cards: lista de dicts {id, tag, gold, text}. gold = gold_tuples(...).
    out[r][c] = [(clave_canónica, valor, índices_gold_acertados)].
    """

    def __init__(self, cards, catalog):
        self.cards, self.catalog = cards, catalog
        self.n_gold = [len(c["gold"]) for c in cards]
        self.cost = []
        self.out = []
        for rule in catalog:
            t0, per_card = time.perf_counter(), []
            for c in cards:
                hits = list(dict.fromkeys(rule.extract(c["text"])))
                per_card.append([
                    (canon(n), v, tuple(i for i, (al, rv) in enumerate(c["gold"])
                                        if same_metric(n, al) and value_close(v, rv)))
                    for n, v in hits])
            self.cost.append((time.perf_counter() - t0) / max(len(cards), 1))
            self.out.append(per_card)

    def subset(self, idx):
        """Instancia sobre un subconjunto de fichas, reutilizando las salidas ya calculadas."""
        new = Instance.__new__(Instance)
        new.catalog, new.cost = self.catalog, self.cost
        new.cards = [self.cards[i] for i in idx]
        new.n_gold = [self.n_gold[i] for i in idx]
        new.out = [[per_card[i] for i in idx] for per_card in self.out]
        return new

    def evaluate(self, config: Configuration, mask: bool = True):
        """Prioridad por métrica: para cada clave gana la primera regla que la produce."""
        per_tag = defaultdict(lambda: [0, 0, 0, 0])   # rec, n_gold, ok, n_hits
        for ci, card in enumerate(self.cards):
            owner, dup, rec, ok, nh = {}, set(), set(), 0, 0
            for r in config.rules:
                for key, v, matched in self.out[r][ci]:
                    if v > config.theta:
                        continue
                    # dueña de la métrica: la primera regla (en prioridad) que la produce;
                    # conserva todos sus valores, las siguientes reglas se descartan
                    if mask and owner.setdefault(key, r) != r:
                        continue
                    if not mask:                      # sin enmascarar: el mismo hit no cuenta dos veces
                        if (key, v) in dup:
                            continue
                        dup.add((key, v))
                    nh += 1
                    ok += bool(matched)
                    rec.update(matched)
            a = per_tag[card["tag"]]
            a[0] += len(rec); a[1] += self.n_gold[ci]; a[2] += ok; a[3] += nh
        tot = [sum(a[i] for a in per_tag.values()) for i in range(4)]
        P, R, F = prf1(tot[2], tot[3], tot[1])
        # recall usa gold recuperadas; prf1 espera TP sobre ambos lados, se calcula directo
        R = tot[0] / tot[1] if tot[1] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        tag_f1 = []
        for rec, ng, ok, nh in per_tag.values():
            p = ok / nh if nh else 0.0
            r = rec / ng if ng else 0.0
            tag_f1.append(2 * p * r / (p + r) if p + r else 0.0)
        return {"P": P, "R": R, "F1": F,
                "macro_F1": sum(tag_f1) / len(tag_f1) if tag_f1 else 0.0,
                "time": sum(self.cost[r] for r in config.rules)}