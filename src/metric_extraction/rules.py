"""Catálogo de reglas: analizador (en línea, fila de tabla, columna de tabla) x familia de métrica."""
import re
from dataclasses import dataclass
from typing import Callable

from .evidence import column_table_hits, inline_hits, table_hits

FAMILIES = ("clf_basic", "clf_avg", "rank", "at_k", "sim", "rouge", "bleu", "reg", "lm")


def family(name: str) -> str:
    n = str(name).lower()
    if re.search(r"\b(ndcg|map|mrr)\b", n):
        return "rank"
    if "@" in n:
        return "at_k"
    if re.search(r"spearman|pearson", n):
        return "sim"
    if "rouge" in n:
        return "rouge"
    if re.search(r"bleu|meteor", n):
        return "bleu"
    if re.search(r"\b(mse|rmse|mae|r2)\b", n):
        return "reg"
    if re.search(r"\b(wer|cer|perplexity|ppl)\b", n):
        return "lm"
    if re.search(r"\b(macro|micro|weighted)\b", n):
        return "clf_avg"
    return "clf_basic"


PARSERS: dict[str, Callable[[str], list]] = {
    "inline": inline_hits,
    "row": table_hits,
    "column": column_table_hits,
}


@dataclass(frozen=True)
class Rule:
    parser: str
    family: str

    @property
    def name(self) -> str:
        return f"{self.parser}:{self.family}"

    def extract(self, text: str) -> list[tuple[str, float]]:
        return [(n, v) for n, v in PARSERS[self.parser](text)
                if "loss" not in n.lower() and family(n) == self.family]


def build_catalog() -> list[Rule]:
    """27 reglas: 3 analizadores x 9 subfamilias de métricas."""
    return [Rule(p, f) for p in PARSERS for f in FAMILIES]