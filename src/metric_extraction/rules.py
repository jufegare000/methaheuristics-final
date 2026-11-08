"""Catálogo de reglas de extracción (fuente, analizador, normalización)."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Rule:
    name: str
    extract: Callable[[str], list[tuple[str, float]]]  # texto -> [(métrica, valor)]


def build_catalog() -> list[Rule]:
    """Devuelve las reglas candidatas. Pendiente de implementar."""
    raise NotImplementedError
