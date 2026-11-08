"""Representación de la solución: reglas ordenadas + umbral."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    rules: tuple[int, ...]  # índices de reglas activas, en orden de prioridad
    threshold: float        # umbral de validación
