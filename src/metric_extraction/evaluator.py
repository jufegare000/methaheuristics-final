"""Evaluación de una configuración: precisión, cobertura y F1."""


def prf1(true_positives: int, n_extracted: int, n_reference: int):
    """Devuelve (precisión, cobertura, F1)."""
    p = true_positives / n_extracted if n_extracted else 0.0
    r = true_positives / n_reference if n_reference else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(config, outputs, reference):
    """Combina las salidas precalculadas de las reglas según la prioridad.

    outputs: salidas de cada regla sobre cada ficha, calculadas una sola vez.
    Pendiente de implementar.
    """
    raise NotImplementedError
