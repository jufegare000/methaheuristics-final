"""Detección de evidencia de métricas en el cuerpo de un README."""
import re
from pathlib import Path

NAMES = ["accuracy", "acc", r"f1(?:[- ]score)?", "f-score", "precision", "recall",
         r"(?:roc[- ])?auc", "(?:sacre)?bleu", r"rouge[-_ ]?(?:lsum|[12l])?", "meteor", "wer", "cer",
         "perplexity", "ppl", r"exact[- ]match", "mrr", "ndcg", "spearman",
         "pearson", r"map(?=@\d)", "ap", "mcc", "mse", "rmse", "mae", "r2"]
STRONG = "(?:" + "|".join(NAMES) + ")"
QUALIFIER = r"(?:(?:macro|micro|weighted|cosine|dot|euclidean|manhattan|max)[\s_-]*)?"
SUFFIX = r"(?:@\d+)?(?:[_ ](?:cosine|dot|euclidean|manhattan|max))?"
SEP = r"[\s:=|*_]*(?:of|is|was)?[\s:=|*_]*"
INLINE = re.compile(
    rf"(?<![A-Za-z0-9])({QUALIFIER}(?:{STRONG}|loss){SUFFIX})(?![A-Za-z0-9]){SEP}(\d+(?:\.\d+)?)", re.I)
HEADER = re.compile(rf"(?<![A-Za-z0-9]){QUALIFIER}{STRONG}{SUFFIX}(?![A-Za-z0-9])", re.I)


def read_body(path) -> str:
    """Lee un README y quita el bloque YAML inicial."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)


def inline_hits(text: str) -> list[tuple[str, float]]:
    """[(métrica normalizada, valor)], conservando macro/micro/weighted."""
    return [(re.sub(r"[\s_-]+", " ", m.group(1).lower()), float(m.group(2)))
            for m in INLINE.finditer(text)]


def table_rows(text: str) -> int:
    """Filas numéricas bajo una tabla con encabezado de métrica."""
    lines, n = text.splitlines(), 0
    for i in range(len(lines) - 2):
        if (lines[i].count("|") >= 2 and HEADER.search(lines[i])
                and re.match(r"\s*\|?\s*:?-{2,}", lines[i + 1])):
            j = i + 2
            while j < len(lines) and lines[j].count("|") >= 2:
                n += bool(re.search(r"\d", lines[j]))
                j += 1
    return n


def signature(text: str) -> tuple[str, ...]:
    """Conjunto ordenado de métricas (sin loss) que aparecen: la 'firma' de la ficha."""
    return tuple(sorted({k for k, _ in inline_hits(text) if k != "loss"}))


ROW = re.compile(
    rf"^\s*\|\s*\**\s*(?P<name>{QUALIFIER}(?:{STRONG}|loss){SUFFIX})\s*\**\s*\|(?P<rest>.*)$",
    re.I | re.M)
CELL_NUM = re.compile(r"^\**\s*(\d+(?:\.\d+)?)\s*\**$")


def table_hits(text: str) -> list[tuple[str, float]]:
    """Filas '| metrica | v1 | v2 |': devuelve cada valor numérico de la fila."""
    hits = []
    for m in ROW.finditer(text):
        name = re.sub(r"[\s_-]+", " ", m.group("name").lower())
        for cell in m.group("rest").split("|"):
            c = CELL_NUM.match(cell.strip())
            if c:
                hits.append((name, float(c.group(1))))
    return hits


EVAL_PREFIX = r"(?:eval|validation|val|test)[\s_-]*"
HDR_CELL = re.compile(rf"^(?:{EVAL_PREFIX})?{QUALIFIER}{STRONG}{SUFFIX}$", re.I)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def column_table_hits(text: str, last_only: bool = True) -> list[tuple[str, float]]:
    """Tablas con las métricas como encabezado de columna (p. ej. 'Training results').

    last_only=True toma solo la última fila (la época final), que es la que
    suele coincidir con el model-index; False toma todas.
    """
    lines, hits, i = text.splitlines(), [], 0
    while i < len(lines) - 2:
        if lines[i].count("|") >= 2 and re.match(r"\s*\|?\s*:?-{2,}", lines[i + 1]):
            cols = {j: re.sub(r"[\s_-]+", " ",
                              re.sub(rf"^{EVAL_PREFIX}", "", h, flags=re.I).lower())
                    for j, h in enumerate(_cells(lines[i])) if HDR_CELL.match(h)}
            rows, k = [], i + 2
            while k < len(lines) and lines[k].count("|") >= 2:
                rows.append(_cells(lines[k]))
                k += 1
            if cols and rows:
                for row in (rows[-1:] if last_only else rows):
                    for j, name in cols.items():
                        if j < len(row) and (c := CELL_NUM.match(row[j])):
                            hits.append((name, float(c.group(1))))
            i = k
        else:
            i += 1
    return hits


def extract_hits(text: str) -> list[tuple[str, float]]:
    """Todas las coincidencias nombre-valor: en línea, filas de tabla y columnas."""
    return list(dict.fromkeys(inline_hits(text) + table_hits(text) + column_table_hits(text)))