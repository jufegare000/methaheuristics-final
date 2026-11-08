from metric_extraction.evaluator import prf1


def test_prf1_perfect():
    assert prf1(10, 10, 10) == (1.0, 1.0, 1.0)


def test_prf1_empty():
    assert prf1(0, 0, 10) == (0.0, 0.0, 0.0)
