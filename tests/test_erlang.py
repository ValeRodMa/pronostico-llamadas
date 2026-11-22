import pytest

from src.erlang import erlang_a, erlang_c, nivel_servicio, operadores_necesarios


def test_ejemplo_120_llamadas():
    # 120 llamadas por hora de 3 minutos = 6 erlangs
    assert nivel_servicio(120, 180, 7) == pytest.approx(0.451, abs=0.01)
    assert nivel_servicio(120, 180, 8) == pytest.approx(0.714, abs=0.01)
    assert nivel_servicio(120, 180, 9) == pytest.approx(0.860, abs=0.01)
    assert operadores_necesarios(120, 180, objetivo=0.8) == 9


def test_con_pocos_operadores_todos_esperan():
    assert erlang_c(6, 6) == 1.0
    assert nivel_servicio(120, 180, 5) == 0.0


def test_sin_llamadas_no_hace_falta_nadie():
    assert operadores_necesarios(0, 180) == 0


def test_erlang_a_en_sobrecarga_se_pierde_lo_que_no_entra():
    # 120 llamadas/h de 3 min son 6 erlangs: con 3 operadores solo se puede atender
    # la mitad del trafico. con paciencia larga la cola amortigua las rachas y el
    # abandono queda en el limite justo (la mitad); con paciencia corta la cola se
    # vacia seguido, los operadores quedan libres a ratos y se pierde algo mas
    assert erlang_a(120, 180, 600, 3) == pytest.approx(0.5, abs=0.02)
    assert 0.5 < erlang_a(120, 180, 60, 3) < 0.6


def test_erlang_a_mas_operadores_menos_abandono():
    con_5 = erlang_a(120, 180, 60, 5)
    con_7 = erlang_a(120, 180, 60, 7)
    con_9 = erlang_a(120, 180, 60, 9)
    assert con_9 < con_7 < con_5
    # con la dotacion que pide erlang C (9) se pierde poco aun con gente impaciente
    assert con_9 < 0.05


def test_erlang_a_casos_borde():
    assert erlang_a(120, 180, 60, 0) == pytest.approx(1.0, abs=1e-6)  # sin operadores se pierde todo
    assert erlang_a(0, 180, 60, 5) == 0.0  # sin llamadas no se pierde nada
