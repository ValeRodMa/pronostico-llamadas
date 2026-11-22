import math


def erlang_c(A, N):
    """Probabilidad de que una llamada tenga que esperar. A = trafico en erlangs, N = operadores."""
    if N <= A:
        return 1.0
    # calculo primero Erlang B de forma recursiva, asi no hay problemas con factoriales grandes
    B = 1.0
    for k in range(1, N + 1):
        B = A * B / (k + A * B)
    return N * B / (N - A * (1 - B))


def nivel_servicio(llamadas_hora, aht_seg, N, t_objetivo=20):
    # porcentaje de llamadas atendidas en menos de t_objetivo segundos
    A = llamadas_hora * aht_seg / 3600
    if N <= A:
        return 0.0
    return 1 - erlang_c(A, N) * math.exp(-(N - A) * t_objetivo / aht_seg)


def operadores_necesarios(llamadas_hora, aht_seg, objetivo=0.8, t_objetivo=20):
    if llamadas_hora <= 0:
        return 0
    A = llamadas_hora * aht_seg / 3600
    N = max(1, math.ceil(A))
    while nivel_servicio(llamadas_hora, aht_seg, N, t_objetivo) < objetivo:
        N += 1
    return N


def erlang_a(llamadas_hora, aht_seg, paciencia_seg, N):
    """Fraccion de llamadas que abandonan segun Erlang A (M/M/N + abandonos).

    Es Erlang C pero sin su supuesto mas irreal: aca cada persona en cola corta si su
    espera supera su paciencia (exponencial de media paciencia_seg). Lo resuelvo
    numericamente con la cadena de nacimiento y muerte: en el estado n (llamadas en el
    sistema) entran a tasa lambda y salen a tasa min(n, N) * mu por atenciones mas
    max(n - N, 0) * theta por abandonos. De la distribucion estacionaria sale la
    fraccion que abandona: tasa de abandonos sobre tasa de llegadas.
    """
    if llamadas_hora <= 0:
        return 0.0
    lam = llamadas_hora / 3600
    mu = 1 / aht_seg
    theta = 1 / paciencia_seg

    p = 1.0      # probabilidad del estado 0, sin normalizar (divido por la suma al final)
    suma = 1.0
    cola = 0.0   # largo de cola esperado, tambien sin normalizar
    n = 0
    while n < 200_000:  # tope por las dudas, con theta > 0 siempre converge antes
        n += 1
        p *= lam / (min(n, N) * mu + max(n - N, 0) * theta)
        suma += p
        if n > N:
            cola += (n - N) * p
            if p < 1e-12 * suma:
                break
    return theta * cola / (lam * suma)
