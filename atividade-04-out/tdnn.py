"""Perceptron multicamadas com topologia Time Delay (TDNN) — previsão do preço de uma mercadoria.

Entradas x(t-1) … x(t-p), uma saída f(t). Backpropagation com momentum, treinamento online,
bias como entrada fixa x0 = -1 em cada camada, parada quando |EQM(época) - EQM(época-1)| <= ε.
Na validação (t = 101..120) as entradas são sempre os valores reais da série (previsão um passo à frente).
"""
import csv
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DADOS = BASE / "dados"
TAXA_APRENDIZADO = 0.1
MOMENTUM = 0.8
PRECISAO = 0.5e-6
NUM_TREINAMENTOS = 3
REDES = {"Rede 1": (5, 10), "Rede 2": (10, 15), "Rede 3": (15, 25)}  # (p, N1)


def carregar(nome):
    with open(DADOS / nome, encoding="utf-8") as f:
        linhas = list(csv.reader(f))[1:]
    return [(int(t.removeprefix("t=")), float(v)) for t, v in linhas if t]


def janelas(serie, p, alvos):
    """Para cada índice t em `alvos`, entrada [-1, x(t-1), …, x(t-p)] e desejado f(t)."""
    x = np.array([[-1.0] + [serie[t - k] for k in range(1, p + 1)] for t in alvos])
    d = np.array([serie[t] for t in alvos])
    return x, d


def sigmoide(u):
    return 1.0 / (1.0 + np.exp(-u))


def saidas(w1, w2, x):
    y1 = sigmoide(x @ w1.T)
    y1b = np.hstack([-np.ones((len(x), 1)), y1])
    return sigmoide(y1b @ w2)


def eqm(w1, w2, x, d):
    """EQM = (1/p) · Σ_k ½ · (d(k) − y(k))²"""
    return float(np.mean(0.5 * (d - saidas(w1, w2, x)) ** 2))


def treinar(x, d, w1_ini, w2_ini):
    w1, w2 = w1_ini.copy(), w2_ini.copy()
    dw1_ant, dw2_ant = np.zeros_like(w1), np.zeros_like(w2)
    historico = [eqm(w1, w2, x, d)]  # EQM na época 0 (pesos iniciais)
    inicio = time.perf_counter()
    while True:
        for xk, dk in zip(x, d):
            y1 = sigmoide(w1 @ xk)
            y1b = np.concatenate(([-1.0], y1))
            y2 = sigmoide(w2 @ y1b)
            delta2 = (dk - y2) * y2 * (1 - y2)
            delta1 = w2[1:] * delta2 * y1 * (1 - y1)
            dw2 = TAXA_APRENDIZADO * delta2 * y1b + MOMENTUM * dw2_ant
            dw1 = TAXA_APRENDIZADO * np.outer(delta1, xk) + MOMENTUM * dw1_ant
            w2 += dw2
            w1 += dw1
            dw1_ant, dw2_ant = dw1, dw2
        historico.append(eqm(w1, w2, x, d))
        if abs(historico[-1] - historico[-2]) <= PRECISAO:
            break
    return w1, w2, historico, time.perf_counter() - inicio


def erro_relativo(d, y):
    """Erro relativo percentual de cada amostra: |d − y| / |d| · 100"""
    return np.abs(d - y) / np.abs(d) * 100


def plotar_eqm(melhores, caminho):
    fig, eixos = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (rede, (t, r)) in zip(eixos, melhores.items()):
        ax.plot(range(len(r["historico"])), r["historico"], color="#1f77b4")
        ax.set_title(f"{rede} (T{t}) — {len(r['historico']) - 1} épocas")
        ax.set_xlabel("Época")
        ax.set_ylabel("EQM")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("TDNN — Erro quadrático médio por época (melhor treinamento de cada rede)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def plotar_previsao(melhores, tempos, desejados, caminho):
    fig, eixos = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (rede, (t, r)) in zip(eixos, melhores.items()):
        ax.plot(tempos, desejados, "o-", color="#1f77b4", label="Desejado f(t)")
        ax.plot(tempos, r["y_teste"], "s--", color="#d62728", label="Estimado pela rede")
        ax.set_title(f"{rede} (T{t}) — erro rel. médio {r['erro_medio']:.2f}%")
        ax.set_xlabel("t")
        ax.set_xticks(tempos[::2])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    eixos[0].set_ylabel("f(t)")
    fig.suptitle("TDNN — Valores desejados × estimados (t = 101..120)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def main():
    treino, teste = carregar("treinamento.csv"), carregar("teste.csv")
    serie = dict(treino + teste)
    tempos_teste = [t for t, _ in teste]
    ultimo_treino = max(t for t, _ in treino)

    resultados = {}
    for rede, (p, n1) in REDES.items():
        x, d = janelas(serie, p, range(p + 1, ultimo_treino + 1))
        x_teste, d_teste = janelas(serie, p, tempos_teste)
        resultados[rede] = []
        for t in range(1, NUM_TREINAMENTOS + 1):
            rng = np.random.default_rng(t)  # gerador reiniciado: pesos iniciais distintos por treinamento
            w1_ini = rng.random((n1, p + 1))
            w2_ini = rng.random(n1 + 1)
            w1, w2, historico, tempo = treinar(x, d, w1_ini, w2_ini)
            y_teste = saidas(w1, w2, x_teste)
            erros = erro_relativo(d_teste, y_teste)
            resultados[rede].append({
                "historico": historico, "tempo": tempo, "y_teste": y_teste,
                "erro_medio": float(erros.mean()), "variancia": float(erros.var()),
            })

    melhores = {rede: min(enumerate(rs, 1), key=lambda tr: tr[1]["erro_medio"])
                for rede, rs in resultados.items()}
    d_teste = np.array([serie[t] for t in tempos_teste])
    plotar_eqm(melhores, BASE / "tdnn_eqm.png")
    plotar_previsao(melhores, tempos_teste, d_teste, BASE / "tdnn_previsao.png")

    print("## Item 2 — Resultados dos treinamentos\n")
    print("| Treinamento | " + " | ".join(f"{r} EQM | {r} Épocas" for r in REDES) + " |")
    print("|---|" + "---|" * 2 * len(REDES))
    for t in range(NUM_TREINAMENTOS):
        celulas = [f"{resultados[r][t]['historico'][-1]:.6f} | {len(resultados[r][t]['historico']) - 1}" for r in REDES]
        print(f"| {t + 1}º (T{t + 1}) | " + " | ".join(celulas) + " |")

    print("\nTempo de processamento (s):\n")
    print("| Treinamento | " + " | ".join(REDES) + " |")
    print("|---|" + "---|" * len(REDES))
    for t in range(NUM_TREINAMENTOS):
        print(f"| T{t + 1} | " + " | ".join(f"{resultados[r][t]['tempo']:.2f}" for r in REDES) + " |")

    print("\n## Item 3 — Validação (t = 101..120)\n")
    cab = [f"{r} (T{t})" for r in REDES for t in range(1, NUM_TREINAMENTOS + 1)]
    print("| Amostra | f(t) | " + " | ".join(cab) + " |")
    print("|---|---|" + "---|" * len(cab))
    for i, t in enumerate(tempos_teste):
        valores = [f"{resultados[r][k]['y_teste'][i]:.4f}" for r in REDES for k in range(NUM_TREINAMENTOS)]
        print(f"| t = {t} | {serie[t]:.4f} | " + " | ".join(valores) + " |")
    for rotulo, chave in (("Erro relativo médio (%)", "erro_medio"), ("Variância (%²)", "variancia")):
        valores = [f"{resultados[r][k][chave]:.4f}" for r in REDES for k in range(NUM_TREINAMENTOS)]
        print(f"| **{rotulo}** | | " + " | ".join(valores) + " |")

    print("\n## Melhor treinamento de cada rede (menor erro relativo médio)\n")
    print("| Rede | Treinamento | EQM final | Épocas | Erro relativo médio (%) | Variância (%²) |")
    print("|---|---|---|---|---|---|")
    for rede, (t, r) in melhores.items():
        print(f"| {rede} | T{t} | {r['historico'][-1]:.6f} | {len(r['historico']) - 1} | "
              f"{r['erro_medio']:.4f} | {r['variancia']:.4f} |")


if __name__ == "__main__":
    main()
