"""Perceptron multicamadas 4-15-3 — escolha do conservante (tipos A, B e C).

Treina a mesma rede (mesmos pesos iniciais) com backpropagation padrão e com
backpropagation com momentum. Bias como entrada fixa x0 = -1 em cada camada.
Treinamento online (ajuste a cada padrão), parada quando |EQM(época) - EQM(época-1)| <= ε.
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
NEURONIOS_OCULTOS = 15
TAXA_APRENDIZADO = 0.1
MOMENTUM = 0.9
PRECISAO = 1e-6
SEMENTE = 1


def carregar(nome):
    with open(DADOS / nome, encoding="utf-8") as f:
        linhas = [[float(v) for v in linha[1:]] for linha in list(csv.reader(f))[1:] if linha]
    dados = np.array(linhas)
    x = np.hstack([-np.ones((len(dados), 1)), dados[:, :4]])  # x0 = -1 (bias)
    return x, dados[:, 4:]


def sigmoide(u):
    return 1.0 / (1.0 + np.exp(-u))


def propagar(w1, w2, x):
    y1 = sigmoide(w1 @ x)
    y1b = np.concatenate(([-1.0], y1))  # bias da camada de saída
    y2 = sigmoide(w2 @ y1b)
    return y1b, y2


def eqm(w1, w2, x, d):
    """EQM = (1/p) · Σ_k ½ · Σ_j (d_j(k) − y_j(k))²"""
    y1 = sigmoide(x @ w1.T)
    y1b = np.hstack([-np.ones((len(x), 1)), y1])
    y2 = sigmoide(y1b @ w2.T)
    return float(np.mean(0.5 * np.sum((d - y2) ** 2, axis=1)))


def treinar(x, d, w1_ini, w2_ini, momentum=0.0):
    w1, w2 = w1_ini.copy(), w2_ini.copy()
    dw1_ant, dw2_ant = np.zeros_like(w1), np.zeros_like(w2)
    historico = [eqm(w1, w2, x, d)]  # EQM na época 0 (pesos iniciais)
    inicio = time.perf_counter()
    while True:
        for xk, dk in zip(x, d):
            y1b, y2 = propagar(w1, w2, xk)
            delta2 = (dk - y2) * y2 * (1 - y2)
            delta1 = (w2[:, 1:].T @ delta2) * y1b[1:] * (1 - y1b[1:])
            dw2 = TAXA_APRENDIZADO * np.outer(delta2, y1b) + momentum * dw2_ant
            dw1 = TAXA_APRENDIZADO * np.outer(delta1, xk) + momentum * dw1_ant
            w2 += dw2
            w1 += dw1
            dw1_ant, dw2_ant = dw1, dw2
        historico.append(eqm(w1, w2, x, d))
        if abs(historico[-1] - historico[-2]) <= PRECISAO:
            break
    return w1, w2, historico, time.perf_counter() - inicio


def pos_processar(y):
    return (y >= 0.5).astype(int)


def plotar_eqm(resultados, caminho):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (nome, (_, _, historico, tempo)) in zip(eixos, resultados.items()):
        ax.plot(range(len(historico)), historico, color="#1f77b4")
        ax.set_title(f"{nome} — {len(historico) - 1} épocas ({tempo:.1f} s)")
        ax.set_xlabel("Época")
        ax.set_ylabel("EQM")
        ax.grid(alpha=0.3)
    fig.suptitle("MLP 4-15-3 — Erro quadrático médio por época")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def main():
    x_treino, d_treino = carregar("treinamento.csv")
    x_teste, d_teste = carregar("teste.csv")

    rng = np.random.default_rng(SEMENTE)
    w1_ini = rng.random((NEURONIOS_OCULTOS, 5))      # 15 neurônios × (bias + 4 entradas)
    w2_ini = rng.random((3, NEURONIOS_OCULTOS + 1))  # 3 saídas × (bias + 15 ocultos)

    resultados = {
        "Backpropagation padrão": treinar(x_treino, d_treino, w1_ini, w2_ini),
        "Backpropagation com momentum": treinar(x_treino, d_treino, w1_ini, w2_ini, MOMENTUM),
    }
    plotar_eqm(resultados, BASE / "mlp_eqm.png")

    print("## Treinamentos\n")
    print("| Algoritmo | Épocas | EQM final | Tempo de processamento (s) |")
    print("|---|---|---|---|")
    for nome, (_, _, historico, tempo) in resultados.items():
        print(f"| {nome} | {len(historico) - 1} | {historico[-1]:.6f} | {tempo:.2f} |")

    for nome, (w1, w2, _, _) in resultados.items():
        print(f"\n## Validação — {nome}\n")
        print("| Amostra | x1 | x2 | x3 | x4 | d1 | d2 | d3 | y1 pós | y2 pós | y3 pós | y1 | y2 | y3 | Acerto |")
        print("|---|" + "---|" * 14)
        acertos = 0
        for i, (xk, dk) in enumerate(zip(x_teste, d_teste), 1):
            y = propagar(w1, w2, xk)[1]
            pos = pos_processar(y)
            ok = bool(np.array_equal(pos, dk.astype(int)))
            acertos += ok
            print(f"| {i} | " + " | ".join(f"{v:.4f}" for v in xk[1:]) + " | "
                  + " | ".join(str(int(v)) for v in dk) + " | "
                  + " | ".join(str(v) for v in pos) + " | "
                  + " | ".join(f"{v:.4f}" for v in y) + f" | {'sim' if ok else 'não'} |")
        print(f"\n**Taxa de acerto:** {acertos}/{len(x_teste)} = {100 * acertos / len(x_teste):.2f}%")


if __name__ == "__main__":
    main()
