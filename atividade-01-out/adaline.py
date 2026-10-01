"""ADALINE treinado pela Regra Delta — roteamento de sinais para as válvulas A/B.

Válvula A = -1, válvula B = +1. Entrada de bias x0 = -1 (peso w0 = θ).
"""
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DADOS = BASE / "dados"
TAXA_APRENDIZADO = 0.0025
PRECISAO = 1e-6
NUM_TREINAMENTOS = 5


def carregar(nome):
    with open(DADOS / nome, encoding="utf-8") as f:
        linhas = list(csv.reader(f))[1:]
    return [[float(v) for v in linha[1:]] for linha in linhas if linha]


def sinal(u):
    return 1 if u >= 0 else -1


def produto(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


def eqm(w, amostras, desejados):
    return sum((d - produto(w, x)) ** 2 for x, d in zip(amostras, desejados)) / len(amostras)


def treinar(amostras, desejados, pesos_iniciais):
    w = list(pesos_iniciais)
    historico = [eqm(w, amostras, desejados)]  # EQM na época 0 (pesos iniciais)
    while True:
        for x, d in zip(amostras, desejados):
            u = produto(w, x)
            w = [wi + TAXA_APRENDIZADO * (d - u) * xi for wi, xi in zip(w, x)]
        historico.append(eqm(w, amostras, desejados))
        if abs(historico[-1] - historico[-2]) <= PRECISAO:
            break
    return w, len(historico) - 1, historico


def plotar_eqm(resultados, caminho):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    for t, ax in enumerate(eixos, 1):
        historico = resultados[t - 1][3]
        ax.plot(range(len(historico)), historico, color="#1f77b4")
        ax.set_title(f"Treinamento T{t} — {len(historico) - 1} épocas")
        ax.set_xlabel("Época")
        ax.set_ylabel("EQM")
        ax.grid(alpha=0.3)
    fig.suptitle("ADALINE — Erro quadrático médio por época")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def main():
    treino = carregar("adaline_treinamento.csv")
    amostras = [[-1.0] + linha[:-1] for linha in treino]
    desejados = [linha[-1] for linha in treino]
    teste = [[-1.0] + linha for linha in carregar("adaline_teste.csv")]

    resultados = []
    for t in range(1, NUM_TREINAMENTOS + 1):
        random.seed(t)  # reinicia o gerador: pesos iniciais distintos e reproduzíveis
        w_inicial = [random.random() for _ in range(5)]
        w_final, epocas, historico = treinar(amostras, desejados, w_inicial)
        saidas = [sinal(produto(w_final, x)) for x in teste]
        resultados.append((w_inicial, w_final, epocas, historico, saidas))

    plotar_eqm(resultados, BASE / "adaline_eqm.png")

    fmt = lambda v: f"{v:.4f}"
    print("## Tabela de treinamentos\n")
    print("| Treinamento | " + " | ".join(f"w{i} ini" for i in range(5)) + " | "
          + " | ".join(f"w{i} fin" for i in range(5)) + " | Épocas | EQM final |")
    print("|---|" + "---|" * 12)
    for t, (wi, wf, ep, hist, _) in enumerate(resultados, 1):
        print(f"| {t}º (T{t}) | " + " | ".join(map(fmt, wi + wf)) + f" | {ep} | {hist[-1]:.6f} |")

    print("\n## Classificação das amostras de teste (-1 = válvula A, +1 = válvula B)\n")
    print("| Amostra | x1 | x2 | x3 | x4 | " + " | ".join(f"y (T{t})" for t in range(1, NUM_TREINAMENTOS + 1)) + " |")
    print("|---|---|---|---|---|" + "---|" * NUM_TREINAMENTOS)
    for i, x in enumerate(teste):
        ys = [f"{r[4][i]:+d} ({'A' if r[4][i] < 0 else 'B'})" for r in resultados]
        print(f"| {i + 1} | " + " | ".join(map(fmt, x[1:])) + " | " + " | ".join(ys) + " |")

    acertos = sum(sinal(produto(resultados[0][1], x)) == d for x, d in zip(amostras, desejados))
    print(f"\nAcerto no conjunto de treinamento (T1): {acertos}/{len(amostras)}")


if __name__ == "__main__":
    main()
