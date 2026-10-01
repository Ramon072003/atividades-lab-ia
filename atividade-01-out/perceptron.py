"""Perceptron treinado pela regra de Hebb — classificação de pureza de óleo.

Classe C1 = -1, classe C2 = +1. Entrada de bias x0 = -1 (peso w0 = θ).
"""
import csv
import random
from pathlib import Path

DADOS = Path(__file__).parent / "dados"
TAXA_APRENDIZADO = 0.01
NUM_TREINAMENTOS = 5
MAX_EPOCAS = 100_000


def carregar(nome):
    with open(DADOS / nome, encoding="utf-8") as f:
        linhas = list(csv.reader(f))[1:]
    return [[float(v) for v in linha[1:]] for linha in linhas if linha]


def sinal(u):
    return 1 if u >= 0 else -1


def produto(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


def treinar(amostras, desejados, pesos_iniciais):
    w = list(pesos_iniciais)
    epocas = 0
    while epocas < MAX_EPOCAS:
        epocas += 1
        houve_erro = False
        for x, d in zip(amostras, desejados):
            y = sinal(produto(w, x))
            if y != d:
                w = [wi + TAXA_APRENDIZADO * (d - y) * xi for wi, xi in zip(w, x)]
                houve_erro = True
        if not houve_erro:
            break
    return w, epocas


def main():
    treino = carregar("oleo_dataset.csv")
    amostras = [[-1.0] + linha[:-1] for linha in treino]
    desejados = [linha[-1] for linha in treino]
    teste = [[-1.0] + linha for linha in carregar("oleo_teste.csv")]

    resultados = []
    for t in range(1, NUM_TREINAMENTOS + 1):
        random.seed(t)  # reinicia o gerador: pesos iniciais distintos e reproduzíveis
        w_inicial = [random.random() for _ in range(4)]
        w_final, epocas = treinar(amostras, desejados, w_inicial)
        saidas = [sinal(produto(w_final, x)) for x in teste]
        resultados.append((w_inicial, w_final, epocas, saidas))

    fmt = lambda v: f"{v:.4f}"
    print("## Tabela de treinamentos\n")
    print("| Treinamento | w0 ini | w1 ini | w2 ini | w3 ini | w0 fin | w1 fin | w2 fin | w3 fin | Épocas |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t, (wi, wf, ep, _) in enumerate(resultados, 1):
        print(f"| {t}º (T{t}) | " + " | ".join(map(fmt, wi + wf)) + f" | {ep} |")

    print("\n## Classificação das amostras de teste\n")
    print("| Amostra | x1 | x2 | x3 | " + " | ".join(f"y (T{t})" for t in range(1, NUM_TREINAMENTOS + 1)) + " |")
    print("|---|---|---|---|" + "---|" * NUM_TREINAMENTOS)
    for i, x in enumerate(teste):
        ys = [f"{r[3][i]:+d}" for r in resultados]
        print(f"| {i + 1} | " + " | ".join(map(fmt, x[1:])) + " | " + " | ".join(ys) + " |")


if __name__ == "__main__":
    main()
