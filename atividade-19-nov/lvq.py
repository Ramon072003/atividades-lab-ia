"""LVQ-1 com 4 neurônios (um protótipo por classe), η = 0,05.

Inicialização: o protótipo de cada classe recebe a primeira amostra dessa classe. A cada
padrão x o neurônio vencedor j (menor distância euclidiana) é ajustado:
  mesma classe   → w_j ← w_j + η (x − w_j)
  classe diferente → w_j ← w_j − η (x − w_j)
Épocas com os 16 padrões na ordem fornecida, até a maior variação de peso entre épocas < ε.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
ETA = 0.05
EPS = 1e-6


def carregar(nome):
    with open(BASE / "dados" / nome, encoding="utf-8") as f:
        linhas = [l for l in list(csv.reader(f))[1:] if l]
    x = np.array([[float(v) for v in l[1:7]] for l in linhas])
    d = np.array([int(l[7]) for l in linhas]) if len(linhas[0]) > 7 else None
    return x, d


def vencedor(w, x):
    return int(np.argmin(((w - x) ** 2).sum(axis=1)))


def treinar(x, d):
    rotulos = np.array(sorted(set(d)))
    w = np.array([x[d == c][0] for c in rotulos])
    w0 = w.copy()
    historico = []
    epocas = 0
    while True:
        epocas += 1
        antes = w.copy()
        erros = 0
        for xk, dk in zip(x, d):
            j = vencedor(w, xk)
            if rotulos[j] == dk:
                w[j] += ETA * (xk - w[j])
            else:
                w[j] -= ETA * (xk - w[j])
                erros += 1
        historico.append(erros)
        if np.abs(w - antes).max() < EPS or epocas >= 100000:
            return w, w0, rotulos, epocas, historico


def main():
    x, d = carregar("lvq_treinamento.csv")
    w, w0, rotulos, epocas, hist = treinar(x, d)
    print("## Resultados — LVQ-1\n")
    print(f"η = {ETA} · épocas até convergir (ε = {EPS}): **{epocas}**\n")
    print("| Neurônio (classe) | Peso inicial | Peso final |")
    print("|---|---|---|")
    for j, c in enumerate(rotulos):
        print(f"| {j + 1} (classe {c}) | " + ", ".join(f"{v:.4f}" for v in w0[j]) + " | " + ", ".join(f"{v:.4f}" for v in w[j]) + " |")

    pred = np.array([rotulos[vencedor(w, xk)] for xk in x])
    print(f"\nAcerto no conjunto de treinamento: {int(np.sum(pred == d))}/{len(d)}\n")

    xt, _ = carregar("lvq_teste.csv")
    print("| Dia | 7h | 8h | 9h | 10h | 11h | 12h | Neurônio vencedor | Classe | Distância |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, xk in enumerate(xt, 1):
        j = vencedor(w, xk)
        print(f"| {i} | " + " | ".join(f"{v:.4f}" for v in xk) + f" | {j + 1} | **{rotulos[j]}** | {np.linalg.norm(xk - w[j]):.4f} |")

    # figura: perfis de potência (curvas) — protótipos e dados
    horas = np.arange(7, 13)
    cores = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for xk, dk in zip(x, d):
        ax[0].plot(horas, xk, color=cores[dk], alpha=.35)
    for j, c in enumerate(rotulos):
        ax[0].plot(horas, w[j], color=cores[c], lw=3, label=f"Protótipo classe {c}")
    ax[0].set_title("Treinamento: protótipos (linha grossa) e amostras"); ax[0].legend(fontsize=7)
    for i, xk in enumerate(xt, 1):
        c = rotulos[vencedor(w, xk)]
        ax[1].plot(horas, xk, color=cores[c], marker="o", ms=3)
        ax[1].annotate(f"dia {i}", (12, xk[-1]), fontsize=7, xytext=(3, 0), textcoords="offset points")
    ax[1].set_title("Teste: cor = classe atribuída"); ax[1].set_xlim(7, 12.8)
    for a in ax:
        a.set_xlabel("Hora"); a.grid(alpha=.3)
    ax[0].set_ylabel("Potência")
    fig.tight_layout(); fig.savefig(BASE / "figuras" / "lvq_perfis.png", dpi=120)


if __name__ == "__main__":
    main()
