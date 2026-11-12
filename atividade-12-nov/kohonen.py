"""Rede de Kohonen 4x4 (N1 = 16), η = 0,001, raio de vizinhança 1.

Vencedor: neurônio de menor distância euclidiana ‖x − w_j‖. Atualização ("Norma Euclidiana"):
w_j ← w_j + η (x − w_j) para o vencedor e seus vizinhos de raio 1 no grid (adjacentes na
horizontal/vertical, como no diagrama do enunciado). Treinamento por épocas (padrões na ordem
fornecida) até que a maior variação de peso entre épocas seja < ε.
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
ETA = 0.001
EPS = 1e-6
LINHAS = COLS = 4
SEMENTE = 1


def carregar(nome):
    with open(BASE / "dados" / nome, encoding="utf-8") as f:
        linhas = list(csv.reader(f))[1:]
    return np.array([[float(v) for v in l[1:4]] for l in linhas]), [l[4] if len(l) > 4 else None for l in linhas]


def vizinhos(j):
    """Neurônios a distância 1 no grid (índice j de 0 a 15, numeração por linhas)."""
    r, c = divmod(j, COLS)
    return [rr * COLS + cc for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= rr < LINHAS and 0 <= cc < COLS]


def vencedor(w, x):
    return int(np.argmin(((w - x) ** 2).sum(axis=1)))


def treinar(x):
    w = np.random.default_rng(SEMENTE).random((LINHAS * COLS, 3))
    w0 = w.copy()
    epocas = 0
    while True:
        antes = w.copy()
        for xk in x:
            j = vencedor(w, xk)
            for n in [j] + vizinhos(j):
                w[n] += ETA * (xk - w[n])
        epocas += 1
        if np.abs(w - antes).max() < EPS or epocas >= 200000:
            return w, w0, epocas


def nome(j):
    return j + 1


def main():
    x, classes = carregar("kohonen-treinamento.csv")
    w, w0, epocas = treinar(x)
    print("## Resultados — Kohonen\n")
    print(f"Épocas: {epocas} (η = {ETA}, ε = {EPS}, semente {SEMENTE})\n")
    print("Pesos iniciais e finais dos 16 neurônios:\n")
    print("| Neurônio | w inicial | w final |")
    print("|---|---|---|")
    for j in range(16):
        print(f"| {nome(j)} | ({w0[j, 0]:.3f}, {w0[j, 1]:.3f}, {w0[j, 2]:.3f}) | ({w[j, 0]:.4f}, {w[j, 1]:.4f}, {w[j, 2]:.4f}) |")

    # Item 1: neurônio vencedor de cada amostra de treinamento, por classe
    venc = [vencedor(w, xk) for xk in x]
    por_neuronio = {j: Counter() for j in range(16)}
    for v, c in zip(venc, classes):
        por_neuronio[v][c] += 1
    print("\n### Item 1 — Neurônios ativados por classe (amostras de treinamento)\n")
    print("| Classe | Amostras | Neurônios vencedores (nº de amostras) |")
    print("|---|---|---|")
    for c, faixa in zip("ABC", ["1–20", "21–60", "61–120"]):
        cont = {nome(j): por_neuronio[j][c] for j in range(16) if por_neuronio[j][c]}
        print(f"| {c} | {faixa} | " + ", ".join(f"{k} ({v})" for k, v in sorted(cont.items())) + " |")
    mistos = [nome(j) for j in range(16) if len(por_neuronio[j]) > 1]
    print(f"\nNeurônios que respondem a mais de uma classe: {mistos or 'nenhum'}")
    sem = [nome(j) for j in range(16) if not por_neuronio[j]]
    print(f"Neurônios que nunca venceram: {sem}")

    # rótulo de cada neurônio: classe majoritária; neurônio sem amostras → classe do neurônio rotulado mais próximo (pesos)
    rot = {j: por_neuronio[j].most_common(1)[0][0] for j in range(16) if por_neuronio[j]}
    for j in range(16):
        if j not in rot:
            rot[j] = rot[min(rotulados := list(k for k in range(16) if por_neuronio[k]),
                             key=lambda k: np.linalg.norm(w[k] - w[j]))]

    # Item 2
    xt, _ = carregar("kohonen-teste.csv")
    print("\n### Item 2 — Classificação das amostras de teste\n")
    print("| Amostra | x1 | x2 | x3 | Neurônio vencedor | Classe |")
    print("|---|---|---|---|---|---|")
    for i, xk in enumerate(xt, 1):
        j = vencedor(w, xk)
        print(f"| {i} | {xk[0]:.4f} | {xk[1]:.4f} | {xk[2]:.4f} | {nome(j)} | {rot[j]} |")

    # figuras
    cores = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for j in range(16):
        r, c = divmod(j, COLS)
        for n in vizinhos(j):
            rn, cn = divmod(n, COLS)
            ax.plot([c, cn], [-r, -rn], color="#bbb", zorder=1)
        cont = por_neuronio[j]
        ax.scatter(c, -r, s=1500, color=cores[rot[j]], alpha=.9 if cont else .25, zorder=2, edgecolor="k")
        ax.text(c, -r + .08, str(nome(j)), ha="center", va="center", fontweight="bold")
        txt = ", ".join(f"{k}:{v}" for k, v in sorted(cont.items())) or "—"
        ax.text(c, -r - .2, txt, ha="center", va="center", fontsize=7)
    ax.set_xlim(-.6, 3.6); ax.set_ylim(-3.6, .6); ax.axis("off")
    ax.set_title("Grid 4×4: classe dominante de cada neurônio\n(cor = classe; texto = amostras de treino ativando o neurônio)", fontsize=9)
    fig.tight_layout(); fig.savefig(BASE / "figuras" / "kohonen_grid.png", dpi=110)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    for c in "ABC":
        pts = np.array([xk for xk, k in zip(x, classes) if k == c])
        ax.scatter(*pts.T, s=8, color=cores[c], label=f"Classe {c}")
    ax.scatter(*w.T, s=60, color="k", marker="^", label="Pesos dos neurônios")
    for j in range(16):
        ax.text(*w[j], str(nome(j)), fontsize=7)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3"); ax.legend(fontsize=7)
    ax.set_title("Dados e pesos treinados", fontsize=9)
    fig.tight_layout(); fig.savefig(BASE / "figuras" / "kohonen_pesos.png", dpi=110)


if __name__ == "__main__":
    main()
