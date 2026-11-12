"""Memória associativa de Hopfield (45 neurônios, imagens 9x5).

W = (1/N) Σ_k p_k p_kᵀ (regra do produto externo), diagonal zerada.
Recuperação: v(t+1) = tanh(β · W · v(t)) com β muito grande (≈ função sinal), atualização
síncrona até a saída não mudar mais. Pixel branco = -1, escuro = +1.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
BETA = 1000.0
LINHAS, COLUNAS = 9, 5


def carregar(nome):
    with open(BASE / "dados" / nome, encoding="utf-8") as f:
        linhas = list(csv.reader(f))[1:]
    return np.array([[float(v) for v in l if v != ""] for l in linhas if l])


def treinar(padroes):
    n = padroes.shape[1]
    w = padroes.T @ padroes / n
    np.fill_diagonal(w, 0)
    return w


def recuperar(w, v, max_iter=100):
    v = v.copy()
    for it in range(1, max_iter + 1):
        novo = np.tanh(BETA * (w @ v))
        novo = np.where(np.abs(novo) < 1e-9, v, np.sign(novo))  # u = 0 mantém o estado
        if np.array_equal(novo, v):
            return v, it - 1
        v = novo
    return v, max_iter


def classificar(v, padroes):
    """Índice (1..4) do padrão armazenado igual a v; 0 se não é padrão (estado espúrio);
    negativo se for a imagem invertida de um padrão."""
    for k, p in enumerate(padroes, 1):
        if np.array_equal(v, p):
            return k
        if np.array_equal(v, -p):
            return -k
    return 0


def desenhar(ax, v, titulo):
    ax.imshow(v.reshape(LINHAS, COLUNAS), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(-.5, COLUNAS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, LINHAS, 1), minor=True)
    ax.grid(which="minor", color="k", lw=.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(titulo, fontsize=8)


def main():
    padroes = carregar("hopfield-treinamento.csv")[:, 1:]
    teste = carregar("hopfield-teste.csv")
    w = treinar(padroes)
    print("## Resultados — Hopfield\n")
    print(f"Correlação cruzada entre padrões (p_i·p_j / 45):\n{np.round(padroes @ padroes.T / 45, 2)}\n")

    # Itens 1 e 2: 12 simulações fornecidas
    print("| Simulação | Padrão | Pixels corrompidos | Iterações | Recuperado | Resultado |")
    print("|---|---|---|---|---|---|")
    fig, eixos = plt.subplots(6, 6, figsize=(10, 14))
    acertos = 0
    for i, linha in enumerate(teste):
        k, d = int(linha[1]), linha[3:]
        orig = padroes[k - 1]
        rec, it = recuperar(w, d)
        r = classificar(rec, padroes)
        ok = r == k
        acertos += ok
        print(f"| {i + 1} | {k} (sim. {int(linha[2])}) | {int(np.sum(d != orig))} ({100 * np.mean(d != orig):.1f}%) | {it} | "
              f"{'padrão ' + str(r) if r > 0 else ('invertido do ' + str(-r) if r < 0 else 'espúrio')} | {'ok' if ok else '**falhou**'} |")
        lin, off = divmod(i, 2)
        for j, (img, t) in enumerate([(orig, "Transmitida"), (d, "Distorcida"), (rec, "Recuperada")]):
            desenhar(eixos[lin, 3 * off + j], img, f"Sim. {i + 1} (padrão {k})\n{t}" if True else t)
    fig.suptitle("12 simulações de transmissão", y=0.995)
    fig.tight_layout()
    fig.savefig(BASE / "figuras" / "hopfield_simulacoes.png", dpi=120)
    print(f"\n**Recuperações corretas: {acertos}/12**\n")

    # Item 3: aumento do nível de ruído (Monte Carlo)
    rng = np.random.default_rng(0)
    print("| Ruído | Padrão correto | Padrão errado / invertido | Estado espúrio |")
    print("|---|---|---|---|")
    niveis = [0, 10, 20, 25, 30, 35, 40, 45, 50, 60, 80]
    taxas = []
    N = 2000
    for nivel in niveis:
        cont = {"ok": 0, "outro": 0, "esp": 0}
        n_flip = round(45 * nivel / 100)
        for _ in range(N):
            k = rng.integers(4)
            d = padroes[k].copy()
            idx = rng.choice(45, n_flip, replace=False)
            d[idx] *= -1
            r = classificar(recuperar(w, d)[0], padroes)
            cont["ok" if r == k + 1 else ("esp" if r == 0 else "outro")] += 1
        taxas.append(100 * cont["ok"] / N)
        print(f"| {nivel}% ({n_flip} pixels) | {100 * cont['ok'] / N:.1f}% | {100 * cont['outro'] / N:.1f}% | {100 * cont['esp'] / N:.1f}% |")
    plt.figure(figsize=(6, 3.5))
    plt.plot(niveis, taxas, "o-")
    plt.xlabel("Ruído (% de pixels invertidos)")
    plt.ylabel("Recuperação correta (%)")
    plt.grid(alpha=.3)
    plt.title("Hopfield — taxa de recuperação × ruído (2000 ensaios/nível)")
    plt.tight_layout()
    plt.savefig(BASE / "figuras" / "hopfield_ruido.png", dpi=110)

    # exemplos visuais com ruído excessivo
    fig, eixos = plt.subplots(4, 3, figsize=(5.2, 7.5))
    rng = np.random.default_rng(3)
    for i, k in enumerate(range(4)):
        d = padroes[k].copy()
        d[rng.choice(45, 23, replace=False)] *= -1  # ~50%
        rec, _ = recuperar(w, d)
        r = classificar(rec, padroes)
        for j, (img, t) in enumerate([(padroes[k], f"Original {k + 1}"), (d, "Distorcida (51%)"), (rec, f"Saída ({'espúrio' if r == 0 else ('padrão %d' % r if r > 0 else 'invertido %d' % -r)})")]):
            desenhar(eixos[i, j], img, t)
    fig.tight_layout()
    fig.savefig(BASE / "figuras" / "hopfield_ruido_excessivo.png", dpi=110)


if __name__ == "__main__":
    main()
