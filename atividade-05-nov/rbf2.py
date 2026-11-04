"""RBF 3-N1-1 — quantidade de gasolina injetada (aproximação funcional).

1º estágio: k-means com todos os padrões de treinamento → centros e variâncias.
2º estágio: neurônio de saída linear treinado pela regra delta generalizada (online),
bias como entrada fixa -1, parada quando |EQM(época) - EQM(época-1)| <= ε.
"""
import matplotlib
import numpy as np

from rbf1 import BASE, camada_escondida, carregar, kmeans, treinar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REDES = {"Rede 1": 5, "Rede 2": 10, "Rede 3": 15}  # N1
NUM_TREINAMENTOS = 3


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
    fig.suptitle("RBF — Erro quadrático médio por época (melhor treinamento de cada rede)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def main():
    x_treino, d_treino = carregar("rbf2_treinamento.csv")
    x_teste, d_teste = carregar("rbf2_teste.csv")

    resultados, clusters = {}, {}
    for rede, n1 in REDES.items():
        centros, variancias, _ = kmeans(x_treino, n1)
        clusters[rede] = (centros, variancias)
        z_treino = camada_escondida(x_treino, centros, variancias)
        z_teste = camada_escondida(x_teste, centros, variancias)
        resultados[rede] = []
        for t in range(1, NUM_TREINAMENTOS + 1):
            rng = np.random.default_rng(t)  # gerador reiniciado: pesos iniciais distintos por treinamento
            w, historico = treinar(z_treino, d_treino, rng.random(n1 + 1))
            y_teste = z_teste @ w
            erros = erro_relativo(d_teste, y_teste)
            resultados[rede].append({
                "historico": historico, "y_teste": y_teste,
                "erro_medio": float(erros.mean()), "variancia": float(erros.var()),
            })

    melhores = {rede: min(enumerate(rs, 1), key=lambda tr: tr[1]["erro_medio"])
                for rede, rs in resultados.items()}
    plotar_eqm(melhores, BASE / "rbf2_eqm.png")

    print("## Camada escondida (k-means com os 150 padrões de treinamento)\n")
    for rede, (centros, variancias) in clusters.items():
        print(f"**{rede} (N1 = {len(centros)})**\n")
        print("| Cluster | Centro (x1, x2, x3) | Variância |")
        print("|---|---|---|")
        for j, (c, v) in enumerate(zip(centros, variancias), 1):
            print(f"| {j} | (" + ", ".join(f"{ci:.4f}" for ci in c) + f") | {v:.4f} |")
        print()

    print("## Item 2 — Resultados dos treinamentos\n")
    print("| Treinamento | " + " | ".join(f"{r} EQM | {r} Épocas" for r in REDES) + " |")
    print("|---|" + "---|" * 2 * len(REDES))
    for t in range(NUM_TREINAMENTOS):
        celulas = [f"{resultados[r][t]['historico'][-1]:.6f} | {len(resultados[r][t]['historico']) - 1}" for r in REDES]
        print(f"| {t + 1}º (T{t + 1}) | " + " | ".join(celulas) + " |")

    print("\n## Item 3 — Validação\n")
    cab = [f"{r} (T{t})" for r in REDES for t in range(1, NUM_TREINAMENTOS + 1)]
    print("| Amostra | x1 | x2 | x3 | d | " + " | ".join(cab) + " |")
    print("|---|---|---|---|---|" + "---|" * len(cab))
    for i, (xk, dk) in enumerate(zip(x_teste, d_teste)):
        valores = [f"{resultados[r][k]['y_teste'][i]:.4f}" for r in REDES for k in range(NUM_TREINAMENTOS)]
        print(f"| {i + 1:02d} | " + " | ".join(f"{v:.4f}" for v in xk) + f" | {dk:.4f} | " + " | ".join(valores) + " |")
    for rotulo, chave in (("Erro relativo médio (%)", "erro_medio"), ("Variância (%²)", "variancia")):
        valores = [f"{resultados[r][k][chave]:.4f}" for r in REDES for k in range(NUM_TREINAMENTOS)]
        print(f"| **{rotulo}** | | | | | " + " | ".join(valores) + " |")

    print("\n## Melhor treinamento de cada rede (menor erro relativo médio)\n")
    print("| Rede | Treinamento | EQM final | Épocas | Erro relativo médio (%) | Variância (%²) |")
    print("|---|---|---|---|---|---|")
    for rede, (t, r) in melhores.items():
        print(f"| {rede} | T{t} | {r['historico'][-1]:.6f} | {len(r['historico']) - 1} | "
              f"{r['erro_medio']:.4f} | {r['variancia']:.4f} |")


if __name__ == "__main__":
    main()
