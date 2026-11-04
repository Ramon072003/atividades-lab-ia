"""RBF 2-2-1 — detecção de radiação em compostos nucleares (presença = +1, ausência = -1).

1º estágio: k-means com os padrões de presença de radiação (d = +1) → centros e variâncias.
2º estágio: neurônio de saída linear treinado pela regra delta generalizada (online),
bias como entrada fixa -1, parada quando |EQM(época) - EQM(época-1)| <= ε.
"""
import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
DADOS = BASE / "dados"
NUM_CLUSTERS = 2
TAXA_APRENDIZADO = 0.01
PRECISAO = 1e-7
SEMENTE = 1


def carregar(nome):
    with open(DADOS / nome, encoding="utf-8") as f:
        linhas = np.array([[float(v) for v in linha[1:]] for linha in list(csv.reader(f))[1:] if linha])
    return linhas[:, :-1], linhas[:, -1]


def kmeans(x, k):
    """k-means em lote; centros iniciais = os k primeiros padrões. Retorna centros e variâncias."""
    centros = x[:k].copy()
    while True:
        grupos = np.argmin(((x[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2), axis=1)
        novos = np.array([x[grupos == j].mean(axis=0) for j in range(k)])
        if np.allclose(novos, centros):
            break
        centros = novos
    variancias = np.array([np.mean(((x[grupos == j] - centros[j]) ** 2).sum(axis=1)) for j in range(k)])
    return centros, variancias, grupos


def camada_escondida(x, centros, variancias):
    """g(u) = exp(−‖x − c‖² / (2σ²)) para cada neurônio, com o bias −1 na primeira coluna."""
    dist2 = ((x[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
    g = np.exp(-dist2 / (2 * variancias))
    return np.hstack([-np.ones((len(x), 1)), g])


def eqm(w, z, d):
    """EQM = (1/p) · Σ_k ½ · (d(k) − y(k))²"""
    return float(np.mean(0.5 * (d - z @ w) ** 2))


def treinar(z, d, w_ini):
    w = w_ini.copy()
    historico = [eqm(w, z, d)]  # EQM na época 0 (pesos iniciais)
    while True:
        for zk, dk in zip(z, d):
            w += TAXA_APRENDIZADO * (dk - zk @ w) * zk
        historico.append(eqm(w, z, d))
        if abs(historico[-1] - historico[-2]) <= PRECISAO:
            break
    return w, historico


def sinal(y):
    return np.where(y >= 0, 1, -1)


def main():
    x_treino, d_treino = carregar("rbf1_treinamento.csv")
    x_teste, d_teste = carregar("rbf1_teste.csv")

    presenca = x_treino[d_treino == 1]
    centros, variancias, grupos = kmeans(presenca, NUM_CLUSTERS)

    z_treino = camada_escondida(x_treino, centros, variancias)
    w_ini = np.random.default_rng(SEMENTE).random(NUM_CLUSTERS + 1)
    w, historico = treinar(z_treino, d_treino, w_ini)

    print("## Item 1 — Camada escondida (k-means com os padrões de presença de radiação)\n")
    print(f"Padrões usados: {len(presenca)} (d = +1)\n")
    print("| Cluster | Centro (x1, x2) | Variância | Padrões no cluster |")
    print("|---|---|---|---|")
    for j in range(NUM_CLUSTERS):
        print(f"| {j + 1} | ({centros[j, 0]:.4f}, {centros[j, 1]:.4f}) | {variancias[j]:.4f} | {np.sum(grupos == j)} |")

    print("\n## Item 2 — Camada de saída (regra delta generalizada)\n")
    print(f"Pesos iniciais: " + ", ".join(f"{v:.4f}" for v in w_ini))
    print(f"Épocas: {len(historico) - 1} · EQM final: {historico[-1]:.6f}\n")
    print("| Peso | Valor |")
    print("|---|---|")
    for i, v in enumerate(w):
        print(f"| W2(1,{i}) | {v:.4f} |")

    y_teste = camada_escondida(x_teste, centros, variancias) @ w
    pos = sinal(y_teste)
    acertos = int(np.sum(pos == d_teste))
    print("\n## Itens 3 e 4 — Validação\n")
    print("| Amostra | x1 | x2 | d | y | y pós | Acerto |")
    print("|---|---|---|---|---|---|---|")
    for i, (xk, dk, yk, pk) in enumerate(zip(x_teste, d_teste, y_teste, pos), 1):
        print(f"| {i} | {xk[0]:.4f} | {xk[1]:.4f} | {int(dk):+d} | {yk:.4f} | {pk:+d} | {'sim' if pk == dk else 'não'} |")
    print(f"\n**Taxa de acerto:** {acertos}/{len(d_teste)} = {100 * acertos / len(d_teste):.2f}%")

    acertos_treino = int(np.sum(sinal(z_treino @ w) == d_treino))
    print(f"\nAcerto no conjunto de treinamento: {acertos_treino}/{len(d_treino)}"
          f" = {100 * acertos_treino / len(d_treino):.2f}%")


if __name__ == "__main__":
    main()
