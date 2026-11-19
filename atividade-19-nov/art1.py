"""Rede ART-1 (Carpenter & Grossberg) para agrupar as 10 situações de 16 variáveis binárias.

Pesos: b (bottom-up, 16 × m) e t (top-down, m × 16). Inicialização: b_ij = L/(L − 1 + n), t_ji = 1,
com L = 2 e n = 16. Para cada padrão x (ordem fornecida):
  1. calcula a entrada líquida y_j = Σ_i b_ij x_i dos neurônios (classes) ainda não inibidos;
  2. escolhe o vencedor J (maior y_j; empate → menor índice);
  3. teste de vigilância: ‖x ∧ t_J‖ / ‖x‖ ≥ ρ ?  Sim → ressonância; não → inibe J e volta ao passo 2;
  4. na ressonância: t_J ← x ∧ t_J;  b_J ← L·(x ∧ t_J) / (L − 1 + ‖x ∧ t_J‖);
  5. se todos os neurônios ativos forem rejeitados, cria-se uma nova classe com t = x.
Os padrões são reapresentados em épocas até que nenhum padrão mude de classe.
"""
import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
L = 2.0
VIGILANCIAS = [0.5, 0.8, 0.9, 0.99]


def carregar(nome):
    with open(BASE / "dados" / nome, encoding="utf-8") as f:
        linhas = [l for l in list(csv.reader(f))[1:] if l]
    return np.array([[int(v) for v in l[1:]] for l in linhas])


def apresentar(x, b, t, rho):
    """Apresenta x à rede; devolve a classe (índice) e atualiza b e t (listas de vetores)."""
    inibidos = set()
    while True:
        candidatos = [j for j in range(len(t)) if j not in inibidos]
        if not candidatos:  # nenhuma classe aceita x → cria nova
            t.append(x.astype(float).copy())
            b.append(L * x / (L - 1 + x.sum()))
            return len(t) - 1
        y = {j: b[j] @ x for j in candidatos}
        j = max(candidatos, key=lambda k: (y[k], -k))
        inter = np.minimum(x, t[j])
        if inter.sum() / x.sum() >= rho:
            t[j] = inter
            b[j] = L * inter / (L - 1 + inter.sum())
            return j
        inibidos.add(j)


def treinar(x, rho):
    b, t = [], []
    classes = [-1] * len(x)
    epocas = 0
    while True:
        epocas += 1
        mudou = False
        for k, xk in enumerate(x):
            j = apresentar(xk, b, t, rho)
            if classes[k] != j:
                mudou = True
                classes[k] = j
        if not mudou:
            break
    return classes, t, epocas


def main():
    x = carregar("art1_treinamento.csv")
    print("## Resultados — ART-1\n")
    print(f"{len(x)} situações, {x.shape[1]} variáveis binárias, L = {L}\n")
    for rho in VIGILANCIAS:
        classes, t, epocas = treinar(x, rho)
        ativas = [j for j in range(len(t)) if j in classes]  # neurônios com pelo menos uma situação
        print(f"### ρ = {rho}\n")
        extra = f" (neurônios criados: {len(t)}; o neurônio {[j + 1 for j in range(len(t)) if j not in classes]} ficou sem situações)" if len(ativas) != len(t) else ""
        print(f"Classes ativas: **{len(ativas)}**{extra} · épocas até estabilizar: {epocas}\n")
        print("| Classe (neurônio) | Situações | Protótipo (t) |")
        print("|---|---|---|")
        for j in ativas:
            sit = [k + 1 for k, c in enumerate(classes) if c == j]
            print(f"| {j + 1} | {', '.join(map(str, sit))} | `{''.join(str(int(v)) for v in t[j])}` |")
        print()


if __name__ == "__main__":
    main()
