## Item 1 — Camada escondida (k-means com os padrões de presença de radiação)

Padrões usados: 19 (d = +1)

| Cluster | Centro (x1, x2) | Variância | Padrões no cluster |
|---|---|---|---|
| 1 | (0.1648, 0.6121) | 0.0298 | 6 |
| 2 | (0.3990, 0.1571) | 0.0385 | 13 |

## Item 2 — Camada de saída (regra delta generalizada)

Pesos iniciais: 0.5118, 0.9505, 0.1442
Épocas: 309 · EQM final: 0.117121

| Peso | Valor |
|---|---|
| W2(1,0) | 1.0019 |
| W2(1,1) | 2.3750 |
| W2(1,2) | 2.6964 |

## Itens 3 e 4 — Validação

| Amostra | x1 | x2 | d | y | y pós | Acerto |
|---|---|---|---|---|---|---|
| 1 | 0.8705 | 0.9329 | -1 | -1.0018 | -1 | sim |
| 2 | 0.0388 | 0.2703 | +1 | -0.3229 | -1 | não |
| 3 | 0.8236 | 0.4458 | -1 | -0.9133 | -1 | sim |
| 4 | 0.7075 | 0.1502 | +1 | -0.2197 | -1 | não |
| 5 | 0.9587 | 0.8663 | -1 | -1.0018 | -1 | sim |
| 6 | 0.6115 | 0.9365 | -1 | -0.9870 | -1 | sim |
| 7 | 0.3534 | 0.3646 | +1 | 0.9659 | +1 | sim |
| 8 | 0.3268 | 0.2766 | +1 | 1.3227 | +1 | sim |
| 9 | 0.6129 | 0.4518 | -1 | -0.4677 | -1 | sim |
| 10 | 0.9948 | 0.4962 | -1 | -0.9959 | -1 | sim |

**Taxa de acerto:** 8/10 = 80.00%

Acerto no conjunto de treinamento: 37/40 = 92.50%
