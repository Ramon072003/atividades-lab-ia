# Lab. Inteligência Artificial — Redes de Base Radial, Exercício 1 (RBF 1)

CEFET-MG Campus VIII – Varginha · Prof. Lázaro Eduardo da Silva · 12/11/2026

**Aluno(a):** Ramon Oliveira Silva

Os resultados numéricos abaixo foram gerados por `rbf1.py` (saída completa em `resultados_rbf1.md`).

## Configuração

Treinamento em dois estágios:

1. **Camada escondida (não supervisionado):** *k-means* em lote com os padrões de presença de radiação; centros iniciais = os 2 primeiros padrões; variância de cada cluster `σ²_j = (1/m_j) · Σ ‖x − c_j‖²`. Ativação gaussiana `g(x) = exp(−‖x − c_j‖² / (2σ²_j))`.
2. **Camada de saída (supervisionado):** neurônio linear treinado pela regra delta generalizada (online), η = 0,01, ε = 10⁻⁷, bias como entrada fixa `−1` (peso W2(1,0)). EQM por época `E = (1/p) · Σ ½ (d − y)²`; parada quando `|EQM(época) − EQM(época anterior)| ≤ ε`. Pesos iniciais sorteados em [0, 1) (`numpy.random.default_rng`, semente 1).

---

## RBF 1 — Detecção de radiação

Topologia 2-2-1. Saída desejada: presença de radiação = +1, ausência = −1.
O Anexo fornecido contém 40 padrões de treinamento (19 com presença de radiação).

### Item 1 — Camada escondida (k-means)

Como pede o enunciado, o k-means usou apenas os 19 padrões com presença de radiação (d = +1):

| Cluster | Centro (x1, x2) | Variância | Padrões no cluster |
|---|---|---|---|
| 1 | (0.1648, 0.6121) | 0.0298 | 6 |
| 2 | (0.3990, 0.1571) | 0.0385 | 13 |

### Item 2 — Camada de saída (regra delta generalizada)

Pesos iniciais: 0.5118, 0.9505, 0.1442 · Convergência em **309 épocas** · EQM final 0.117121

| Peso | Valor |
|---|---|
| W2(1,0) | 1.0019 |
| W2(1,1) | 2.3750 |
| W2(1,2) | 2.6964 |

Interpretação: com o bias `−1`, a saída é `y = −1,0019 + 2,3750·g₁ + 2,6964·g₂`. Longe dos
dois centros (g₁ ≈ g₂ ≈ 0) a rede responde ≈ −1 (ausência); perto de um dos centros
de radiação a saída sobe para valores positivos (presença).

### Item 3 — Pós-processamento

Implementado em `sinal()`, aplicado somente às saídas do conjunto de teste:

```
y_pós =  1, se y ≥ 0
        −1, se y < 0
```

### Item 4 — Validação

| Amostra | x1 | x2 | d | y | y pós | Acerto |
|---|---|---|---|---|---|---|
| 1 | 0.8705 | 0.9329 | -1 | -1.0018 | -1 | sim |
| 2 | 0.0388 | 0.2703 | +1 | -0.3229 | -1 | **não** |
| 3 | 0.8236 | 0.4458 | -1 | -0.9133 | -1 | sim |
| 4 | 0.7075 | 0.1502 | +1 | -0.2197 | -1 | **não** |
| 5 | 0.9587 | 0.8663 | -1 | -1.0018 | -1 | sim |
| 6 | 0.6115 | 0.9365 | -1 | -0.9870 | -1 | sim |
| 7 | 0.3534 | 0.3646 | +1 | 0.9659 | +1 | sim |
| 8 | 0.3268 | 0.2766 | +1 | 1.3227 | +1 | sim |
| 9 | 0.6129 | 0.4518 | -1 | -0.4677 | -1 | sim |
| 10 | 0.9948 | 0.4962 | -1 | -0.9959 | -1 | sim |

**Taxa de acerto: 8/10 = 80%** (no próprio conjunto de treinamento a rede acerta 37/40 = 92,5%).

### Item 5 — Estratégias para aumentar a taxa de acerto

Os dois erros (amostras 2 e 4) são padrões de presença de radiação que ficam **nas bordas**
da região de radiação, longe dos dois centros: a amostra 2 (x1 = 0,04) está à esquerda de
ambos os clusters e a amostra 4 (x1 = 0,71) à direita do cluster 2. Com apenas duas
gaussianas, as funções de ativação já valem pouco nesses pontos, e o bias (−1) prevalece.
Estratégias possíveis:

1. **Aumentar o número de neurônios na camada escondida** (mais clusters): a região de
   presença de radiação tem formato alongado/irregular e não é bem coberta por apenas duas
   gaussianas; com 3 ou 4 centros ela é coberta com mais fidelidade.
2. **Ajustar a abertura (variância) das gaussianas:** usar σ² maiores (por exemplo, um
   fator multiplicativo sobre a variância calculada, ou σ baseado na distância média entre
   centros) faz cada neurônio responder numa área maior, alcançando padrões de borda.
3. **Posicionar centros também para a classe de ausência:** aplicar o k-means sobre todos
   os padrões (ou nas duas classes separadamente) dá à camada de saída mais informação
   para traçar a fronteira, em vez de depender só do bias para a classe −1.
4. **Treinar os centros e variâncias de forma supervisionada** (ajuste por gradiente de
   c_j e σ_j junto com os pesos de saída), em vez de fixá-los pelo k-means.
5. **Melhorar a inicialização do k-means:** o resultado depende dos centros iniciais;
   executar o k-means várias vezes com inicializações diferentes (ou k-means++) e manter a
   solução de menor variância intra-cluster evita mínimos locais ruins.
6. **Mais dados de treinamento** nas regiões de fronteira, que são justamente onde a rede
   errou.
