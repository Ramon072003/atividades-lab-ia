# Atividade 05 (novembro) — Redes RBF

Lab. Inteligência Artificial — CEFET-MG Campus VIII (Varginha) · Prof. Lázaro Eduardo da Silva

- **RBF 1** (2-2-1): classifica a presença (+1) ou ausência (−1) de radiação em compostos
  nucleares. k-means com os padrões de presença de radiação + regra delta generalizada
  (η = 0,01, ε = 10⁻⁷).
- **RBF 2** (3-N1-1, N1 = 5, 10 e 15): estima a quantidade de gasolina injetada
  (aproximação funcional). 3 treinamentos por topologia, η = 0,01, ε = 10⁻⁷.

As respostas completas (tabelas, gráfico e análises) estão em [`RESPOSTAS.md`](RESPOSTAS.md).

## Estrutura

```
atividade-05-nov/
├── dados/
│   ├── rbf1_treinamento.csv   # 40 padrões do Apêndice da RBF 1
│   ├── rbf1_teste.csv         # 10 padrões de validação
│   ├── rbf2_treinamento.csv   # 150 padrões do Anexo da RBF 2
│   └── rbf2_teste.csv         # 15 padrões de validação
├── rbf1.py
├── rbf2.py                    # reutiliza k-means e regra delta de rbf1.py
├── requirements.txt
├── RESPOSTAS.md               # respostas do trabalho
├── resultados_rbf1.md         # saída gerada pelo rbf1.py
├── resultados_rbf2.md         # saída gerada pelo rbf2.py
└── rbf2_eqm.png               # EQM × época do melhor treinamento de cada rede (RBF 2)
```

## Preparação

Requer Python 3.

```bash
cd atividade-05-nov
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python rbf1.py > resultados_rbf1.md
python rbf2.py > resultados_rbf2.md
```

- `rbf1.py` imprime os centros e variâncias dos 2 clusters, os pesos finais da camada de
  saída e a validação (saída real, saída pós-processada pela função sinal e taxa de acerto).
- `rbf2.py` imprime os centros e variâncias de cada rede, a tabela de EQM/épocas dos
  9 treinamentos, as estimativas para as 15 amostras de teste com erro relativo médio e
  variância, e gera `rbf2_eqm.png`.

## Observações

- **k-means:** em lote, com os centros iniciais iguais aos N1 primeiros padrões
  considerados; para quando nenhum centro muda. É determinístico, então os centros são os
  mesmos em T1, T2 e T3 da RBF 2 — só os pesos iniciais da camada de saída mudam.
- **Reprodutibilidade:** pesos de saída sorteados em [0, 1) com `numpy.random.default_rng`
  (semente 1 na RBF 1; sementes 1, 2 e 3 para T1–T3 na RBF 2).
- **Bias:** entrada fixa `−1` no neurônio de saída (peso W2(1,0)).
- **Erro relativo:** `|d − y| / |d| × 100` por amostra; a variância é a populacional (`numpy.var`).
- Se algum script for alterado, atualize as tabelas em `RESPOSTAS.md` manualmente.
