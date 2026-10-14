# Atividade 03 (outubro) — Perceptron Multicamadas (classificação)

Lab. Inteligência Artificial — CEFET-MG Campus VIII (Varginha) · Prof. Lázaro Eduardo da Silva

Rede **MLP 4-15-3** que indica o tipo de conservante (A, B ou C) a partir de x1…x4, treinada com
**backpropagation padrão** (η = 0,1) e **backpropagation com momentum** (η = 0,1, α = 0,9),
ambos com ε = 10⁻⁶ e as mesmas matrizes de pesos iniciais.

As respostas completas (tabelas, gráficos e análises) estão em [`RESPOSTAS.md`](RESPOSTAS.md).

## Estrutura

```
atividade-03-out/
├── dados/
│   ├── treinamento.csv     # 130 padrões do Anexo
│   └── teste.csv           # 18 padrões de validação
├── mlp.py
├── requirements.txt
├── RESPOSTAS.md            # respostas do trabalho
├── resultados_mlp.md       # saída gerada pelo mlp.py
└── mlp_eqm.png             # gráficos EQM × época (padrão e momentum)
```

## Preparação

Requer Python 3.

```bash
cd atividade-03-out
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python mlp.py > resultados_mlp.md
```

O script:

- treina a rede com os dois algoritmos, medindo o tempo de processamento de cada um;
- gera `mlp_eqm.png` com os gráficos de EQM por época lado a lado;
- imprime, em Markdown, a tabela de épocas/EQM/tempo e a validação com o conjunto de teste
  (saídas reais, saídas pós-processadas e taxa de acerto).

## Observações

- **Reprodutibilidade:** os pesos iniciais vêm de `numpy.random.default_rng(1)` (valores em [0, 1)).
  Épocas e EQM se repetem a cada execução; o tempo de processamento varia com a máquina.
- **Bias:** tratado como entrada fixa `−1` em cada camada.
- **Treino × teste:** os pesos são ajustados apenas com `treinamento.csv`; o pós-processamento
  (`y ≥ 0,5 → 1`) é aplicado somente às saídas do conjunto de teste.
- Se o script for alterado, atualize as tabelas em `RESPOSTAS.md` manualmente.
