# Atividade 04 (outubro) — Perceptron Multicamadas TDNN (previsão de série temporal)

Lab. Inteligência Artificial — CEFET-MG Campus VIII (Varginha) · Prof. Lázaro Eduardo da Silva

Rede **TDNN** (perceptron multicamadas com atrasos de tempo) que prevê f(t) a partir de
x(t−1)…x(t−p). Três topologias candidatas, 3 treinamentos cada, com **backpropagation com
momentum** (η = 0,1, α = 0,8, ε = 0,5×10⁻⁶):

| Rede | Entradas (p) | Neurônios ocultos (N1) |
|---|---|---|
| Rede 1 | 5 | 10 |
| Rede 2 | 10 | 15 |
| Rede 3 | 15 | 25 |

As respostas completas (tabelas, gráficos, escolha da topologia e questão teórica) estão em
[`RESPOSTAS.md`](RESPOSTAS.md).

## Estrutura

```
atividade-04-out/
├── dados/
│   ├── treinamento.csv     # série t = 1..100 (Anexo)
│   └── teste.csv           # série t = 101..120 (validação)
├── tdnn.py
├── requirements.txt
├── RESPOSTAS.md            # respostas do trabalho
├── resultados_tdnn.md      # saída gerada pelo tdnn.py
├── tdnn_eqm.png            # EQM × época do melhor treinamento de cada rede
└── tdnn_previsao.png       # desejado × estimado (t = 101..120) do melhor treinamento de cada rede
```

## Preparação

Requer Python 3.

```bash
cd atividade-04-out
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python tdnn.py > resultados_tdnn.md
```

O script executa os 9 treinamentos (3 redes × T1–T3), gera os dois gráficos e imprime, em Markdown:

- a tabela de EQM final e número de épocas de cada treinamento (item 2);
- as previsões para t = 101..120 com o erro relativo médio e a variância de cada treinamento (item 3);
- o melhor treinamento de cada rede (menor erro relativo médio), usado nos gráficos.

## Observações

- **Reprodutibilidade:** o gerador é reiniciado em cada treinamento (`default_rng(1)`, `(2)`, `(3)`),
  então os pesos iniciais diferem entre T1, T2 e T3, mas se repetem a cada execução. As mesmas
  sementes são usadas nas três redes.
- **Padrões de treinamento:** para a rede com p entradas, os padrões vão de t = p+1 até t = 100
  (95, 90 e 85 padrões para as redes 1, 2 e 3).
- **Validação:** previsão um passo à frente — as entradas de cada t são os valores reais da série
  (os de t ≥ 101 vêm de `teste.csv`), nunca as previsões anteriores da rede.
- **Erro relativo:** `|d − y| / |d| × 100` por amostra; a variância é a populacional (`numpy.var`).
- Se o script for alterado, atualize as tabelas em `RESPOSTAS.md` manualmente.
