# Atividade 01 (outubro) — Perceptron e ADALINE

Lab. Inteligência Artificial — CEFET-MG Campus VIII (Varginha) · Prof. Lázaro Eduardo da Silva

- **Perceptron** (regra de Hebb, η = 0,01): classifica a pureza de um óleo nas classes C1 (−1) e C2 (+1).
- **ADALINE** (regra Delta, η = 0,0025, ε = 10⁻⁶): decide se um sinal vai para a válvula A (−1) ou B (+1).

As respostas completas (tabelas, gráfico e questões teóricas) estão em [`RESPOSTAS.md`](RESPOSTAS.md).

## Estrutura

```
atividade-01-out/
├── dados/
│   ├── oleo_dataset.csv          # treinamento do perceptron (30 padrões)
│   ├── oleo_teste.csv            # amostras a classificar (10)
│   ├── adaline_treinamento.csv   # treinamento do ADALINE (35 padrões)
│   └── adaline_teste.csv         # amostras a classificar (15)
├── perceptron.py
├── adaline.py
├── requirements.txt
├── RESPOSTAS.md                  # respostas do trabalho
├── resultados_perceptron.md      # saída gerada pelo perceptron.py
├── resultados_adaline.md         # saída gerada pelo adaline.py
└── adaline_eqm.png               # gráfico EQM × época (T1 e T2)
```

## Preparação

Requer Python 3. O perceptron usa só a biblioteca padrão; o ADALINE precisa do
`matplotlib` para gerar o gráfico.

```bash
cd atividade-01-out
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Treinamento do Perceptron

```bash
python perceptron.py
```

Executa os 5 treinamentos (T1–T5) e imprime, em Markdown:

- a tabela com os pesos iniciais, pesos finais e número de épocas de cada treinamento;
- a classificação das 10 amostras de `oleo_teste.csv` para cada treinamento.

Para salvar a saída em arquivo:

```bash
python perceptron.py > resultados_perceptron.md
```

## Treinamento do ADALINE

```bash
python adaline.py
```

Executa os 5 treinamentos (T1–T5) e:

- imprime a tabela de pesos iniciais, pesos finais, número de épocas e EQM final;
- imprime a classificação das 15 amostras de `adaline_teste.csv` (A ou B) para cada treinamento;
- gera `adaline_eqm.png` com os gráficos de EQM por época de T1 e T2 lado a lado.

Para salvar a saída em arquivo:

```bash
python adaline.py > resultados_adaline.md
```

## Observações

- **Reprodutibilidade:** em cada treinamento o gerador aleatório é reiniciado com uma semente
  diferente (`random.seed(1)` … `random.seed(5)`), então os pesos iniciais são distintos entre
  os treinamentos, mas os resultados se repetem a cada execução.
- **Dados:** os CSVs são cópias exatas do anexo dos enunciados, sem normalização. O bias é
  tratado como a entrada fixa `x0 = −1` com peso `w0 = θ`.
- **Treino × teste:** os pesos são ajustados apenas com o conjunto de treinamento; os arquivos
  de teste são usados somente para a classificação após o treinamento.
- Se algum script for alterado, atualize as tabelas em `RESPOSTAS.md` manualmente.
