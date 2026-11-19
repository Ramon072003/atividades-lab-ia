# Lab. Inteligência Artificial — Rede ART-1

CEFET-MG Campus VIII – Varginha · Prof. Lázaro Eduardo da Silva · 19/11/2026

**Aluno(a):** Ramon Oliveira Silva

Resultados gerados por `art1.py` (saída completa em `resultados_art1.md`).

## Enunciado

10 situações de um processo industrial, descritas por 16 variáveis binárias de status (x1 a x16). A rede ART-1 deve agrupar as situações parecidas, para dar um provável diagnóstico de manutenção, com graus de vigilância ρ = 0,5; 0,8; 0,9 e 0,99. Após cada simulação, indicar quantas classes estão ativas e quais situações estão em cada uma.

## Implementação

Rede ART-1 de Carpenter e Grossberg, com 16 entradas e neurônios de saída criados à medida que são necessários:

- **Pesos:** bottom-up `b_j` e top-down `t_j` para cada classe j. Inicialização de uma nova classe com `t = x` (primeiro padrão que a ocupa); a regra de atualização abaixo usa L = 2.
- **Competição:** para cada situação x (na ordem 1 a 10) calcula-se a entrada líquida `y_j = b_j · x`; vence a classe de maior `y_j`.
- **Teste de vigilância:** `‖x ∧ t_J‖ / ‖x‖ ≥ ρ`, onde ∧ é o AND bit a bit e ‖·‖ conta os bits iguais a 1. Se falhar, a classe é inibida e a próxima mais ativa é testada. Se todas falharem, cria-se uma nova classe.
- **Ressonância:** `t_J ← x ∧ t_J` e `b_J ← L · (x ∧ t_J) / (L − 1 + ‖x ∧ t_J‖)`.
- **Treinamento:** as 10 situações são reapresentadas, em épocas, até nenhuma mudar de classe.

Quanto maior o ρ, mais parecida precisa ser a situação com o protótipo da classe para entrar nela, e portanto mais classes são criadas.

## Resultados

### ρ = 0,5 — 3 classes ativas

| Classe | Situações |
|---|---|
| A | 3, 4, 8, 9 |
| B | 2, 5, 7, 10 |
| C | 1, 6 |

O treinamento criou 4 neurônios, mas um deles (o primeiro) ficou sem nenhuma situação depois que as situações foram reclassificadas nas épocas seguintes. Por isso são 3 as classes ativas. Estabilizou em 3 épocas.

### ρ = 0,8 — 5 classes ativas

| Classe | Situações |
|---|---|
| 1 | 1, 6 |
| 2 | 2, 7 |
| 3 | 3, 8 |
| 4 | 4, 9 |
| 5 | 5, 10 |

### ρ = 0,9 — 7 classes ativas

| Classe | Situações |
|---|---|
| 1 | 1, 6 |
| 2 | 2 |
| 3 | 3, 8 |
| 4 | 4 |
| 5 | 5, 10 |
| 6 | 7 |
| 7 | 9 |

### ρ = 0,99 — 8 classes ativas

| Classe | Situações |
|---|---|
| 1 | 1 |
| 2 | 2 |
| 3 | 3, 8 |
| 4 | 4 |
| 5 | 5, 10 |
| 6 | 6 |
| 7 | 7 |
| 8 | 9 |

### Resumo

| ρ | Classes ativas | Agrupamentos |
|---|---|---|
| 0,5 | 3 | {3, 4, 8, 9} · {2, 5, 7, 10} · {1, 6} |
| 0,8 | 5 | {1, 6} · {2, 7} · {3, 8} · {4, 9} · {5, 10} |
| 0,9 | 7 | {1, 6} · {2} · {3, 8} · {4} · {5, 10} · {7} · {9} |
| 0,99 | 8 | {1} · {2} · {3, 8} · {4} · {5, 10} · {6} · {7} · {9} |

Os protótipos finais (vetores top-down) de cada classe estão em `resultados_art1.md`.

## Análise

- **Efeito da vigilância.** O número de classes cresce com ρ (3, 5, 7, 8): com ρ baixo a rede aceita situações apenas razoavelmente parecidas e forma poucos grupos grandes; com ρ alto exige quase igualdade e quebra os grupos.
- **Situações idênticas.** As situações 3 e 8 e as situações 5 e 10 têm exatamente os mesmos 16 valores, e por isso ficam juntas mesmo com ρ = 0,99. Todas as outras são distintas entre si e, a 0,99, ficam cada uma em sua classe.
- **Configuração mais informativa para o diagnóstico.** Com ρ = 0,8 as situações se organizam em cinco pares (1 e 6, 2 e 7, 3 e 8, 4 e 9, 5 e 10), estável também para ρ = 0,9 (que apenas separa os pares 2/7 e 4/9). Isso sugere cinco comportamentos recorrentes do processo, cada um com duas ocorrências, e é uma boa granularidade para associar cada classe a um diagnóstico de manutenção. Com ρ = 0,5 o agrupamento é grosseiro demais (agrupa situações que diferem em vários status) e com ρ = 0,99 praticamente não há agrupamento.
- **Observação sobre ρ = 0,5.** Se cada situação for apresentada uma única vez (sem reapresentar até estabilizar), a rede cria 4 classes: {1, 2}, {3, 4, 8, 9}, {5, 7, 10} e {6}. Com a reapresentação, os protótipos se refinam e as situações migram para 3 classes. Para os demais valores de ρ, os resultados com passada única e com reapresentação são iguais.
