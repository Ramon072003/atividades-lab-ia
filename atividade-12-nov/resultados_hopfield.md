## Resultados — Hopfield

Correlação cruzada entre padrões (p_i·p_j / 45):
[[ 1.   -0.16  0.02 -0.11]
 [-0.16  1.    0.64  0.07]
 [ 0.02  0.64  1.    0.42]
 [-0.11  0.07  0.42  1.  ]]

| Simulação | Padrão | Pixels corrompidos | Iterações | Recuperado | Resultado |
|---|---|---|---|---|---|
| 1 | 1 (sim. 1) | 9 (20.0%) | 1 | padrão 1 | ok |
| 2 | 1 (sim. 2) | 9 (20.0%) | 1 | padrão 1 | ok |
| 3 | 1 (sim. 3) | 9 (20.0%) | 1 | padrão 1 | ok |
| 4 | 2 (sim. 1) | 9 (20.0%) | 1 | espúrio | **falhou** |
| 5 | 2 (sim. 2) | 9 (20.0%) | 1 | padrão 3 | **falhou** |
| 6 | 2 (sim. 3) | 9 (20.0%) | 1 | padrão 2 | ok |
| 7 | 3 (sim. 1) | 9 (20.0%) | 1 | padrão 3 | ok |
| 8 | 3 (sim. 2) | 9 (20.0%) | 1 | padrão 3 | ok |
| 9 | 3 (sim. 3) | 9 (20.0%) | 1 | padrão 3 | ok |
| 10 | 4 (sim. 1) | 9 (20.0%) | 1 | padrão 4 | ok |
| 11 | 4 (sim. 2) | 9 (20.0%) | 1 | padrão 4 | ok |
| 12 | 4 (sim. 3) | 9 (20.0%) | 1 | padrão 4 | ok |

**Recuperações corretas: 10/12**

| Ruído | Padrão correto | Padrão errado / invertido | Estado espúrio |
|---|---|---|---|
| 0% (0 pixels) | 100.0% | 0.0% | 0.0% |
| 10% (4 pixels) | 85.9% | 0.8% | 13.3% |
| 20% (9 pixels) | 76.2% | 6.8% | 17.1% |
| 25% (11 pixels) | 71.5% | 9.3% | 19.2% |
| 30% (14 pixels) | 61.8% | 16.1% | 22.2% |
| 35% (16 pixels) | 49.4% | 25.5% | 25.1% |
| 40% (18 pixels) | 36.2% | 36.7% | 27.1% |
| 45% (20 pixels) | 16.1% | 53.5% | 30.4% |
| 50% (22 pixels) | 2.4% | 71.2% | 26.4% |
| 60% (27 pixels) | 0.0% | 71.8% | 28.2% |
| 80% (36 pixels) | 0.0% | 84.2% | 15.8% |
