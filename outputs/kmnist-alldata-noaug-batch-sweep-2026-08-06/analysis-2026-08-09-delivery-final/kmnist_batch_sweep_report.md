# Análise técnica da varredura de batch size do KMNIST

## Resumo técnico

A varredura foi concluída com **32 de 32 runs** e **1600 épocas** persistidas. Todas as runs têm status `completed`, resumo de treino, métricas de teste e telemetria disponíveis. Não há comando de retomada pendente.

A maior macro-F1 de teste foi **98.4855%** no batch **16**. Para comparação de eficiência, o batch **432** foi o mais rápido entre as runs com cronometragem completa, com **50.73 s por época**. O maior throughput comparável foi **965.81 exemplos/s** no batch **432**.

## Escopo, dados e definições

- Dataset: KMNIST, 70.000 imagens em escala de cinza, 64x64 px, 10 classes.
- Split estratificado fixo, seed 42: 49.000 treino, 10.500 validação e 10.500 teste; 4.900 exemplos de treino por classe.
- Protocolo fixo: `unit_interval`, `all_raw`, FP16, Adam com learning rate 0,0003 e 50 épocas.
- Augmentation desativada: `extra_fraction=0.0` e todas as transformações em valores neutros.
- Variável experimental: somente o batch size, de 16 a 512 em passos de 16.
- Macro-F1 de teste: média não ponderada do F1 das dez classes.
- Tempo canônico: soma de `epoch_seconds` disponíveis. Uma run só é comparável em eficiência quando há 50 tempos de época finitos.

## Verificação de qualidade dos artefatos

Os controles validaram status final, arquivos obrigatórios, sequência 1..50 de épocas, configuração constante fora do batch, split, contagem de previsões e suporte das classes de teste. Resultado dos checks:

| Severidade | Resultado | Checks |
| --- | --- | --- |
| critical | Aprovado | 352 |
| high | Aprovado | 64 |
| medium | Aprovado | 57 |
| medium | Falhou | 7 |

Ressalvas: batch 48: cobertura parcial de telemetria (4/50 eventos epoch_end); batch 64: cobertura parcial de telemetria (41/50 eventos epoch_end); batch 224: cobertura parcial de telemetria (7/50 eventos epoch_end)

Essas ressalvas não invalidam as métricas finais de qualidade nem os tempos por época dos batches afetados: as 50 épocas, previsões e avaliação de teste estão presentes. Elas limitam a interpretação de seus máximos de hardware, que podem não representar a execução inteira.

## Métricas de qualidade por run

As diferenças de desempenho são descritivas deste único seed. Não há replicações suficientes para afirmar superioridade estatística entre batches próximos.

| Batch | Melhor época | Melhor val F1 | Val F1 final | Teste acc. | Teste bal. acc. | Teste macro-F1 | Teste loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 42 | 0.9858 | 0.9829 | 0.9849 | 0.9849 | 0.9849 | 0.0911 |
| 32 | 50 | 0.9854 | 0.9854 | 0.9844 | 0.9844 | 0.9844 | 0.0811 |
| 48 | 47 | 0.9830 | 0.9829 | 0.9834 | 0.9834 | 0.9834 | 0.0981 |
| 64 | 44 | 0.9839 | 0.9810 | 0.9812 | 0.9812 | 0.9812 | 0.0844 |
| 80 | 37 | 0.9809 | 0.9790 | 0.9813 | 0.9813 | 0.9813 | 0.0892 |
| 96 | 43 | 0.9817 | 0.9802 | 0.9797 | 0.9797 | 0.9797 | 0.1100 |
| 112 | 37 | 0.9798 | 0.9772 | 0.9800 | 0.9800 | 0.9800 | 0.1014 |
| 128 | 50 | 0.9789 | 0.9789 | 0.9790 | 0.9790 | 0.9790 | 0.1189 |
| 144 | 43 | 0.9790 | 0.9777 | 0.9797 | 0.9797 | 0.9798 | 0.1083 |
| 160 | 40 | 0.9801 | 0.9772 | 0.9781 | 0.9781 | 0.9781 | 0.1214 |
| 176 | 39 | 0.9779 | 0.9754 | 0.9775 | 0.9775 | 0.9775 | 0.1197 |
| 192 | 50 | 0.9769 | 0.9769 | 0.9765 | 0.9765 | 0.9765 | 0.1361 |
| 208 | 42 | 0.9773 | 0.9760 | 0.9780 | 0.9780 | 0.9780 | 0.1238 |
| 224 | 43 | 0.9780 | 0.9748 | 0.9766 | 0.9766 | 0.9766 | 0.1266 |
| 240 | 42 | 0.9771 | 0.9760 | 0.9767 | 0.9767 | 0.9767 | 0.1200 |
| 256 | 38 | 0.9752 | 0.9741 | 0.9730 | 0.9730 | 0.9730 | 0.1408 |
| 272 | 41 | 0.9766 | 0.9765 | 0.9784 | 0.9784 | 0.9784 | 0.1156 |
| 288 | 50 | 0.9765 | 0.9765 | 0.9743 | 0.9743 | 0.9742 | 0.1496 |
| 304 | 41 | 0.9759 | 0.9748 | 0.9769 | 0.9769 | 0.9769 | 0.1148 |
| 320 | 49 | 0.9757 | 0.9687 | 0.9744 | 0.9744 | 0.9744 | 0.1365 |
| 336 | 36 | 0.9748 | 0.9744 | 0.9735 | 0.9735 | 0.9735 | 0.1324 |
| 352 | 40 | 0.9764 | 0.9712 | 0.9774 | 0.9774 | 0.9774 | 0.1215 |
| 368 | 41 | 0.9760 | 0.9723 | 0.9768 | 0.9768 | 0.9768 | 0.1234 |
| 384 | 49 | 0.9751 | 0.9723 | 0.9737 | 0.9737 | 0.9738 | 0.1400 |
| 400 | 46 | 0.9748 | 0.9743 | 0.9750 | 0.9750 | 0.9750 | 0.1408 |
| 416 | 50 | 0.9743 | 0.9743 | 0.9750 | 0.9750 | 0.9751 | 0.1386 |
| 432 | 39 | 0.9735 | 0.9694 | 0.9715 | 0.9715 | 0.9715 | 0.1308 |
| 448 | 44 | 0.9737 | 0.9719 | 0.9741 | 0.9741 | 0.9741 | 0.1390 |
| 464 | 36 | 0.9742 | 0.9737 | 0.9739 | 0.9739 | 0.9739 | 0.1355 |
| 480 | 50 | 0.9747 | 0.9747 | 0.9757 | 0.9757 | 0.9757 | 0.1358 |
| 496 | 45 | 0.9758 | 0.9739 | 0.9733 | 0.9733 | 0.9733 | 0.1436 |
| 512 | 50 | 0.9735 | 0.9735 | 0.9753 | 0.9753 | 0.9753 | 0.1400 |

## Tempo de treinamento por run

O tempo registrado em todas as runs soma **29.84 h**. Deste total, **29.84 h** pertencem às 32 runs com cobertura integral, apropriadas para comparação de eficiência.

| Batch | Épocas | Cronometradas | Cobertura | Treino s | s/época | Exemplos/s | Avaliação s | Comparável |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 50 | 50 | 1.0000 | 9339.0678 | 186.7814 | 262.3388 | 65.7787 | Sim |
| 32 | 50 | 50 | 1.0000 | 6692.6405 | 133.8528 | 366.0737 | 44.8621 | Sim |
| 48 | 50 | 50 | 1.0000 | 4794.6719 | 95.8934 | 510.9839 | 30.3895 | Sim |
| 64 | 50 | 50 | 1.0000 | 4357.7814 | 87.1556 | 562.2127 | 29.1929 | Sim |
| 80 | 50 | 50 | 1.0000 | 3908.6473 | 78.1729 | 626.8153 | 27.3664 | Sim |
| 96 | 50 | 50 | 1.0000 | 3676.8050 | 73.5361 | 666.3394 | 25.9867 | Sim |
| 112 | 50 | 50 | 1.0000 | 3285.0902 | 65.7018 | 745.7938 | 23.4225 | Sim |
| 128 | 50 | 50 | 1.0000 | 3278.5809 | 65.5716 | 747.2745 | 23.1759 | Sim |
| 144 | 50 | 50 | 1.0000 | 3013.6345 | 60.2727 | 812.9718 | 22.1989 | Sim |
| 160 | 50 | 50 | 1.0000 | 3052.3606 | 61.0472 | 802.6575 | 23.8677 | Sim |
| 176 | 50 | 50 | 1.0000 | 2940.0937 | 58.8019 | 833.3068 | 21.5432 | Sim |
| 192 | 50 | 50 | 1.0000 | 2672.7216 | 53.4544 | 916.6686 | 21.3743 | Sim |
| 208 | 50 | 50 | 1.0000 | 2664.0631 | 53.2813 | 919.6479 | 21.1575 | Sim |
| 224 | 50 | 50 | 1.0000 | 2913.5959 | 58.2719 | 840.8853 | 30.9624 | Sim |
| 240 | 50 | 50 | 1.0000 | 3602.7958 | 72.0559 | 680.0274 | 23.6924 | Sim |
| 256 | 50 | 50 | 1.0000 | 2916.7465 | 58.3349 | 839.9770 | 21.4079 | Sim |
| 272 | 50 | 50 | 1.0000 | 3682.9029 | 73.6581 | 665.2361 | 24.9541 | Sim |
| 288 | 50 | 50 | 1.0000 | 3185.1952 | 63.7039 | 769.1836 | 21.2742 | Sim |
| 304 | 50 | 50 | 1.0000 | 2895.5590 | 57.9112 | 846.1233 | 24.5061 | Sim |
| 320 | 50 | 50 | 1.0000 | 2950.4625 | 59.0092 | 830.3783 | 21.7012 | Sim |
| 336 | 50 | 50 | 1.0000 | 2688.8637 | 53.7773 | 911.1656 | 20.8137 | Sim |
| 352 | 50 | 50 | 1.0000 | 2682.1746 | 53.6435 | 913.4379 | 18.7663 | Sim |
| 368 | 50 | 50 | 1.0000 | 2611.6364 | 52.2327 | 938.1091 | 18.8550 | Sim |
| 384 | 50 | 50 | 1.0000 | 2720.3428 | 54.4069 | 900.6218 | 18.8196 | Sim |
| 400 | 50 | 50 | 1.0000 | 2732.4497 | 54.6490 | 896.6313 | 18.8058 | Sim |
| 416 | 50 | 50 | 1.0000 | 2554.9443 | 51.0989 | 958.9250 | 18.4806 | Sim |
| 432 | 50 | 50 | 1.0000 | 2536.7223 | 50.7344 | 965.8132 | 18.8706 | Sim |
| 448 | 50 | 50 | 1.0000 | 2648.9045 | 52.9781 | 924.9107 | 18.4769 | Sim |
| 464 | 50 | 50 | 1.0000 | 2601.8697 | 52.0374 | 941.6306 | 18.0615 | Sim |
| 480 | 50 | 50 | 1.0000 | 2577.3662 | 51.5473 | 950.5828 | 17.9850 | Sim |
| 496 | 50 | 50 | 1.0000 | 2610.7185 | 52.2144 | 938.4390 | 18.3576 | Sim |
| 512 | 50 | 50 | 1.0000 | 2629.2961 | 52.5859 | 931.8083 | 18.2940 | Sim |

## Uso de hardware por run

O pico de VRAM observado foi **5.36 GiB** no batch **224**, abaixo dos 8 GiB da RTX 3050. A maior temperatura observada foi **57.0 C** no batch **16**. Não há status de OOM/falha na matriz concluída.

| Batch | Epoch ends | Cobertura | GPU média % | GPU p95 % | VRAM máx. GiB | GPU máx. C | RAM máx. % | RSS máx. GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16.0000 | 49.0000 | 0.9800 | 33.7229 | 44.0000 | 5.0089 | 57.0000 | 23.1000 | 3.6460 |
| 32.0000 | 50.0000 | 1.0000 | 27.5952 | 41.0000 | 4.1447 | 49.0000 | 23.9000 | 3.7776 |
| 48.0000 | 4.0000 | 0.0800 | 31.0275 | 45.6000 | 2.5236 | 52.0000 | 20.4000 | 3.2707 |
| 64.0000 | 41.0000 | 0.8200 | 28.5162 | 42.0000 | 2.9195 | 50.0000 | 23.7000 | 3.7639 |
| 80.0000 | 50.0000 | 1.0000 | 22.7279 | 40.0000 | 2.8293 | 49.0000 | 22.2000 | 3.5382 |
| 96.0000 | 50.0000 | 1.0000 | 21.2570 | 41.7500 | 2.7620 | 50.0000 | 23.7000 | 3.7556 |
| 112.0000 | 50.0000 | 1.0000 | 18.5816 | 43.0000 | 2.7889 | 50.0000 | 24.1000 | 3.8236 |
| 128.0000 | 50.0000 | 1.0000 | 18.4982 | 41.0000 | 2.7761 | 50.0000 | 24.0000 | 3.8121 |
| 144.0000 | 50.0000 | 1.0000 | 18.3082 | 43.0000 | 2.7745 | 54.0000 | 22.3000 | 3.5547 |
| 160.0000 | 50.0000 | 1.0000 | 18.8525 | 40.0000 | 2.7834 | 48.0000 | 22.4000 | 3.5622 |
| 176.0000 | 50.0000 | 1.0000 | 19.2289 | 43.0000 | 2.7996 | 48.0000 | 22.8000 | 3.6243 |
| 192.0000 | 50.0000 | 1.0000 | 19.5634 | 50.0000 | 2.7866 | 49.0000 | 24.6000 | 3.8913 |
| 208.0000 | 50.0000 | 1.0000 | 23.6614 | 53.0000 | 3.6494 | 52.0000 | 23.6000 | 3.7493 |
| 224.0000 | 7.0000 | 0.1400 | 36.9252 | 54.0000 | 5.3645 | 56.0000 | 20.9000 | 3.3398 |
| 240.0000 | 50.0000 | 1.0000 | 36.4918 | 57.0000 | 5.1889 | 56.0000 | 22.9000 | 3.6543 |
| 256.0000 | 50.0000 | 1.0000 | 33.7207 | 57.0000 | 4.5055 | 54.0000 | 23.5000 | 3.7490 |
| 272.0000 | 50.0000 | 1.0000 | 33.6527 | 52.0000 | 4.5771 | 54.0000 | 22.7000 | 3.6260 |
| 288.0000 | 50.0000 | 1.0000 | 27.9324 | 51.0000 | 4.5146 | 53.0000 | 23.2000 | 3.6996 |
| 304.0000 | 50.0000 | 1.0000 | 20.1028 | 55.0000 | 4.2117 | 51.0000 | 23.9000 | 3.8164 |
| 320.0000 | 50.0000 | 1.0000 | 23.0671 | 54.0000 | 4.2103 | 53.0000 | 23.0000 | 3.6739 |
| 336.0000 | 50.0000 | 1.0000 | 24.0445 | 57.0000 | 4.2039 | 53.0000 | 21.7000 | 3.4774 |
| 352.0000 | 50.0000 | 1.0000 | 19.9263 | 54.0000 | 4.1863 | 51.0000 | 24.0000 | 3.8285 |
| 368.0000 | 50.0000 | 1.0000 | 26.7633 | 61.0000 | 4.1959 | 53.0000 | 22.5000 | 3.5999 |
| 384.0000 | 50.0000 | 1.0000 | 25.3376 | 57.0000 | 4.2755 | 53.0000 | 25.0000 | 3.9634 |
| 400.0000 | 50.0000 | 1.0000 | 22.9116 | 57.0000 | 4.1516 | 52.0000 | 23.3000 | 3.7027 |
| 416.0000 | 50.0000 | 1.0000 | 21.2836 | 62.9000 | 3.6667 | 52.0000 | 23.2000 | 3.6960 |
| 432.0000 | 50.0000 | 1.0000 | 21.3187 | 65.0000 | 4.1731 | 51.0000 | 24.3000 | 3.8567 |
| 448.0000 | 50.0000 | 1.0000 | 19.3759 | 59.0000 | 4.1692 | 51.0000 | 24.4000 | 3.8758 |
| 464.0000 | 50.0000 | 1.0000 | 20.1595 | 58.5000 | 4.1725 | 51.0000 | 23.4000 | 3.7265 |
| 480.0000 | 50.0000 | 1.0000 | 19.8195 | 61.0000 | 4.1669 | 51.0000 | 22.7000 | 3.6102 |
| 496.0000 | 50.0000 | 1.0000 | 20.2273 | 60.4000 | 4.1577 | 51.0000 | 22.7000 | 3.6043 |
| 512.0000 | 50.0000 | 1.0000 | 19.9275 | 58.0000 | 4.1538 | 50.0000 | 22.5000 | 3.5814 |

## Limitações, incerteza e robustez

- Há uma só réplica (seed 42) por batch; a variação de qualidade não possui intervalo de confiança.
- Os batches 48, 64, 224 têm cobertura parcial de telemetria; seus resultados de hardware são amostras e não devem ser usados para comparar picos. Os registros de tempo por época estão completos e entram no ranking de eficiência.
- A telemetria mede amostras a cada cinco segundos; picos muito breves podem não ter sido observados.
- As conclusões valem para KMNIST, a arquitetura CNN do projeto e este protocolo de pré-processamento.

## Próximos passos recomendados

1. Reexecutar batches 48, 64, 224 em novo diretório caso seja necessário completar o perfil de VRAM e demais métricas de hardware desses pontos.
2. Repetir os batches candidatos com pelo menos três seeds antes de escolher batch por qualidade.
3. Para uso operacional, escolher o menor batch que cumpra a meta de macro-F1 e tenha tempo/hardware aceitáveis; usar a tabela de cobertura para evitar decisões com medições parciais.

## Arquivos de apoio

- `run_metrics.csv`: métricas de qualidade, tempo e cobertura por batch.
- `hardware_summary.csv`: telemetria agregada por batch.
- `epoch_metrics_all.csv`: 1.600 linhas de histórico por época.
- `per_class_metrics.csv`: métricas de precisão, recall, F1 e suporte por classe.
- `quality_checks.csv` e `quality_summary.csv`: evidências da validação.
- `report_artifact.json`: artefato canônico usado para gerar o HTML/PDF.
