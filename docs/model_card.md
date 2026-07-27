# Model Card — Recomendador de produtos (MLP com embeddings)

## Visão geral

| Item | Valor |
| --- | --- |
| Nome no registry | `ecommerce-recsys-recommender` |
| Modelo promovido | `mlp` — `MLPRecommender` (PyTorch) |
| Estágio | `Production` (alias `champion`) |
| Tarefa | Prever a relevância de um par usuário–item e ordenar itens por relevância |
| Entrada | `user_index`, `item_index` e quatro features comportamentais escalonadas |
| Saída | Probabilidade de relevância em `[0, 1]` |
| Framework | PyTorch (embeddings + MLP), comparado a baselines do Scikit-Learn |

## Dados

- **Fonte:** MovieLens `ml-latest-small` (GroupLens), usado como proxy público de
  interações usuário–item de e-commerce.
- **Volume após limpeza:** 90.274 interações, 610 usuários e 3.650 itens
  (acima do mínimo de 10.000 interações exigido).
- **Limpeza:** remoção de nulos e duplicatas por par usuário–item, além do corte
  de usuários e itens com menos de 5 interações.
- **Rótulo:** feedback implícito — a interação é positiva quando a nota é
  maior ou igual a 4,0. A base fica praticamente balanceada (50,1% de positivos
  no treino).
- **Divisão:** cronológica **dentro do histórico de cada usuário** — 80% treino,
  10% validação, 10% teste. O modelo nunca vê o futuro do próprio usuário e
  todos os usuários permanecem representados no treino.
- **Features:** `user_activity`, `item_popularity`, `user_mean_rating` e
  `item_mean_rating`, calculadas **somente no split de treino** e padronizadas
  com o `StandardScaler` do Scikit-Learn.

## Arquitetura e treino

- Embeddings de usuário e item (32 dimensões cada) concatenados às quatro
  features numéricas.
- Torre densa: `Linear(68 → 64) → ReLU → Dropout(0,2) → Linear(64 → 32) → ReLU
  → Dropout(0,2) → Linear(32 → 1)`.
- Perda `BCEWithLogitsLoss`, otimizador Adam (`lr = 0,003`,
  `weight_decay = 1e-5`), lotes de 512, no máximo 30 épocas.
- **Early stopping** com paciência de 3 épocas monitorando a perda de validação;
  os pesos da melhor época são restaurados no fim do treino.
- Seeds fixados (`RANDOM_SEED = 42`) em `random`, NumPy e PyTorch — dois treinos
  consecutivos produzem as mesmas previsões (verificado em teste automatizado).

## Desempenho (split de teste, 8.767 interações)

| Modelo | ROC AUC | Acurácia | Precisão | Recall | F1 | Log loss | Precision@10 | Recall@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **mlp (promovido)** | **0,7593** | **0,6948** | 0,6751 | **0,7142** | **0,6941** | **0,5899** | 0,6552 | 0,8325 | 0,8163 |
| logistic (baseline) | 0,7468 | 0,6859 | 0,6725 | 0,6864 | 0,6794 | 0,6049 | 0,6575 | 0,8356 | 0,8160 |
| popularity (baseline) | 0,6762 | 0,6261 | 0,6079 | 0,6448 | 0,6258 | 0,6452 | 0,6561 | 0,8276 | 0,8222 |

- A métrica primária de promoção é o **ROC AUC**, no qual a rede neural supera a
  regressão logística em 1,25 ponto e a popularidade em 8,31 pontos.
- As métricas de ranking (`@10`) são calculadas por usuário sobre as interações
  do split de teste (561 usuários com ao menos um item relevante) e ficam
  praticamente empatadas entre os três modelos — ver limitações.
- Números reproduzidos a cada `dvc repro` em `reports/metrics.json` e
  `reports/comparison.md`.

## Limitações

- **Ranking sobre candidatos observados.** As métricas `@10` reordenam apenas os
  itens que o usuário realmente avaliou no período de teste, não o catálogo
  inteiro. Como cada usuário tem poucas interações no holdout e quase metade
  delas é relevante, o teto dessas métricas é alto para qualquer modelo — é aí
  que os três ficam empatados. O ROC AUC e a log loss discriminam melhor.
- **Cold start não é atendido.** Usuários e itens ausentes do treino não têm
  embedding; essas linhas são descartadas na avaliação (`drop_cold_start`) e, em
  produção, exigiriam fallback para o modelo de popularidade.
- **Proxy de dados.** MovieLens são avaliações de filmes, não cliques de
  e-commerce. Notas explícitas convertidas em feedback implícito não capturam
  sinais reais de navegação (tempo de sessão, carrinho, recompra).
- **Escala modesta.** 610 usuários e 3.648 itens. A capacidade do modelo é
  limitada de propósito para caber no pipeline de estudo; catálogos reais exigem
  amostragem de negativos e treino distribuído.
- **Sem calibração explícita.** O limiar de decisão é fixo em 0,5, sem calibração
  de probabilidade (Platt/isotônica).

## Vieses e riscos

- **Viés de popularidade.** `item_popularity` é uma das features e o histórico já
  concentra interações em itens populares, então o modelo tende a reforçar a
  cauda curta do catálogo. Isso reduz a diversidade das recomendações e
  desfavorece itens novos — mitigável com penalização de popularidade ou
  reordenação por diversidade.
- **Viés de atividade.** Usuários muito ativos dominam o conjunto de treino;
  usuários esporádicos recebem recomendações de qualidade inferior.
- **Viés de exposição / feedback loop.** Só há rótulo para itens que o usuário
  encontrou. Ao entrar em produção, o próprio sistema passa a determinar o que é
  visto, o que realimenta seus próprios vieses se não houver exploração.
- **Ausência de atributos sensíveis.** O dataset não traz gênero, idade ou
  localização, o que impede auditoria de justiça por subgrupo. A ausência do
  atributo não garante ausência de disparidade.
- **Uso pretendido.** Ranqueamento de vitrine e recomendações de apoio. Não deve
  embasar decisões de preço, crédito ou qualquer efeito adverso sobre pessoas.

## Manutenção

- Retreino via `dvc repro`; cada execução gera runs no MLflow e recalcula a
  comparação com os baselines.
- Promoção automatizada: o melhor modelo pela métrica primária é registrado,
  passa por `Staging` e é promovido a `Production` (`reports/model_registry.json`
  guarda versão, estágio e run de origem).
- Monitoramento recomendado em produção: ROC AUC por coorte semanal, cobertura
  de catálogo e taxa de itens novos recomendados.
