# Comparacao de modelos (split de teste)

| modelo | accuracy | evaluated_users | f1 | log_loss | n_interactions | ndcg_at_k | precision | precision_at_k | recall | recall_at_k | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp (melhor) | 0.6948 | 561.0000 | 0.6941 | 0.5899 | 8767.0000 | 0.8163 | 0.6751 | 0.6552 | 0.7142 | 0.8325 | 0.7593 |
| popularity | 0.6261 | 561.0000 | 0.6258 | 0.6452 | 8767.0000 | 0.8222 | 0.6079 | 0.6561 | 0.6448 | 0.8276 | 0.6762 |
| logistic | 0.6859 | 561.0000 | 0.6794 | 0.6049 | 8767.0000 | 0.8160 | 0.6725 | 0.6575 | 0.6864 | 0.8356 | 0.7468 |
