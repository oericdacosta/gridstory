# MLFlow Setup e Métricas de ML - gridstory

## ✅ Implementação Concluída

Este documento resume a integração completa do **MLFlow** e **métricas de avaliação** para o pipeline de Machine Learning do gridstory.

---

## 📦 O Que Foi Implementado

### 1. Métricas de Avaliação Completas (`src/ml/metrics.py`)

#### Métricas de Clustering
- **Silhouette Score**: Coesão vs separação dos clusters [-1, 1]
- **Davies-Bouldin Index**: Compacidade vs separação [0, ∞), menor é melhor
- **Calinski-Harabasz Score**: Ratio between/within variance [0, ∞), maior é melhor
- **Inércia**: Soma das distâncias aos centróides (K-Means específico)

#### Métricas de Anomaly Detection
- **n_anomalies**: Número total de anomalias detectadas
- **anomaly_rate**: Taxa de anomalias (%)
- **score_mean/std**: Estatísticas dos scores de anomalia
- **anomaly_score_mean**: Média dos scores das anomalias
- **normal_score_mean**: Média dos scores dos pontos normais

#### Funções Utilitárias
- `calculate_clustering_metrics()`: Calcula todas as métricas de clustering
- `calculate_anomaly_metrics()`: Calcula métricas de anomaly detection
- `calculate_cluster_statistics()`: Estatísticas descritivas por cluster
- `evaluate_clustering_quality()`: Avalia qualidade (excellent, good, fair, poor)

---

### 2. Integração MLFlow (`src/ml/tracking.py`)

#### Funcionalidades Principais
- **setup_mlflow()**: Configuração inicial do MLFlow (autolog **desabilitado** por padrão)
- **track_clustering_run()**: Tracka runs de clustering (K-Means, DBSCAN)
- **track_anomaly_detection_run()**: Tracka runs de anomaly detection
- **track_pipeline_run()**: Tracka pipeline completo + salva artefatos CSV
- **compare_runs()**: Compara múltiplos runs
- **get_best_run()**: Encontra melhor run baseado em métrica

> **Por que autolog está desabilitado?** O autolog do scikit-learn cria um child run para cada `fit()`, gerando centenas de runs com `conda.yaml`, `requirements.txt` e pickles de modelo — mais de 10 mil arquivos por análise. As métricas relevantes são logadas manualmente com mais controle.

#### O Que é Trackeado
**Parâmetros (Inputs)**:
- `n_clusters`, `eps`, `min_samples`, `contamination`
- `random_state`, `scaler_type`, `structural_filter_threshold`
- `cluster_features`, `anomaly_features`

**Métricas (Outputs)** — todas calculadas **por piloto**:
- `clustering_silhouette_mean`, `clustering_silhouette_std`
- `clustering_davies_bouldin_mean`
- `clustering_calinski_harabasz_mean`
- `clustering_n_drivers_evaluated`
- `clustering_n_structural_filtered`
- `clustering_driver_VER_silhouette`, `clustering_driver_HAM_silhouette`, etc.
- Métricas DBSCAN: `clustering_dbscan_n_noise`, `clustering_dbscan_silhouette_mean`
- Anomaly detection: `anomaly_n_anomalies`, `anomaly_anomaly_rate`, `anomaly_score_mean`

**Artefatos** (visíveis na aba **Artifacts** do MLFlow UI):
- `results/laps_clustered.csv` — voltas com `cluster_label` por piloto
- `results/laps_anomalies.csv` — voltas com `is_anomaly` e `anomaly_score`
- `results/cluster_statistics.csv` — média/std por cluster
- `results/per_driver_metrics.csv` — silhouette, DB, CH por piloto

---

### 3. Pipeline Atualizado (`src/ml/pipeline.py`)

**Função `run_race_analysis()` agora inclui**:
- Parâmetro `enable_mlflow`: Habilita tracking
- Parâmetro `experiment_name`: Nome do experimento
- Parâmetro `run_name`: Nome do run (opcional)
- Retorna `mlflow_run_id` nos resultados
- Retorna `clustering_metrics` e `anomaly_metrics` completas

---

### 4. Tracking Config-Driven

O tracking MLFlow é habilitado diretamente no `config.yaml`:

```yaml
mlflow:
  enabled: true                    # habilitar/desabilitar sem mudar código
  tracking_uri: "file:./mlruns"
  experiment_prefix: "F1"          # experimento: F1_{year}_Round_{round:02d}
```

Com `enabled: true`, cada execução de `uv run python cli/pipeline.py 2025 1` cria automaticamente um run no MLFlow com métricas, parâmetros e artefatos CSV.

---

### 5. Exemplo Completo (`examples/mlflow_example.py`)

**Script de demonstração com 3 exemplos**:
1. **Basic Tracking**: Tracking básico de uma análise completa
2. **Experimentation**: Experimentação com diferentes hiperparâmetros
3. **Comparison**: Comparação de runs e seleção do melhor

```bash
# Executar exemplos
uv run python examples/mlflow_example.py

# Depois visualizar no MLFlow UI
mlflow ui
# Acesse: http://localhost:5000
```

---

### 6. Documentação Completa (`src/ml/README.md`)

**Seções adicionadas**:
- Seção 4: Métricas de Avaliação
  - 4.1. Métricas de Clustering
  - 4.2. Métricas de Anomaly Detection
- Seção 5: MLFlow Tracking
  - 5.1. Visão Geral
  - 5.2. Setup Inicial
  - 5.3. O Que é Trackeado
  - 5.4. Tracking no Pipeline
  - 5.5. CLI de Análise
  - 5.6. Comparar Experimentos
  - 5.7. Tracking Manual (Avançado)
  - 5.8. Fluxo de Trabalho Recomendado
  - 5.9. Interpretação de Resultados

---

## 🚀 Como Usar

### Setup Inicial (Primeira Vez)

```bash
# 1. Instalar dependências (MLFlow já incluído)
uv sync

# 2. Habilitar MLFlow no config.yaml (já vem habilitado por padrão)
# mlflow.enabled: true

# 3. Rodar pipeline — tracking acontece automaticamente
uv run python cli/pipeline.py 2025 1
```

### Visualizar Resultados

```bash
# Iniciar MLFlow UI (sempre use uv run)
uv run mlflow ui

# Acessar: http://localhost:5000
```

### Exemplo Programático

```python
from src.ml import setup_mlflow, run_race_analysis
import pandas as pd

# 1. Carregar dados
laps_df = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')

# 2. Setup MLFlow
setup_mlflow("F1_2025_Round_01")

# 3. Executar análise COM tracking
results = run_race_analysis(
    laps_df=laps_df,
    analysis_type='all',
    enable_mlflow=True,
    experiment_name='F1_2025_Round_01',
    run_name='Full_Analysis',
)

# 4. Ver resultados
print(f"MLFlow Run ID: {results['mlflow_run_id']}")
print(results['clustering_metrics'])
print(results['anomaly_metrics'])
```

---

## 📊 Métricas: Como Interpretar

### Clustering (K-Means)

**Bom clustering**:
- ✅ Silhouette > 0.5
- ✅ Davies-Bouldin < 1.0
- ✅ Clusters fazem sentido no contexto F1

**Clustering ruim**:
- ❌ Silhouette < 0.25
- ❌ Davies-Bouldin > 2.0
- 🔧 Ações: Ajustar features, tentar DBSCAN, revisar pré-processamento

### Anomaly Detection

**Configuração adequada**:
- ✅ Taxa de anomalias: 2-5% (esperado para corrida limpa)
- ✅ Anomalias correspondem a eventos reais
- ✅ Scores das anomalias << scores dos normais

**Ajustes necessários**:
- Taxa muito alta (>10%): Reduzir `contamination`
- Taxa muito baixa (<1%): Aumentar `contamination`
- Anomalias não fazem sentido: Revisar features

---

## 🔬 Fluxo de Trabalho Recomendado

### 1. Rodar pipeline com tracking

```bash
# MLFlow habilitado via config.yaml (mlflow.enabled: true)
uv run python cli/pipeline.py 2025 1
```

### 2. Análise de Resultados (MLFlow UI)

```bash
# Iniciar UI
uv run mlflow ui

# Acesse http://localhost:5000
# Compare runs, visualize métricas, identifique melhor configuração
# Na aba "Artifacts" de cada run: laps_clustered.csv, per_driver_metrics.csv, tire_cliffs.csv, etc.
```

### 4. Comparação Programática

```python
from src.ml import compare_runs, get_best_run

# Ver todos os runs
comparison = compare_runs("F1_2025_Round_01")
print(comparison)

# Encontrar melhor configuração
best = get_best_run("F1_2025_Round_01", "silhouette_score")
print(f"Melhor: {best['params']}")
```

### 5. Produção (Melhor Modelo)

```python
import mlflow

# Carregar melhor modelo
best_run_id = best['run_id']
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

# Usar em produção
predictions = model.predict(new_data)
```

---

## 📁 Estrutura de Arquivos

```
src/ml/
├── pipeline.py             # run_race_analysis() — clustering + anomaly + changepoint + mlflow
├── clustering.py           # K-Means e DBSCAN por piloto
├── anomaly_detection.py    # Isolation Forest
├── change_point.py         # Ruptures/PELT — tire cliffs
├── metrics.py              # Métricas de avaliação (silhouette, Davies-Bouldin, etc.)
├── tracking.py             # Integração MLFlow
└── README.md               # Documentação completa

cli/
├── pipeline.py             # Único ponto de entrada do pipeline completo
└── ruptures_analysis.py    # Calibração de penalty (penalty-search)

mlruns/                     # ✅ Gerado automaticamente pelo MLFlow
└── [experiments]/
    └── [runs]/
        ├── metrics/
        ├── params/
        ├── artifacts/
        └── meta.yaml
```

---

## ✅ Checklist de Validação

### Métricas Implementadas
- [x] Silhouette Score
- [x] Davies-Bouldin Index
- [x] Calinski-Harabasz Score
- [x] Inércia (K-Means)
- [x] Métricas de anomaly detection
- [x] Estatísticas por cluster
- [x] Avaliação qualitativa de clustering

### MLFlow Implementado
- [x] Setup e configuração
- [x] Tracking de clustering (métricas por piloto)
- [x] Tracking de anomaly detection
- [x] Tracking de pipeline completo
- [x] Autolog **desabilitado** (evita 10k+ child runs)
- [x] Comparação de runs
- [x] Seleção do melhor run
- [x] Artefatos CSV visíveis na UI (laps_clustered, laps_anomalies, per_driver_metrics)
- [x] Logging de métricas por piloto (silhouette, davies-bouldin por driver)

### Código e Documentação
- [x] Módulo de métricas (`metrics.py`)
- [x] Módulo de tracking (`tracking.py`)
- [x] Pipeline atualizado com MLFlow
- [x] Tracking config-driven via pipeline.py
- [x] Exemplos funcionais
- [x] Documentação completa
- [x] README atualizado

### Dependências
- [x] MLFlow instalado (`>=3.10.0rc0`)
- [x] Compatibilidade com PyArrow 23+
- [x] `uv sync --prerelease=allow` funcional

---

## 🎯 Próximos Passos

### Imediato (Já Pode Fazer)
1. ✅ **Executar análises com tracking**
   ```bash
   uv run python -m cli.ml_analysis --year 2025 --round 1 --mlflow
   ```

2. ✅ **Visualizar no MLFlow UI**
   ```bash
   uv run mlflow ui
   ```

3. ✅ **Experimentar com diferentes hiperparâmetros**
   - Variar `contamination` (0.03, 0.05, 0.10)
   - Variar `n_clusters` (auto-detect vs fixo)
   - Variar `scaler_type` (standard vs robust)

4. ✅ **Comparar resultados e selecionar melhor configuração**

### Curto Prazo (Próximas Features)
- [ ] Visualizações (matplotlib) com tracking de plots
- [ ] Testes unitários para métricas
- [ ] Validação cruzada para hiperparâmetros

### Médio Prazo (Integração Completa)
- [ ] Integrar tracking de Ruptures
- [ ] Adicionar tracking de LLM (quando implementar DSPY/Agno)
- [ ] Dashboard customizado de métricas
- [ ] Alertas automáticos de qualidade de modelo

---

## 📚 Referências

- **Documentação MLFlow**: https://mlflow.org/docs/latest/
- **Documentação do Módulo**: [src/ml/README.md](src/ml/README.md)
- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
- **Código-fonte**: `src/ml/`

---

## 🐛 Troubleshooting

### MLFlow não está logando métricas

**Problema**: `autolog` não captura métricas de clustering

**Solução**: Métricas de clustering não supervisionado requerem logging manual (já implementado):
```python
from sklearn.metrics import silhouette_score
import mlflow

with mlflow.start_run():
    score = silhouette_score(X, labels)
    mlflow.log_metric("silhouette_score", score)
```

### Experimentos não aparecem no UI

**Soluções**:
1. Verificar se `mlflow ui` está rodando no diretório correto
2. Verificar se `tracking_uri` está configurado: `file:./mlruns`
3. Verificar se experimento existe: `mlflow.get_experiment_by_name(name)`

### Métricas retornam None

**Causas comuns**:
- Menos de 2 clusters
- Clusters com 1 amostra apenas
- Dados insuficientes após filtrar ruído (DBSCAN)

**Solução**: Verificar `n_clusters` e `n_samples` nas métricas retornadas

### Conflito de dependências MLFlow/PyArrow

**Solução**: Usar versão rc do MLFlow
```bash
uv sync --prerelease=allow
```

---

## ✨ Conclusão

Você agora tem:
- ✅ **Métricas completas** para avaliar qualidade do ML
- ✅ **MLFlow tracking** para rastrear experimentos
- ✅ **CLI dedicado** para análise com tracking
- ✅ **Exemplos funcionais** para aprender
- ✅ **Documentação completa** para referência

**Pronto para:**
1. Avaliar corretamente seus modelos de ML
2. Experimentar com diferentes configurações
3. Comparar resultados e escolher a melhor abordagem
4. Avançar para Ruptures com confiança

**Próximo passo sugerido:**
```bash
# 1. Rodar pipeline (MLFlow habilitado via config.yaml)
uv run python cli/pipeline.py 2025 1

# 2. Visualizar resultados
uv run mlflow ui

# 3. Verificar aba "Artifacts" de cada run para ver os CSVs gerados
```

---

**Data de Implementação**: 2026-02-16
**Versão do Projeto**: 0.1.0
**MLFlow**: 3.10.0rc0
