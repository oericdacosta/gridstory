# Guia de Machine Learning - PitWall AI

Documentação completa do pipeline de **Machine Learning não supervisionado** usando Scikit-learn para análise de corridas de Fórmula 1.

## Visão Geral

O módulo de ML transforma dados pré-processados em **eventos e insights estatísticos** usando aprendizado **não supervisionado**:

- **Clusterização** (K-Means, DBSCAN) - Agrupa voltas semelhantes para identificar ritmos
- **Detecção de Anomalias** (Isolation Forest) - Identifica eventos raros e outliers
- **Pipeline** - Integra pré-processamento + ML em um fluxo unificado
- **Métricas** - Avaliação completa (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **MLFlow Tracking** - Rastreamento de experimentos, parâmetros e métricas

**Por quê não supervisionado?**

O sistema analisa padrões **dentro da própria corrida conforme ela ocorre**. Não há labels históricos de "ritmo puro" vs "gestão de pneus" - o algoritmo descobre esses padrões automaticamente.

---

## Instalação

```bash
uv add scikit-learn  # Já incluído no projeto
```

---

## Fluxo do Pipeline

```
Dados Brutos (Parquet)
    ↓
[1. Pré-processamento]
    ├─ Imputação (SimpleImputer/KNNImputer)
    ├─ Encoding (OneHotEncoder)
    └─ Escalonamento (StandardScaler/RobustScaler)
    ↓
Dados Limpos e Escalonados
    ↓
[2. Machine Learning]
    ├─ Clustering (K-Means / DBSCAN)
    └─ Anomaly Detection (Isolation Forest)
    ↓
Eventos Estruturados (JSON) → [Futuro: LLM]
```

---

## 1. Clusterização: Análise de Ritmo

### Objetivo

Agrupar voltas semelhantes para identificar diferentes **modos de pilotagem**:
- **Push** (`cluster_semantic='push'`) — voltas mais rápidas que o ritmo dominante do piloto
- **Base pace** (`cluster_semantic='base'`) — ritmo sustentado na maior parte da corrida
- **Degraded** (`cluster_semantic='degraded'`) — voltas mais lentas (degradação de pneu, tráfego)

---

### 1.1. K-Means (Agrupamento Rígido)

**Função:** `cluster_laps_kmeans()`

**Localização:** `src/ml/clustering.py`

#### Como Funciona

K-Means divide as voltas em **k grupos** (clusters) onde cada volta pertence ao cluster com centroide mais próximo.

**Centroide** = ponto médio do cluster = ritmo típico daquele modo

#### Quando Usar

- Você **sabe** quantos ritmos existem (ex: 3 = puro, gestão, tráfego)
- Clusters têm formas esféricas (bem distribuídos)
- Dataset grande e limpo

#### Parâmetros

- `n_clusters`: Número de clusters (se None, detecta automaticamente usando silhouette)
- `feature_columns`: Colunas para usar (ex: `['LapTime_seconds', 'Sector1Time_seconds']`)
- `group_by`: Agrupar por piloto (`'Driver'`) para analisar cada um separadamente

#### Exemplo Básico

```python
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats, scale_features
from src.ml.clustering import cluster_laps_kmeans
import pandas as pd

# Carregar e pré-processar
laps = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')
laps_processed = enrich_dataframe_with_stats(laps, value_column='LapTime_seconds')
laps_scaled = scale_features(laps_processed, ['LapTime_seconds', 'Sector1Time_seconds'])

# Clustering
laps_clustered = cluster_laps_kmeans(
    laps_scaled,
    feature_columns=['LapTime_seconds', 'Sector1Time_seconds'],
    n_clusters=3,  # 3 ritmos
    group_by='Driver'
)

# Analisar clusters
for driver in laps_clustered['Driver'].unique():
    driver_laps = laps_clustered[laps_clustered['Driver'] == driver]
    print(f"\n{driver}:")
    print(driver_laps.groupby('cluster_label')['LapTime_seconds'].agg(['mean', 'count']))
```

**Saída esperada:**
```
VER:
cluster_label  cluster_semantic  mean      count
0              push              89.234    15
1              base              91.456    8
2              degraded          93.789    3
```

#### Encontrar k Ótimo

**Função:** `find_optimal_k()`

```python
from src.ml.clustering import find_optimal_k

optimal_k = find_optimal_k(
    laps_scaled[['LapTime_seconds', 'Sector1Time_seconds']],
    k_range=range(2, 6),
    method='silhouette'  # ou 'elbow'
)

print(f"Número ótimo de clusters: {optimal_k}")
```

**Métodos:**
- `silhouette`: Usa silhouette score (mede coesão intra-cluster e separação inter-cluster)
- `elbow`: Método do cotovelo (busca "joelho" na curva de inércia)

#### Colunas Adicionadas

- `cluster_label`: Label do cluster com semântica fixa: `0=push`, `1=base`, `2=degraded`
- `cluster_semantic`: Label textual (`'push'`, `'base'`, `'degraded'`) — contrato para downstream (Pydantic, LLM)
- `cluster_semantic_clean`: `True` quando `push_delta < base_delta < degraded_delta` (separação semântica válida). `False` indica caso degenerado — o piloto não teve um cluster de push genuinamente mais rápido, ou o degraded é mais rápido que o base
- `cluster_centroid_distance`: Distância ao centroide (quanto menor, mais típico do cluster)

#### Lógica de Semântica Determinística (`normalize_cluster_semantics`)

A numeração do K-Means é arbitrária. A função `normalize_cluster_semantics()` aplica lógica determinística por piloto:

1. **Base** → cluster com mais voltas (ritmo dominante da corrida). Desempate: menor `abs(delta_mean)` — cluster mais próximo da mediana
2. **Push** → entre os restantes, menor `LapTime_delta_mean` (mais rápido)
3. **Degraded** → entre os restantes, maior `LapTime_delta_mean` (mais lento)

Isso garante que `cluster_label=1` é sempre base pace para todos os pilotos, independente do índice original atribuído pelo K-Means.

---

### 1.2. DBSCAN (Densidade e Ruído)

**Função:** `cluster_laps_dbscan()`

**Localização:** `src/ml/clustering.py`

#### Como Funciona

DBSCAN (Density-Based Spatial Clustering) identifica clusters baseado em **densidade de pontos**.

**Conceitos:**
- **Core point**: Ponto com pelo menos `min_samples` vizinhos dentro de `eps`
- **Border point**: Vizinho de core point
- **Noise**: Pontos isolados (label = -1)

#### Quando Usar

- Você **não sabe** quantos ritmos existem
- Clusters têm formas irregulares (não esferas)
- Quer identificar **outliers automaticamente** como ruído
- Dataset com voltas atípicas (tráfego, erros)

#### Parâmetros

- `eps`: Distância máxima entre voltas para serem vizinhas (se None, estima automaticamente)
- `min_samples`: Mínimo de voltas para formar um cluster (padrão: 3)
- `feature_columns`: Colunas para usar
- `group_by`: Agrupar por piloto

#### Exemplo Básico

```python
from src.ml.clustering import cluster_laps_dbscan

laps_clustered = cluster_laps_dbscan(
    laps_scaled,
    feature_columns=['LapTime_seconds', 'Sector1Time_seconds'],
    eps=None,  # Auto-detecta
    min_samples=5,
    group_by='Driver'
)

# Filtrar ruído
clean_laps = laps_clustered[~laps_clustered['is_noise']]
noise_laps = laps_clustered[laps_clustered['is_noise']]

print(f"Voltas válidas: {len(clean_laps)}")
print(f"Ruído: {len(noise_laps)}")
print(f"Número de clusters: {clean_laps['cluster_label'].nunique()}")
```

**Saída esperada:**
```
Voltas válidas: 52
Ruído: 5
Número de clusters: 2
```

#### Colunas Adicionadas

- `cluster_label`: Label do cluster (-1 = ruído, 0+ = cluster válido)
- `is_noise`: Flag binária (True = ruído/outlier)

---

### Comparação: K-Means vs DBSCAN

| Característica     | K-Means                          | DBSCAN                           |
|--------------------|----------------------------------|----------------------------------|
| Número de clusters | Precisa especificar              | Detecta automaticamente          |
| Forma dos clusters | Esferas (distâncias euclidianas) | Formas arbitrárias (densidade)   |
| Outliers           | Força a pertencer a algum cluster| Marca como ruído (label -1)      |
| Velocidade         | Rápido (O(n))                    | Mais lento (O(n log n))          |
| Quando usar        | Dataset limpo, k conhecido       | Dataset ruidoso, k desconhecido  |

**Recomendação:**
- **K-Means**: Análise de stints limpos, ritmo de piloto específico
- **DBSCAN**: Análise exploratória, corridas com tráfego/safety car

---

## 2. Detecção de Anomalias

### Objetivo

Detectar **eventos pontuais raros**:
- Erros de piloto (rodada, saída de pista)
- Falhas mecânicas (quebra súbita)
- Voltas excepcionalmente rápidas (pole lap)
- Tráfego extremo

---

### 2.1. Isolation Forest

**Função:** `detect_anomalies_isolation_forest()`

**Localização:** `src/ml/anomaly_detection.py`

#### Como Funciona

Isolation Forest **isola** observações selecionando aleatoriamente uma feature e um valor de divisão.

**Ideia:** Anomalias são mais fáceis de isolar (caminhos curtos na árvore) do que pontos normais (caminhos longos).

**Matematicamente:** Cria árvores de decisão aleatórias e mede o comprimento médio do caminho até isolar cada ponto.

#### Parâmetros

- `feature_columns`: Colunas para detectar anomalias (ex: `['LapTime_seconds', 'Sector1Time_seconds']`)
- `contamination`: Proporção esperada de anomalias (padrão: 0.05 = 5%)
- `group_by`: Agrupar por piloto
- `return_scores`: Se True, retorna anomaly score (quanto mais negativo, mais anômalo)

#### Exemplo Básico

```python
from src.ml.anomaly_detection import detect_anomalies_isolation_forest

laps_anomalies = detect_anomalies_isolation_forest(
    laps_scaled,
    feature_columns=['LapTime_seconds', 'Sector1Time_seconds'],
    contamination=0.05,  # Espera 5% de anomalias
    group_by='Driver',
    return_scores=True
)

# Ver anomalias
anomalies = laps_anomalies[laps_anomalies['is_anomaly']]
print(f"\nAnomalias detectadas: {len(anomalies)}")
print(anomalies[['Driver', 'LapNumber', 'LapTime_seconds', 'anomaly_score']].sort_values('anomaly_score'))
```

**Saída esperada:**
```
Anomalias detectadas: 4

   Driver  LapNumber  LapTime_seconds  anomaly_score
   VER     14         102.567         -0.234        (Rodada)
   HAM     8          95.123          -0.189        (Tráfego)
   LEC     23         87.234          -0.156        (Volta excepcional)
   PER     19         DNF             -0.298        (Quebra)
```

#### Ajustar Sensibilidade (contamination)

```python
# Mais sensível (detecta mais anomalias)
laps_anomalies = detect_anomalies_isolation_forest(..., contamination=0.10)

# Menos sensível (apenas anomalias extremas)
laps_anomalies = detect_anomalies_isolation_forest(..., contamination=0.02)
```

**Recomendação:**
- Corrida limpa: `contamination=0.03` (3%)
- Corrida com tráfego/safety car: `contamination=0.10` (10%)

#### Colunas Adicionadas

- `is_anomaly`: Flag binária (True = anomalia)
- `anomaly_score`: Score (valores negativos = anômalos, positivos = normais)

#### Sumarizar Anomalias

**Função:** `summarize_anomalies()`

```python
from src.ml.anomaly_detection import summarize_anomalies

summary = summarize_anomalies(laps_anomalies, group_by='Driver')
print(summary)
```

**Saída:**
```
  Driver  total_laps  anomalies_count  anomaly_rate  anomaly_laps
  VER     55          2                3.64%         [14, 23]
  HAM     54          1                1.85%         [8]
  LEC     56          3                5.36%         [12, 23, 45]
```

---

## 3. Pipeline Completo de ML

### Função de Alto Nível

**Função:** `run_race_analysis()`

**Localização:** `src/ml/pipeline.py`

#### O Que Faz

Executa pipeline completo end-to-end:
1. Pré-processamento (imputação, encoding, escalonamento)
2. Clustering (K-Means)
3. Detecção de Anomalias (Isolation Forest)

#### Exemplo de Uso

```python
from src.ml.pipeline import run_race_analysis
import pandas as pd

# Carregar dados brutos
laps_df = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')

# Executar análise completa
results = run_race_analysis(
    laps_df,
    analysis_type='all',  # ou 'clustering' ou 'anomaly'
    driver=None  # ou 'VER' para analisar apenas Verstappen
)

# Ver resultados
print(results['summary'])
print("\n=== CLUSTERS ===")
print(results['laps_clustered'].groupby(['Driver', 'cluster_label'])['LapTime_seconds'].mean())
print("\n=== ANOMALIAS ===")
anomalies = results['laps_anomalies'][results['laps_anomalies']['is_anomaly']]
print(anomalies[['Driver', 'LapNumber', 'LapTime_seconds']])
```

---

### Pipeline Scikit-learn (Avançado)

**Função:** `create_ml_pipeline()`

Para criar um `sklearn.pipeline.Pipeline` completo:

```python
from src.ml.pipeline import create_ml_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

# Criar pipeline de pré-processamento
preprocessor = create_ml_pipeline(
    numeric_columns=['LapTime_seconds', 'TyreLife'],
    categorical_columns=['Compound'],
    scaler_type='robust'
)

# Criar pipeline completo
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('detector', IsolationForest(contamination=0.05))
])

# Treinar e predizer em um único comando
predictions = full_pipeline.fit_predict(laps_df)
laps_df['is_anomaly'] = predictions == -1
```

---

## 4. Métricas de Avaliação

### 4.0. Estatísticas por Cluster

**Função:** `calculate_cluster_statistics()`

Retorna um DataFrame com estatísticas agregadas por cluster, incluindo as novas colunas semânticas:

```python
from src.ml.metrics import calculate_cluster_statistics

stats = calculate_cluster_statistics(
    laps_clustered,
    cluster_column='cluster_label',
    feature_columns=['LapTime_delta', 'TyreAge_normalized'],
    driver_column='Driver',  # agrupa por piloto → cluster_size_pct por piloto
)
# Colunas: Driver | cluster_label | cluster_semantic | size | cluster_size_pct | LapTime_delta_mean | ...
```

| Coluna | Descrição |
|--------|-----------|
| `cluster_label` | ID do cluster (0=push, 1=base, 2=degraded) |
| `cluster_semantic` | Label textual (quando `cluster_semantic` existe no df) |
| `size` | Número de voltas no cluster |
| `cluster_size_pct` | % das voltas do piloto nesse cluster (com `driver_column`) |

---

### 4.1. Métricas de Clustering

**Módulo:** `src/ml/metrics.py`

#### Métricas Disponíveis

| Métrica | Range | Melhor | Descrição |
|---------|-------|--------|-----------|
| **Silhouette Score** | [-1, 1] | Maior | Coesão vs separação dos clusters |
| **Davies-Bouldin Index** | [0, ∞) | Menor | Compacidade vs separação |
| **Calinski-Harabasz** | [0, ∞) | Maior | Ratio between/within variance |
| **Inércia** | [0, ∞) | Menor | Distância total aos centróides (K-Means) |

#### Interpretação do Silhouette Score

- **0.7 - 1.0**: Estrutura forte, clusters bem separados
- **0.5 - 0.7**: Estrutura razoável, clustering aceitável
- **0.25 - 0.5**: Estrutura fraca, clusters se sobrepõem
- **< 0.25**: Sem estrutura significativa

#### Uso

```python
from src.ml.metrics import calculate_clustering_metrics, evaluate_clustering_quality

# Calcular métricas
metrics = calculate_clustering_metrics(X, labels, remove_noise=True)
print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
print(f"Davies-Bouldin: {metrics['davies_bouldin_score']:.3f}")
print(f"Calinski-Harabasz: {metrics['calinski_harabasz_score']:.1f}")

# Avaliar qualidade
evaluation = evaluate_clustering_quality(metrics)
print(f"Qualidade: {evaluation['quality']}")  # excellent, good, fair, poor
print(f"Recomendação: {evaluation['recommendation']}")
```

---

### 4.2. Métricas de Anomaly Detection

**Módulo:** `src/ml/metrics.py`

#### Métricas Disponíveis

| Métrica | Descrição |
|---------|-----------|
| **n_anomalies** | Número total de anomalias detectadas |
| **anomaly_rate** | Taxa de anomalias (%) |
| **score_mean** | Média dos scores de anomalia |
| **anomaly_score_mean** | Média dos scores das anomalias |
| **normal_score_mean** | Média dos scores dos pontos normais |

#### Uso

```python
from src.ml.metrics import calculate_anomaly_metrics

# Calcular métricas
anomaly_metrics = calculate_anomaly_metrics(predictions, scores)
print(f"Anomalias: {anomaly_metrics['n_anomalies']} ({anomaly_metrics['anomaly_rate']:.2f}%)")
print(f"Score médio das anomalias: {anomaly_metrics['anomaly_score_mean']:.3f}")
```

---

## 5. MLFlow Tracking

### 5.1. Visão Geral

**MLFlow** é uma plataforma open-source para gerenciar o ciclo de vida de Machine Learning:
- **Tracking**: Registrar parâmetros, métricas e artefatos
- **Models**: Gerenciar e versionar modelos
- **Compare**: Comparar experimentos e encontrar melhor configuração

**Módulo:** `src/ml/tracking.py`

---

### 5.2. Setup Inicial

#### 1. Instalar dependência

```bash
uv sync  # MLFlow já está incluído nas dependências
```

#### 2. Configurar MLFlow

```python
from src.ml import setup_mlflow

# Configurar experimento
setup_mlflow(
    experiment_name="F1_2025_Round_01",
    tracking_uri="file:./mlruns",  # Local (padrão)
    enable_autolog=True  # Autolog do scikit-learn
)
```

#### 3. Iniciar MLFlow UI

```bash
mlflow ui
# Acesse: http://localhost:5000
```

---

### 5.3. O Que é Trackeado?

#### Parâmetros (Inputs)
- `n_clusters`: Número de clusters (K-Means)
- `contamination`: Taxa esperada de anomalias (Isolation Forest)
- `eps`, `min_samples`: Parâmetros do DBSCAN
- `random_state`: Seed de aleatoriedade
- `scaler_type`: Tipo de escalonamento usado

#### Métricas (Outputs)
- **Clustering**:
  - `silhouette_score`
  - `davies_bouldin_score`
  - `calinski_harabasz_score`
  - `n_clusters`, `n_samples`

- **Anomaly Detection**:
  - `n_anomalies`, `anomaly_rate`
  - `score_mean`, `score_std`
  - `anomaly_score_mean`

#### Artefatos
- Modelos treinados (`.pkl`)
- DataFrames de resultados (`.csv`, `.parquet`)

---

### 5.4. Tracking no Pipeline

```python
from src.ml import run_race_analysis
import pandas as pd

# Carregar dados
laps_df = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')

# Executar análise COM tracking
results = run_race_analysis(
    laps_df=laps_df,
    analysis_type='all',
    enable_mlflow=True,
    experiment_name='F1_2025_Round_01',
    run_name='Full_Analysis_AllDrivers',
)

# Ver resultados
print(f"MLFlow Run ID: {results['mlflow_run_id']}")
print(results['clustering_metrics'])
print(results['anomaly_metrics'])
```

---

### 5.5. CLI de Análise com MLFlow

```bash
# Análise completa com tracking
uv run python cli/ml_analysis.py --year 2025 --round 1 --mlflow --show-metrics

# Apenas clustering
uv run python cli/ml_analysis.py --year 2025 --round 1 --clustering --mlflow

# Piloto específico
uv run python cli/ml_analysis.py --year 2025 --round 1 --driver VER --mlflow --save

# Comparar runs anteriores
uv run python cli/ml_analysis.py --compare --experiment "F1_2025_Round_01" --max-runs 5
```

---

### 5.6. Comparar Experimentos

```python
from src.ml import compare_runs, get_best_run

# Comparar últimos 10 runs
comparison = compare_runs(
    experiment_name="F1_2025_Round_01",
    metric_names=['silhouette_score', 'davies_bouldin_score'],
    max_runs=10
)
print(comparison)

# Encontrar melhor run
best = get_best_run(
    experiment_name="F1_2025_Round_01",
    metric_name='silhouette_score',
    ascending=False  # Maior silhouette é melhor
)
print(f"Melhor run: {best['run_name']}")
print(f"Silhouette: {best['metrics']['silhouette_score']:.3f}")
print(f"Parâmetros: {best['params']}")
```

---

### 5.7. Tracking Manual (Avançado)

```python
from src.ml.tracking import track_clustering_run, track_anomaly_detection_run
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Clustering com tracking
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

run_id = track_clustering_run(
    run_name="KMeans_k3_RobustScaler",
    X=X,
    labels=labels,
    params={'n_clusters': 3, 'random_state': 42, 'scaler': 'robust'},
    model=kmeans,
    tags={'driver': 'VER', 'algorithm': 'kmeans'}
)

# Anomaly detection com tracking
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(X)
scores = iso_forest.score_samples(X)

run_id = track_anomaly_detection_run(
    run_name="IsolationForest_cont005",
    X=X,
    predictions=predictions,
    scores=scores,
    params={'contamination': 0.05, 'n_estimators': 100},
    model=iso_forest,
    tags={'session': 'Race', 'track': 'Bahrain'}
)
```

---

### 5.8. Fluxo de Trabalho Recomendado

#### 1. Experimentação Inicial (Sem MLFlow)

```bash
# Testar pipeline básico
uv run python cli/ml_analysis.py --year 2025 --round 1 --show-metrics
```

#### 2. Experimentação com Tracking

```bash
# Experimentar com diferentes configurações
uv run python cli/ml_analysis.py --year 2025 --round 1 --mlflow --save
```

#### 3. Análise de Resultados

```bash
# Iniciar MLFlow UI
mlflow ui

# Acesse http://localhost:5000
# Compare runs, visualize métricas, identifique melhor configuração
```

#### 4. Comparação Programática

```python
from src.ml import compare_runs, get_best_run

# Ver todos os runs
comparison = compare_runs("F1_2025_Round_01")
print(comparison)

# Encontrar melhor configuração
best = get_best_run("F1_2025_Round_01", "silhouette_score")
print(f"Melhor configuração: {best['params']}")
```

#### 5. Carregar Melhor Modelo

```python
import mlflow

# Carregar melhor modelo
best_run_id = best['run_id']
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

# Usar em produção
predictions = model.predict(new_data)
```

---

### 5.9. Interpretação de Resultados

#### Clustering (K-Means)

**Bom clustering**:
- Silhouette > 0.5
- Davies-Bouldin < 1.0
- Clusters fazem sentido no contexto F1 (ritmo puro, gestão, tráfego)

**Clustering ruim**:
- Silhouette < 0.25
- Davies-Bouldin > 2.0
- Considere: Ajustar features, tentar DBSCAN, revisar pré-processamento

#### Anomaly Detection

**Configuração adequada**:
- Taxa de anomalias compatível com eventos esperados (2-5%)
- Anomalias detectadas correspondem a eventos reais
- Scores das anomalias significativamente menores que dos normais

**Ajustes necessários**:
- Taxa muito alta (>10%): Reduzir `contamination`
- Taxa muito baixa (<1%): Aumentar `contamination`
- Anomalias não fazem sentido: Revisar features, adicionar domain knowledge

---

## 6. Casos de Uso Práticos

### Caso 1: Identificar Ritmo de Corrida Real

**Objetivo:** Filtrar voltas de tráfego/safety car para ver ritmo puro

```python
from src.ml.clustering import cluster_laps_dbscan

# Clustering DBSCAN (detecta ruído automaticamente)
laps_clustered = cluster_laps_dbscan(
    laps_scaled,
    feature_columns=['LapTime_seconds'],
    min_samples=5,
    group_by='Driver'
)

# Filtrar apenas cluster principal (maior cluster)
for driver in laps_clustered['Driver'].unique():
    driver_laps = laps_clustered[laps_clustered['Driver'] == driver]
    main_cluster = driver_laps[~driver_laps['is_noise']]
    avg_pace = main_cluster['LapTime_seconds'].mean()
    print(f"{driver}: Ritmo Real = {avg_pace:.3f}s")
```

---

### Caso 2: Detectar Mudança de Estratégia

**Objetivo:** Identificar quando piloto mudou de ritmo (puro → gestão)

```python
from src.ml.clustering import cluster_laps_kmeans

# K-Means com 2 clusters (puro vs gestão)
laps_clustered = cluster_laps_kmeans(
    laps_scaled,
    feature_columns=['LapTime_seconds'],
    n_clusters=2,
    group_by='Driver'
)

# Ver quando mudou de cluster
for driver in ['VER', 'HAM']:
    driver_laps = laps_clustered[laps_clustered['Driver'] == driver].sort_values('LapNumber')
    changes = driver_laps['cluster_label'].diff()
    change_laps = driver_laps[changes != 0]['LapNumber'].tolist()
    print(f"{driver} mudou estratégia nas voltas: {change_laps}")
```

---

### Caso 3: Encontrar Erros de Piloto

**Objetivo:** Detectar rodadas, saídas de pista, erros

```python
from src.ml.anomaly_detection import detect_anomalies_isolation_forest

# Detecção de anomalias multidimensional
laps_anomalies = detect_anomalies_isolation_forest(
    laps_scaled,
    feature_columns=['LapTime_seconds', 'Sector1Time_seconds', 'Sector2Time_seconds', 'Sector3Time_seconds'],
    contamination=0.05,
    group_by='Driver',
    return_scores=True
)

# Filtrar apenas anomalias LENTAS (erros, rodadas)
anomalies = laps_anomalies[laps_anomalies['is_anomaly']]
# Voltas anormalmente lentas
slow_anomalies = anomalies[anomalies['LapTime_seconds'] > anomalies['LapTime_seconds'].median()]
print(slow_anomalies[['Driver', 'LapNumber', 'LapTime_seconds']])
```

---

## 5. Resumo: Entradas e Saídas

| Etapa          | Ferramenta Scikit-learn      | Entrada (Features)                  | Saída                                    |
|----------------|------------------------------|-------------------------------------|------------------------------------------|
| **Pré-proc**   | `StandardScaler`             | Tempos de volta, Idade Pneu         | Dados na mesma escala (Z-score)          |
| **Ritmo**      | `KMeans` ou `DBSCAN`         | Tempo de Volta, Tempo Setores       | "Piloto X teve 2 ritmos: A e B"          |
| **Eventos**    | `IsolationForest`            | Delta Telemetria, Variação Vel.     | "Volta 14 foi uma anomalia (erro/quebra)"|

---

## 6. Performance

| Operação              | Tempo (1000 voltas) | Memória  |
|-----------------------|---------------------|----------|
| Imputação (Simple)    | <1s                 | ~5MB     |
| Imputação (KNN)       | ~3s                 | ~10MB    |
| Encoding              | <1s                 | ~2MB     |
| Escalonamento         | <1s                 | ~1MB     |
| K-Means (k=3)         | ~1s                 | ~5MB     |
| DBSCAN                | ~2s                 | ~8MB     |
| Isolation Forest      | ~2s                 | ~10MB    |
| **Pipeline Completo** | **~10s**            | **~40MB**|

---

## 7. Troubleshooting

### Erro: "Coluna não encontrada"
**Causa:** Features não existem no DataFrame
**Solução:** Verificar nomes das colunas após encoding

### Warning: "Found array with 0 sample(s)"
**Causa:** Grupo muito pequeno (menos de `min_samples`)
**Solução:** Aumentar dados ou diminuir `min_samples`

### K-Means não converge
**Causa:** Dados não escalonados ou k muito alto
**Solução:** Usar `scale_features()` antes e reduzir k

### DBSCAN retorna tudo como ruído
**Causa:** `eps` muito pequeno ou `min_samples` muito alto
**Solução:** Deixar `eps=None` para auto-detectar

### Isolation Forest detecta pouquíssimas anomalias
**Causa:** `contamination` muito baixo
**Solução:** Aumentar para 0.08-0.10

---

## 8. Próximos Passos

Após executar pipeline ML:
1. **Exportar eventos para JSON** - Estruturar saída (Pydantic)
2. **LLM (DSPY/Agno)** - Gerar narrativas a partir dos eventos
3. **Visualização** - Plotar clusters e anomalias
4. **API (FastAPI)** - Servir análises via REST

---

## Referências

- **Scikit-learn User Guide:** https://scikit-learn.org/stable/user_guide.html
- **K-Means:** https://scikit-learn.org/stable/modules/clustering.html#k-means
- **DBSCAN:** https://scikit-learn.org/stable/modules/clustering.html#dbscan
- **Isolation Forest:** https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest
- **Pipeline:** https://scikit-learn.org/stable/modules/compose.html
- **Código:** `src/ml/`
- **Pré-processamento:** [PREPROCESSING.md](PREPROCESSING.md)
