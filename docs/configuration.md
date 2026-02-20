# Guia de Configuração - PitWall AI

Todos os parâmetros do pipeline estão centralizados no arquivo `config.yaml` na raiz do projeto. Isso permite customizar o comportamento do sistema sem modificar código.

## 📋 Estrutura do config.yaml

### Diretórios de Dados

```yaml
data:
  raw:
    base_dir: "data/raw"
    races_dir: "data/raw/races"
    calendar_dir: "data/raw/calendar"
  processed:
    base_dir: "data/processed"
  ml:
    races_dir: "data/ml/races"
```

**Customização**: Altere esses caminhos se quiser armazenar dados em outros diretórios.

---

### Pré-processamento

```yaml
preprocessing:
  interpolation:
    num_points: 5000  # Pontos no grid de sincronização de telemetria

  signal_processing:
    median_filter_kernel_size: 5    # Tamanho do kernel para filtro mediano
    savgol_kernel_size: 11          # Tamanho do kernel Savitzky-Golay
    savgol_polyorder: 3             # Ordem polinomial do Savitzky-Golay
    outlier_threshold: 3.0          # Desvios padrão para outliers

  imputation:
    strategy: "median"  # Estratégia de imputação: mean, median, mode
    use_knn: false      # Usar KNNImputer (mais lento, mais preciso)

  encoding:
    drop_first: true    # Evitar multicolinearidade em one-hot encoding

  scaling:
    type: "robust"      # Tipo de scaler: standard, minmax, robust
```

**Customização**:
- `num_points`: Aumente para maior resolução (mais lento), diminua para menor (mais rápido)
- `median_filter_kernel_size`: Kernel maior = mais suavização
- `outlier_threshold`: Valor menor = mais sensível a outliers
- `scaling.type`: Use `robust` para dados com outliers, `standard` para dados normais

---

### Machine Learning

```yaml
ml:
  random_state: 42  # Seed para reprodutibilidade

  clustering:
    algorithm: "KMeans"
    n_clusters: 3
    k_range_min: 2    # Mínimo de clusters para análise
    k_range_max: 6    # Máximo de clusters para análise

  dbscan:
    min_samples: 3    # Mínimo de amostras para formar cluster
    eps: 0.5          # Raio de vizinhança

  anomaly:
    contamination: 0.05     # Proporção esperada de anomalias (5%)
    n_estimators: 100       # Número de árvores no Isolation Forest

    # Perfis de contaminação por tipo de corrida
    contamination_profiles:
      clean: 0.03     # Corrida limpa, sem incidentes (3%)
      normal: 0.05    # Corrida padrão (5%)
      chaotic: 0.10   # Corrida com muitos safety cars/incidentes (10%)
```

**Customização**:
- `random_state`: Mude para obter diferentes resultados (ou mantenha 42 para reprodutibilidade)
- `contamination`: Aumente se espera mais anomalias, diminua se espera menos
- `k_range_min/max`: Ajuste o range de clusters a testar
- Use `contamination_profiles` para diferentes tipos de corrida

---

### Change Point Detection (Ruptures/PELT)

```yaml
ml:
  degradation:
    algorithm: "Pelt"
    model: "l2"               # CostL2: detecta mudança de média (tire cliff)
    penalty: 3                # Sensibilidade — menor = mais cliffs, maior = menos
    min_size: 3               # Mínimo de voltas entre dois breakpoints
    jump: 1                   # Step do grid de busca
    min_cliff_magnitude: 0.3  # Segundos — magnitude mínima para cliff real
    penalty_search_range: [1, 2, 3, 5, 8, 13, 21]  # Range para calibração
    validation:
      enabled: true
      window_laps: 5          # Voltas antes do cliff para calcular slope
      slope_threshold: 0.05   # Slope mínimo positivo (s/volta) para validar cliff
```

**Customização**:
- `penalty`: Parâmetro principal. Calibre com `cli/ruptures_analysis.py --penalty-search`, escolha o melhor valor no MLFlow UI e sete aqui
- `min_cliff_magnitude`: Valores menores detectam cliffs sutis; maiores filtram falsos positivos
- `penalty_search_range`: Range de valores a testar durante calibração

---

### MLFlow

```yaml
mlflow:
  enabled: true                 # true = tracking em toda execução do pipeline
  tracking_uri: "file:./mlruns" # Local de armazenamento dos runs
  experiment_prefix: "F1"       # Experimento: F1_{year}_Round_{round:02d}
```

**Customização**:
- `enabled: false`: Desabilitar tracking sem mudar código
- `tracking_uri`: Apontar para servidor MLFlow remoto (ex: `http://mlflow-server:5000`)
- Visualizar runs: `uv run mlflow ui` → http://localhost:5000

---

### Cache e Extração

```yaml
cache:
  enabled: true
  dir: "~/.cache/fastf1"

extraction:
  timeout: 300  # Timeout em segundos

  polling:
    max_retries: 10
    retry_interval: 300  # 5 minutos
```

**Customização**:
- `cache.dir`: Mude se quiser cache em outro local
- `extraction.timeout`: Aumente para conexões lentas
- `polling.max_retries`: Aumente para aguardar mais tempo por dados recentes

---

## 🔧 Uso da Configuração no Código

### Acessar Configurações

```python
from src.utils.config import get_config

config = get_config()

# Acessar valores
num_points = config.get_num_points()
contamination = config.get_contamination(profile='normal')
random_state = config.get_random_state()

# Acessar qualquer valor com notação de pontos
value = config.get('preprocessing.interpolation.num_points')
```

### Métodos Disponíveis

**Pré-processamento**:
- `get_num_points()` - Pontos de interpolação
- `get_median_filter_kernel_size()` - Tamanho kernel filtro mediano
- `get_savgol_kernel_size()` - Tamanho kernel Savitzky-Golay
- `get_savgol_polyorder()` - Ordem polinomial Savitzky-Golay
- `get_outlier_threshold()` - Threshold de outliers
- `get_imputation_strategy()` - Estratégia de imputação
- `get_scaling_type()` - Tipo de escalonamento

**Machine Learning**:
- `get_random_state()` - Seed de reprodutibilidade
- `get_contamination(profile='normal')` - Contaminação para anomaly detection
- `get_n_estimators()` - Número de estimadores
- `get_k_range_min()` - Mínimo de clusters
- `get_k_range_max()` - Máximo de clusters
- `get_dbscan_min_samples()` - Min samples DBSCAN
- `get_dbscan_eps()` - Epsilon DBSCAN

**Change Point Detection (Ruptures)**:
- `get_ruptures_penalty()` - Penalty do algoritmo PELT
- `get_ruptures_min_size()` - Mínimo de voltas entre breakpoints
- `get_ruptures_min_cliff_magnitude()` - Magnitude mínima para cliff válido
- `get_ruptures_penalty_search_range()` - Range de penalties para calibração
- `get_ruptures_validation_enabled()` - Habilitar validação por slope
- `get_ruptures_validation_window()` - Janela de voltas para calcular slope
- `get_ruptures_validation_slope_threshold()` - Slope mínimo para validar cliff

**MLFlow**:
- `get_mlflow_enabled()` - Se tracking está habilitado
- `get_mlflow_tracking_uri()` - URI do servidor MLFlow
- `get_mlflow_experiment_prefix()` - Prefixo do nome do experimento

---

## 📊 Exemplos de Customização

### Exemplo 1: Telemetria de Alta Resolução

```yaml
preprocessing:
  interpolation:
    num_points: 10000  # Dobrar resolução (padrão: 5000)
  signal_processing:
    median_filter_kernel_size: 3  # Menos suavização
```

### Exemplo 2: Corrida Caótica (Muitos Incidentes)

```yaml
ml:
  anomaly:
    contamination: 0.10  # Esperar 10% de anomalias
```

Ou use o perfil no código:
```python
from src.ml.anomaly_detection import detect_anomalies_isolation_forest

anomalies = detect_anomalies_isolation_forest(
    df,
    feature_columns=['LapTime_seconds'],
    contamination_profile='chaotic'  # Usa 0.10
)
```

### Exemplo 3: Pipeline Mais Rápido

```yaml
preprocessing:
  interpolation:
    num_points: 2500  # Metade da resolução
ml:
  anomaly:
    n_estimators: 50  # Menos árvores (padrão: 100)
```

---

## ⚠️ Avisos Importantes

1. **Modificações no config.yaml afetam TODOS os pipelines**
   - As mudanças são globais para o projeto
   - Considere fazer backup antes de grandes mudanças

2. **Valores padrão são sensatos**
   - Os valores padrão foram testados e funcionam bem para a maioria dos casos
   - Mude apenas se souber o que está fazendo

3. **Cache vs Performance**
   - `num_points` maior = mais memória e tempo de processamento
   - `n_estimators` maior = melhor qualidade mas mais lento

4. **Reprodutibilidade**
   - Mantenha `random_state: 42` para resultados reproduzíveis
   - Mude se quiser explorar variações aleatórias

---

## 🔗 Documentação Relacionada

- [README.md](../README.md) - Visão geral do projeto
- [src/preprocessing/README.md](../src/preprocessing/README.md) - Pré-processamento
- [src/ml/README.md](../src/ml/README.md) - Machine Learning
- [cli/README.md](../cli/README.md) - CLIs disponíveis
