# Preprocessing Module - SciPy Layer

Módulo de pré-processamento matemático de dados de telemetria F1 usando **SciPy**. Este módulo transforma dados brutos e dessincronizados do FastF1 em sinais matemáticos limpos e comparáveis para análise de Machine Learning.

## 🎯 Objetivo

O SciPy atua como **motor matemático de pré-processamento e engenharia de features**, preparando dados para:
- **Scikit-learn**: modelos de clustering e detecção de anomalias
- **Ruptures**: detecção de pontos de mudança (degradação de pneus)

## 📦 Componentes

### 1. Sincronização de Telemetria (`interpolation.py`)

**Problema**: Dados de telemetria chegam dessincronizados - diferentes pilotos têm medições em pontos diferentes da pista.

**Solução**: Interpolação cúbica spline para criar um grid comum de distância.

**Ferramentas**:
- `scipy.interpolate.make_interp_spline`
- `scipy.interpolate.CubicSpline`

**Funções**:
```python
synchronize_telemetry(
    telemetry: pd.DataFrame,
    track_length: float,
    num_points: int = 5000,
) -> pd.DataFrame
```

**Exemplo**:
```python
from src.preprocessing.interpolation import synchronize_telemetry

# Sincronizar telemetria de um piloto
ver_telemetry = lap.get_telemetry()
synchronized = synchronize_telemetry(
    ver_telemetry,
    track_length=5281.0,  # Monaco
    num_points=5000
)

# Agora pode comparar diretamente com outros pilotos
ham_synchronized = synchronize_telemetry(ham_telemetry, 5281.0, 5000)
speed_delta = synchronized['Speed'] - ham_synchronized['Speed']
```

**Resultado**:
- Matriz onde cada linha = uma volta
- Cada coluna = um ponto exato da pista
- Pronta para comparações diretas e cálculos de delta

---

### 2. Tratamento de Sinal (`signal_processing.py`)

**Problema**: Sensores têm ruído - picos repentinos que não representam ações reais do piloto.

**Solução**: Filtros de processamento de sinais para suavizar curvas preservando informação importante.

**Ferramentas**:
- `scipy.signal.medfilt` - Remove outliers pontuais preservando bordas
- `scipy.signal.savgol_filter` - Suavização e cálculo de derivadas

**Funções**:
```python
clean_signal(
    signal: np.ndarray,
    method: str = "median",  # ou "savgol"
    kernel_size: int = 5,
) -> np.ndarray

calculate_derivative(
    signal: np.ndarray,
    delta_x: float = 1.0,
    smooth: bool = True,
) -> np.ndarray

apply_telemetry_pipeline(
    telemetry_dict: dict[str, np.ndarray],
    noise_reduction: bool = True,
    outlier_removal: bool = True,
    calculate_derivatives: bool = False,
) -> dict[str, np.ndarray]
```

**Exemplo**:
```python
from src.preprocessing.signal_processing import apply_telemetry_pipeline

telemetry = {
    'Speed': speed_array,
    'Throttle': throttle_array,
    'Brake': brake_array,
}

# Pipeline completo
processed = apply_telemetry_pipeline(
    telemetry,
    noise_reduction=True,      # Remove ruído
    outlier_removal=True,       # Remove spikes
    calculate_derivatives=True, # Calcula aceleração
)

# Resultado inclui:
# - 'Speed', 'Throttle', 'Brake' (limpos)
# - 'Speed_derivative' (aceleração)
# - 'Throttle_derivative', 'Brake_derivative'
```

**Resultado**:
- Dados "polidos" onde variações representam apenas física do carro
- Facilita detecção de anomalias reais pelo Isolation Forest

---

### 3. Engenharia de Features Estatísticas (`feature_engineering.py`)

**Problema**: Identificar outliers simples antes de usar IA complexa (ex: volta lenta por tráfego).

**Solução**: Estatística clássica para filtrar o óbvio e criar features descritivas.

**Ferramentas**:
- `scipy.stats.zscore` - Normalização e detecção de outliers
- `scipy.stats.describe` - Estatísticas descritivas
- `scipy.stats.linregress` - Taxa de degradação

**Funções**:
```python
calculate_statistical_features(
    df: pd.DataFrame,
    value_column: str = 'LapTime',
    group_by: list[str] | None = None,
) -> pd.DataFrame

calculate_degradation_rate(
    df: pd.DataFrame,
    lap_column: str = 'LapNumber',
    time_column: str = 'LapTime',
    group_by: list[str] | None = None,
) -> pd.DataFrame

enrich_dataframe_with_stats(
    df: pd.DataFrame,
    value_column: str = 'LapTime',
    group_by: list[str] | None = None,
    include_degradation: bool = True,
) -> pd.DataFrame
```

**Exemplo**:
```python
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats

# DataFrame com tempos de volta
laps_df = session.laps[['LapNumber', 'LapTime', 'Driver', 'Compound']]

# Adicionar features estatísticas
enriched = enrich_dataframe_with_stats(
    laps_df,
    value_column='LapTime',
    group_by=['Driver', 'Compound'],
    include_degradation=True
)

# Colunas adicionadas:
# - z_score: score normalizado
# - is_outlier: flag para |z| > 3
# - group_mean, group_std: estatísticas do grupo
# - degradation_slope: taxa de degradação (s/lap)
# - degradation_r_squared: qualidade do fit
# - degradation_intercept: tempo estimado primeira volta

# Filtrar outliers
clean_laps = enriched[~enriched['is_outlier']]

# Analisar degradação
for driver in enriched['Driver'].unique():
    driver_data = enriched[enriched['Driver'] == driver].iloc[0]
    print(f"{driver}: {driver_data['degradation_slope']:.3f}s/lap")
```

**Resultado**:
- DataFrame com flags de outliers
- Features de degradação para Pydantic
- Entrada limpa para Scikit-learn

---

## 🔄 Fluxo Completo

```python
# 1. Entrada: Dados brutos do FastF1
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()
laps = session.laps.pick_driver('VER')

# 2. Processamento SciPy

# 2.1 - Interpolação: Sincronizar telemetria
from src.preprocessing.interpolation import synchronize_telemetry
synchronized = synchronize_telemetry(
    telemetry,
    track_length=session.get_circuit_info().total_distance
)

# 2.2 - Signal: Limpar ruído
from src.preprocessing.signal_processing import clean_signal
clean_speed = clean_signal(synchronized['Speed'], method="median")

# 2.3 - Stats: Calcular features
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats
enriched = enrich_dataframe_with_stats(
    laps_df,
    group_by=['Stint', 'Compound']
)

# 3. Saída: DataFrame "Enriched" e limpo

# 4. Próximos passos:
# - Ruptures: detectar pontos de mudança
# - Scikit-learn: clustering por estratégia
```

## 📊 Quando Usar Cada Módulo

| Módulo | Quando Usar |
|--------|-------------|
| **interpolation** | Comparar pilotos, calcular deltas, criar matriz de voltas |
| **signal_processing** | Remover ruído de sensores, calcular aceleração/derivadas |
| **feature_engineering** | Detectar outliers, calcular degradação, preparar para ML |

## 🧪 Testes

Todos os módulos possuem testes unitários completos:

```bash
# Rodar todos os testes
uv run pytest tests/preprocessing/ -v

# Rodar testes de um módulo específico
uv run pytest tests/preprocessing/test_interpolation.py -v
uv run pytest tests/preprocessing/test_signal_processing.py -v
uv run pytest tests/preprocessing/test_feature_engineering.py -v
```

## 📖 Exemplos Práticos

Veja `examples/preprocessing_example.py` para exemplos completos de:
1. Sincronização de telemetria para comparação de pilotos
2. Processamento de sinal e cálculo de derivadas
3. Engenharia de features para análise de stint
4. Pipeline completo para ML

```bash
# Rodar exemplos
uv run python examples/preprocessing_example.py
```

## 🔗 Integração com Outras Camadas

```
FastF1 (extração)
    ↓
SciPy (pré-processamento)
    ↓
├─→ Scikit-learn (clustering, anomalias)
├─→ Ruptures (pontos de mudança)
└─→ Pydantic (validação de features)
```

## 📚 Referências

- Documentação SciPy Interpolate: https://docs.scipy.org/doc/scipy/reference/interpolate.html
- Documentação SciPy Signal: https://docs.scipy.org/doc/scipy/reference/signal.html
- Documentação SciPy Stats: https://docs.scipy.org/doc/scipy/reference/stats.html
