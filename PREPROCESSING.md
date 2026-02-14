# Guia de Pré-processamento - PitWall AI

Documentação completa do pré-processamento de **TODOS os dados** de corridas F1 usando NumPy, Pandas e SciPy.

## Visão Geral

O módulo de pré-processamento transforma **5 tipos de dados brutos** em features matemáticas limpas e estruturadas, prontas para análise ML:

1. **Laps** - Voltas e estratégia
2. **Telemetria** - Dados do carro
3. **Race Control** - Eventos da corrida
4. **Weather** - Condições meteorológicas
5. **Results** - Classificação final

**Bibliotecas utilizadas:**
- **NumPy** - Cálculos vetoriais e arrays
- **Pandas** - Manipulação de DataFrames
- **SciPy** - Ferramentas científicas especializadas:
  - `scipy.interpolate` - Sincronização de telemetria
  - `scipy.signal` - Processamento de sinal e remoção de ruído
  - `scipy.stats` - Estatísticas e features

---

## Instalação

```bash
uv sync  # NumPy, Pandas e SciPy já incluídos
```

---

## Uso Básico

### Pipeline Completo (Recomendado)

```bash
# Extração + Pré-processamento de TUDO em um comando
uv run python cli/pipeline.py 2025 1

# Ver amostras dos dados processados
uv run python cli/pipeline.py 2025 1 --show-sample
```

### Pré-processamento Individual

```bash
# Apenas pré-processar (dados já extraídos)
uv run python cli/preprocess.py --year 2025 --round 1 --all --save

# Apenas laps
uv run python cli/preprocess.py --year 2025 --round 1 --laps --save

# Apenas telemetria
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --save
```

---

## 1. Pré-processamento de LAPS

### Dados Brutos (Entrada)

**Arquivo:** `data/raw/races/YEAR/round_XX/laps.parquet`

**Estrutura:**
```
Driver | LapNumber | LapTime_seconds | Compound | TyreLife | Stint
-------|-----------|-----------------|----------|----------|------
VER    | 1         | 90.234          | SOFT     | 1        | 1
VER    | 2         | 89.987          | SOFT     | 2        | 1
HAM    | 1         | 90.456          | MEDIUM   | 1        | 1
```

### O Que é Feito

**Função:** `enrich_dataframe_with_stats()`

**Localização:** `src/preprocessing/feature_engineering.py`

#### 1. Features Estatísticas (Z-score)
```python
# Calcular Z-score por grupo (Driver, Compound)
df['z_score'] = df.groupby(['Driver', 'Compound'])['LapTime_seconds'].transform(
    lambda x: stats.zscore(x, nan_policy='omit')
)

# Detectar outliers (|z| > 3)
df['is_outlier'] = np.abs(df['z_score']) > 3
```

**Por quê:** Identifica voltas anormais (tráfego, safety car, erro de piloto)

#### 2. Taxa de Degradação de Pneu
```python
# Regressão linear: LapTime vs LapNumber
slope, intercept, r_value, p_value, std_err = stats.linregress(lap_numbers, lap_times)

df['degradation_slope'] = slope      # segundos/volta
df['degradation_r_squared'] = r_value ** 2
```

**Por quê:** Quantifica quanto o pneu degrada por volta

**Interpretação:**
- `+0.5 s/volta` = Pneu degradando (tempos piorando)
- `-0.5 s/volta` = Tempos melhorando (pista secando, combustível queimando)
- `R² > 0.4` = Degradação linear clara

#### 3. Estatísticas Descritivas
```python
stats_dict = {
    'nobs': número de observações,
    'mean': média,
    'variance': variância,
    'skewness': assimetria,
    'kurtosis': curtose,
    'min': mínimo,
    'max': máximo
}
```

**Por quê:** Contexto estatístico para análise ML

### Dados Processados (Saída)

**Arquivo:** `data/processed/races/YEAR/round_XX/laps_processed.parquet`

**Novas colunas:**
- `z_score` - Score padronizado
- `is_outlier` - Flag de outlier (True/False)
- `group_mean`, `group_std` - Estatísticas do grupo
- `degradation_slope` - Taxa de degradação (s/volta)
- `degradation_r_squared` - Qualidade do ajuste
- `group_nobs`, `group_skewness`, `group_kurtosis` - Estatísticas descritivas

### Exemplo de Uso

```python
import pandas as pd

laps = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')

# Filtrar outliers
clean_laps = laps[~laps['is_outlier']]

# Analisar degradação por piloto
for driver in clean_laps['Driver'].unique():
    driver_data = clean_laps[clean_laps['Driver'] == driver]
    print(f"{driver}: {driver_data['degradation_slope'].iloc[0]:.3f} s/volta")
```

---

## 2. Pré-processamento de TELEMETRIA

### Dados Brutos (Entrada)

**Arquivo:** `data/raw/races/YEAR/round_XX/telemetry/VER.parquet`

**Estrutura:**
```
Time_seconds | Speed | RPM   | Throttle | Brake | nGear | DRS  | Distance
-------------|-------|-------|----------|-------|-------|------|----------
0.0          | 285   | 11200 | 100      | False | 8     | True | 0.0
0.01         | 287   | 11350 | 100      | False | 8     | True | 2.9
```

**Problemas:**
- Frequências diferentes entre pilotos
- Ruído de sensores
- Não sincronizados (difícil comparar)

### O Que é Feito

#### Passo 1: Sincronização (scipy.interpolate)

**Função:** `synchronize_telemetry()`

**Localização:** `src/preprocessing/interpolation.py`

```python
# Criar grid comum de distância
dist_grid = np.linspace(0, track_length, num=1000)

# Interpolar cada canal usando spline cúbica
spline = make_interp_spline(original_distance, channel_values, k=3)
interpolated = spline(dist_grid)
```

**Por quê:** Alinha dados de diferentes pilotos em grid comum, permitindo comparações diretas.

**Antes:**
```
VER: [0m, 2.9m, 5.7m, ...]  (pontos irregulares)
HAM: [0m, 3.1m, 6.2m, ...]  (pontos diferentes)
```

**Depois:**
```
VER: [0m, 5.28m, 10.56m, ...]  (grid uniforme)
HAM: [0m, 5.28m, 10.56m, ...]  (mesmo grid!)
```

#### Passo 2: Limpeza de Ruído (scipy.signal)

**Função:** `apply_telemetry_pipeline()`

**Localização:** `src/preprocessing/signal_processing.py`

```python
# Remover outliers
cleaned, outlier_mask = remove_outliers(signal, threshold=3.0, method='median')

# Filtro mediano (remove spikes)
cleaned = medfilt(signal, kernel_size=5)

# Filtro Savitzky-Golay (suavização)
cleaned = savgol_filter(signal, kernel_size, polyorder)
```

**Por quê:** Sensores têm ruído. Filtros preservam informação importante (pontos de frenagem) removendo spikes.

#### Passo 3: Calcular Derivadas

```python
# Aceleração a partir de velocidade
df['Speed_derivative'] = calculate_derivative(speed, delta_x=distance_step)

# Taxa de mudança do acelerador
df['Throttle_derivative'] = calculate_derivative(throttle)
```

**Por quê:** Derivadas revelam aceleração, jerk, zonas de frenagem.

### Dados Processados (Saída)

**Arquivo:** `data/processed/races/YEAR/round_XX/telemetry/VER_processed.parquet`

**Novas colunas:**
- `Speed_derivative` - Aceleração (km/h/s)
- `Throttle_derivative` - Taxa de mudança do acelerador
- `Brake_derivative` - Taxa de mudança do freio
- Todos os canais sincronizados em 1000 pontos
- Ruído removido, outliers corrigidos

### Exemplo de Uso

```python
import pandas as pd

# Carregar telemetria processada
ver = pd.read_parquet('data/processed/races/2025/round_01/telemetry/VER_processed.parquet')
ham = pd.read_parquet('data/processed/races/2025/round_01/telemetry/HAM_processed.parquet')

# Comparar velocidades (sincronizadas!)
speed_delta = ver['Speed'] - ham['Speed']
print(f"Vantagem máxima VER: {speed_delta.max():.2f} km/h")

# Analisar aceleração
acceleration = ver['Speed_derivative']
print(f"Aceleração máxima: {acceleration.max():.2f} km/h/s")
```

---

## 3. Pré-processamento de RACE CONTROL

### Dados Brutos (Entrada)

**Arquivo:** `data/raw/races/YEAR/round_XX/race_control.parquet`

**Estrutura:**
```
Time             | Category   | Message                    | Status
-----------------|------------|----------------------------|--------
00:10:23.456     | Flag       | YELLOW FLAG - TURN 1       | None
00:15:30.123     | SafetyCar  | SAFETY CAR DEPLOYED        | DEPLOYED
00:20:45.789     | Other      | CAR 44 - INVESTIGATION     | None
```

**Problemas:**
- Dados são texto livre (difícil analisar)
- Falta severidade do evento
- Timestamp em formato variado

### O Que é Feito

**Função:** `preprocess_race_control()`

**Localização:** `src/preprocessing/feature_engineering.py`

#### 1. Normalizar Timestamps
```python
df['time_seconds'] = df['Time'].dt.total_seconds()
```

#### 2. Criar Indicadores Binários
```python
# Safety Car
df['is_safety_car'] = df.apply(
    lambda row: any(kw in str(row['message']).upper()
                   for kw in ['SAFETY CAR', 'VSC', 'SC DEPLOYED']),
    axis=1
)

# Bandeiras
df['is_flag'] = df.apply(
    lambda row: any(kw in str(row['message']).upper()
                   for kw in ['YELLOW FLAG', 'RED FLAG', 'FLAG']),
    axis=1
)

# Penalidades
df['is_penalty'] = df.apply(
    lambda row: any(kw in str(row['message']).upper()
                   for kw in ['PENALTY', 'TIME PENALTY']),
    axis=1
)
```

**Por quê:** Transforma texto em flags 0/1 para análise ML.

#### 3. Calcular Severidade
```python
def calculate_severity(row):
    msg = str(row['message']).upper()

    # Crítico (2)
    if any(kw in msg for kw in ['RED FLAG', 'SAFETY CAR DEPLOYED']):
        return 2

    # Atenção (1)
    if any(kw in msg for kw in ['YELLOW FLAG', 'VSC', 'PENALTY']):
        return 1

    # Info (0)
    return 0
```

**Por quê:** Hierarquia de importância (0=info, 1=warning, 2=critical).

### Dados Processados (Saída)

**Arquivo:** `data/processed/races/YEAR/round_XX/race_control_processed.parquet`

**Novas colunas:**
- `time_seconds` - Tempo normalizado
- `is_safety_car` - Flag (0/1)
- `is_flag` - Flag (0/1)
- `is_penalty` - Flag (0/1)
- `is_drs` - Flag (0/1)
- `category_encoded` - Categoria numérica (0-4)
- `event_severity` - Severidade (0/1/2)

### Exemplo de Uso

```python
import pandas as pd

rc = pd.read_parquet('data/processed/races/2025/round_01/race_control_processed.parquet')

# Safety car events
safety_cars = rc[rc['is_safety_car'] == 1]
print(f"Safety cars: {len(safety_cars)}")

# Eventos críticos
critical = rc[rc['event_severity'] == 2]
```

---

## 4. Pré-processamento de WEATHER

### Dados Brutos (Entrada)

**Arquivo:** `data/raw/races/YEAR/round_XX/weather.parquet`

**Estrutura:**
```
Time     | AirTemp | TrackTemp | Humidity | Pressure | Rainfall | WindSpeed
---------|---------|-----------|----------|----------|----------|----------
00:00:00 | 24.5    | 38.2      | 55       | 1013     | False    | 2.3
00:01:00 | 24.6    | NaN       | 56       | 1013     | False    | 2.1
00:02:00 | NaN     | 38.5      | 55       | NaN      | False    | 2.4
```

**Problemas:**
- Valores faltantes (NaN)
- Temperaturas não normalizadas
- Falta tendências (subindo/descendo)

### O Que é Feito

**Função:** `preprocess_weather()`

**Localização:** `src/preprocessing/feature_engineering.py`

#### 1. Interpolar Valores Faltantes
```python
for col in ['AirTemp', 'TrackTemp', 'Humidity']:
    df[col] = df[col].interpolate(method='linear', limit_direction='both')
    df[col] = df[col].fillna(df[col].mean())
```

**Por quê:** Sensores podem falhar. Interpolação linear preenche gaps.

**Exemplo:**
```
Antes:  [24.5, NaN, NaN, 27.5]
Depois: [24.5, 25.5, 26.5, 27.5]
```

#### 2. Normalizar Temperaturas (Z-score)
```python
air_mean = df['AirTemp'].mean()
air_std = df['AirTemp'].std()
df['air_temp_normalized'] = (df['AirTemp'] - air_mean) / air_std
```

**Por quê:** Z-score coloca valores em escala comum (média=0, std=1).

#### 3. Calcular Delta e Tendências
```python
# Diferença pista-ar
df['temp_delta'] = df['TrackTemp'] - df['AirTemp']

# Tendência de temperatura
df['temp_trend'] = df['TrackTemp'].diff().fillna(0)

# Direção: 1=subindo, -1=descendo, 0=estável
df['temp_trend_direction'] = np.where(df['temp_trend'] > 0.5, 1,
                              np.where(df['temp_trend'] < -0.5, -1, 0))
```

#### 4. Detectar Mudanças Bruscas
```python
# Temperatura: > 2 desvios padrão
temp_std = df['temp_trend'].std()
df['weather_change'] = (np.abs(df['temp_trend']) > 2 * temp_std).astype(int)

# Chuva começando
rain_change = df['Rainfall'].diff()
df['weather_change'] = np.maximum(df['weather_change'], (rain_change > 0).astype(int))
```

**Por quê:** Mudanças bruscas ou início de chuva são eventos importantes.

### Dados Processados (Saída)

**Arquivo:** `data/processed/races/YEAR/round_XX/weather_processed.parquet`

**Novas colunas:**
- `time_seconds` - Tempo normalizado
- `air_temp_normalized` - Z-score
- `track_temp_normalized` - Z-score
- `temp_delta` - Diferença pista-ar
- `rainfall_indicator` - Binário (0/1)
- `temp_trend` - Variação de temperatura
- `temp_trend_direction` - 1=subindo, -1=descendo, 0=estável
- `weather_change` - Flag de mudança brusca (0/1)

### Exemplo de Uso

```python
import pandas as pd

weather = pd.read_parquet('data/processed/races/2025/round_01/weather_processed.parquet')

# Períodos de chuva
rain = weather[weather['rainfall_indicator'] == 1]

# Mudanças bruscas
changes = weather[weather['weather_change'] == 1]

# Temperatura subindo
heating = weather[weather['temp_trend_direction'] == 1]
```

---

## 5. Pré-processamento de RESULTS

### Dados Brutos (Entrada)

**Arquivo:** `data/raw/races/YEAR/round_XX/results.parquet`

**Estrutura:**
```
Position | GridPosition | DriverNumber | Abbreviation | Points | Status
---------|--------------|--------------|--------------|--------|----------
1        | 3            | 1            | VER          | 25     | Finished
2        | 1            | 44           | HAM          | 18     | Finished
20       | 15           | 16           | LEC          | 0      | Collision
```

**Problemas:**
- Posições são strings às vezes
- Status é texto livre
- Falta análise de desempenho relativo

### O Que é Feito

**Função:** `preprocess_results()`

**Localização:** `src/preprocessing/feature_engineering.py`

#### 1. Garantir Posições Numéricas
```python
df['final_position'] = pd.to_numeric(df['Position'], errors='coerce')
df['grid_position'] = pd.to_numeric(df['GridPosition'], errors='coerce')
```

#### 2. Calcular Mudança de Posição
```python
df['position_change'] = df['final_position'] - df['grid_position']
df['position_gain'] = (df['position_change'] < 0).astype(int)
```

**Por quê:** Negativo = ganhou posições, Positivo = perdeu.

**Exemplo:**
```
Grid: 10, Final: 3  → position_change = -7 (ganhou 7!)
Grid: 2,  Final: 8  → position_change = +6 (perdeu 6)
```

#### 3. Categorizar DNF
```python
def categorize_dnf(status):
    status_upper = str(status).upper()
    if 'COLLISION' in status_upper:
        return 'collision'
    elif 'MECHANICAL' in status_upper or 'ENGINE' in status_upper:
        return 'mechanical'
    elif 'FINISHED' in status_upper:
        return 'finished'
    else:
        return 'other'
```

#### 4. Calcular Performance Score
```python
# Combina: posição final + ganho de posições + pontos
df['performance_score'] = 0.0

# Posição final (invertida)
df['performance_score'] += (max_pos + 1 - df['final_position']) / max_pos

# Ganho de posições
df['performance_score'] += df['position_change'] * -0.1

# Pontos normalizados
df['performance_score'] += df['points_normalized']

# Normalizar para [0, 1]
df['performance_score'] = (df['performance_score'] - min) / (max - min)
```

**Por quê:** Score único de desempenho relativo.

### Dados Processados (Saída)

**Arquivo:** `data/processed/races/YEAR/round_XX/results_processed.parquet`

**Novas colunas:**
- `final_position` - Numérica
- `grid_position` - Numérica
- `position_change` - Mudança (negativo = ganhou)
- `position_gain` - Flag (0/1)
- `finish_status` - 1=finished, 0=DNF
- `dnf_category` - Tipo (collision, mechanical, electrical, finished, other)
- `points_normalized` - [0-1]
- `performance_score` - [0-1]

### Exemplo de Uso

```python
import pandas as pd

results = pd.read_parquet('data/processed/races/2025/round_01/results_processed.parquet')

# Top 5 por ganho de posições
best_gainers = results.nsmallest(5, 'position_change')

# Top 5 por performance
best_performers = results.nlargest(5, 'performance_score')

# DNFs por categoria
dnf_stats = results[results['finish_status'] == 0]['dnf_category'].value_counts()
```

---

## Resumo: Transformações por Tipo

| Tipo         | Biblioteca     | Transformações                                   |
|--------------|----------------|--------------------------------------------------|
| Laps         | scipy.stats    | Z-score, regressão linear, outliers              |
| Telemetria   | scipy.interpolate, scipy.signal | Spline, filtro mediano, Savitzky-Golay |
| Race Control | pandas, numpy  | Flags binários, severidade, normalização         |
| Weather      | scipy, pandas  | Interpolação, Z-score, tendências                |
| Results      | pandas, numpy  | Mudança de posições, categorias, performance score |

---

## Estrutura dos Arquivos

### Dados Brutos
```
data/raw/races/YEAR/round_XX/
├── laps.parquet
├── telemetry/
│   ├── VER.parquet
│   ├── HAM.parquet
│   └── ...
├── race_control.parquet
├── weather.parquet
├── results.parquet
└── metadata.json
```

### Dados Processados
```
data/processed/races/YEAR/round_XX/
├── laps_processed.parquet
├── telemetry/
│   ├── VER_processed.parquet
│   ├── HAM_processed.parquet
│   └── ...
├── race_control_processed.parquet
├── weather_processed.parquet
└── results_processed.parquet
```

---

## Visualizando os Dados Processados

### Durante a Execução
```bash
uv run python cli/pipeline.py 2025 1 --show-sample
```

### Depois de Processar
```bash
uv run python -c "
import pandas as pd

# Laps
laps = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')
print('=== LAPS ===')
print(laps[['Driver', 'LapNumber', 'z_score', 'is_outlier', 'degradation_slope']].head())

# Race Control
rc = pd.read_parquet('data/processed/races/2025/round_01/race_control_processed.parquet')
print('\n=== RACE CONTROL ===')
print(rc[['time_seconds', 'is_safety_car', 'is_flag', 'event_severity']].head())

# Weather
w = pd.read_parquet('data/processed/races/2025/round_01/weather_processed.parquet')
print('\n=== WEATHER ===')
print(w[['time_seconds', 'AirTemp', 'TrackTemp', 'temp_delta', 'rainfall_indicator']].head())

# Results
r = pd.read_parquet('data/processed/races/2025/round_01/results_processed.parquet')
print('\n=== RESULTS ===')
print(r[['Abbreviation', 'final_position', 'position_change', 'performance_score']].head())
"
```

---

## Performance

| Operação             | Tempo (com cache) | Dados         |
|----------------------|-------------------|---------------|
| Laps                 | ~2s               | 927 voltas    |
| Telemetria (20 pilotos) | ~15s           | ~11MB         |
| Race Control         | <1s               | ~50 eventos   |
| Weather              | <1s               | ~100 registros|
| Results              | <1s               | 20 pilotos    |
| **TOTAL**            | **~20s**          | **~12MB**     |

---

## Troubleshooting

### Erro: "Nenhum valor de distância válido"
**Causa:** Telemetria sem coluna Distance
**Solução:** Verificar extração, usar dados completos

### Warning: "R² muito baixo"
**Causa:** Condições instáveis (chuva, safety car)
**Solução:** Normal, não é erro

### Erro: "Coluna não encontrada"
**Causa:** Dados usando nomes diferentes
**Solução:** CLI detecta automaticamente

---

## 6. Pré-processamento para SCIKIT-LEARN

### Visão Geral

Algoritmos de ML do Scikit-learn (K-Means, DBSCAN, Isolation Forest) são baseados em **distância**.
Três etapas de pré-processamento são **obrigatórias** antes de alimentar esses algoritmos:

1. **Imputação** - Preencher valores faltantes (NaN)
2. **Encoding** - Converter categorias em números
3. **Escalonamento** - Colocar todas as variáveis na mesma escala

**Por quê essas etapas são críticas?**

Se você misturar "Tempo de Volta" (ex: 90 segundos) com "Idade do Pneu" (ex: 5 voltas), a variável de maior magnitude dominará o cálculo de distância se não houver escalonamento.

**Localização:** `src/preprocessing/feature_engineering.py`

---

### 6.1. Imputação de Valores Faltantes

**Função:** `impute_missing_values()`

#### O Problema

Algoritmos de ML não aceitam valores NaN. Pequenas falhas na telemetria ou dados faltantes causam erros.

#### Soluções Disponíveis

##### A. SimpleImputer (Rápido)
```python
from src.preprocessing.feature_engineering import impute_missing_values

laps_imputed = impute_missing_values(
    laps_df,
    numeric_columns=['LapTime_seconds', 'Sector1Time_seconds'],
    strategy='median',  # ou 'mean', 'most_frequent'
    use_knn=False
)
```

**Estratégias:**
- `mean`: Média dos valores válidos
- `median`: Mediana (mais resistente a outliers)
- `most_frequent`: Valor mais comum

**Quando usar:** Poucos valores faltantes (<5%), dados simples

##### B. KNNImputer (Sofisticado)
```python
laps_imputed = impute_missing_values(
    laps_df,
    numeric_columns=['LapTime_seconds', 'Sector1Time_seconds'],
    use_knn=True,
    n_neighbors=5
)
```

**Como funciona:** Usa os 5 vizinhos mais próximos para estimar o valor faltante

**Quando usar:** Telemetria, muitos valores faltantes, padrões complexos

#### Por Quê

- **SimpleImputer:** Rápido, assume independência entre variáveis
- **KNNImputer:** Mais preciso, considera padrões e correlações nos dados

#### Exemplo Completo

```python
import pandas as pd
from src.preprocessing.feature_engineering import impute_missing_values

# Carregar dados
laps = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')

# Verificar valores faltantes
print(laps.isnull().sum())

# Imputar
laps_clean = impute_missing_values(
    laps,
    numeric_columns=['LapTime_seconds', 'Sector1Time_seconds', 'TyreLife'],
    strategy='median'
)

# Verificar
print(laps_clean.isnull().sum())  # Deve ser 0
```

---

### 6.2. Encoding de Variáveis Categóricas

**Função:** `encode_categorical_variables()`

#### O Problema

Variáveis como **Composto do Pneu** (Soft, Medium, Hard, Inter) são qualitativas.
Algoritmos baseados em distância não entendem que "SOFT" ≠ "HARD".

#### Solução: OneHotEncoder

Transforma categorias em colunas binárias:

```python
from src.preprocessing.feature_engineering import encode_categorical_variables

laps_encoded = encode_categorical_variables(
    laps_df,
    categorical_columns=['Compound'],
    drop_first=True  # Evita multicolinearidade
)
```

**Antes:**
```
Compound
--------
SOFT
SOFT
MEDIUM
HARD
```

**Depois:**
```
Compound_MEDIUM | Compound_HARD
----------------|---------------
0               | 0              (era SOFT)
0               | 0              (era SOFT)
1               | 0              (é MEDIUM)
0               | 1              (é HARD)
```

#### Parâmetro `drop_first`

- `drop_first=True`: Remove primeira categoria (evita redundância)
- `drop_first=False`: Mantém todas (mais explícito)

**Recomendação:** Use `True` para regressão, `False` para interpretabilidade

#### Por Quê

OneHotEncoding cria features binárias que representam "presença" de uma categoria.
Algoritmos de distância podem então calcular:
- Distância entre duas voltas com SOFT: pequena
- Distância entre SOFT e HARD: grande

#### Exemplo Completo

```python
import pandas as pd
from src.preprocessing.feature_engineering import encode_categorical_variables

laps = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')

# Antes
print(laps['Compound'].unique())  # ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE']

# Encoding
laps_encoded = encode_categorical_variables(
    laps,
    categorical_columns=['Compound', 'TrackStatus'],
    drop_first=True
)

# Depois
print(laps_encoded.columns)  # Compound_MEDIUM, Compound_HARD, Compound_INTERMEDIATE, ...
```

---

### 6.3. Escalonamento de Features (CRÍTICO)

**Função:** `scale_features()`

#### O Problema

Esta é a etapa **MAIS CRÍTICA** para algoritmos baseados em distância.

**Exemplo do problema:**
```
LapTime_seconds: 90.5  (magnitude ~90)
TyreLife:        5     (magnitude ~5)
Speed:           287   (magnitude ~300)
```

Sem escalonamento, a distância é dominada por `Speed` (maior magnitude), ignorando `LapTime` e `TyreLife`.

#### Solução: Escalonamento

Coloca todas as variáveis na mesma escala (média=0, variância=1)

##### A. StandardScaler (Padrão)
```python
from src.preprocessing.feature_engineering import scale_features

laps_scaled = scale_features(
    laps_df,
    numeric_columns=['LapTime_seconds', 'TyreLife', 'Sector1Time_seconds'],
    scaler_type='standard'
)
```

**Como funciona:**
```python
X_scaled = (X - mean(X)) / std(X)
```

**Quando usar:** Distribuição aproximadamente normal, sem outliers extremos

##### B. RobustScaler (Resistente a Outliers)
```python
laps_scaled = scale_features(
    laps_df,
    numeric_columns=['LapTime_seconds', 'TyreLife'],
    scaler_type='robust'
)
```

**Como funciona:** Usa quartis (mediana e IQR) em vez de média e desvio padrão

**Quando usar:** Dados com outliers extremos (ex: rodadas, colisões que aumentam tempo em 30s)

#### Por Quê

Sem escalonamento:
- K-Means agrupa baseado em magnitude, não em padrão
- DBSCAN usa `eps` (distância) que não funciona para escalas diferentes
- Isolation Forest se confunde com magnitudes diferentes

Com escalonamento:
- Todas as variáveis têm peso igual
- Distâncias são comparáveis
- Algoritmos funcionam corretamente

#### Retornar o Scaler

Para aplicar a mesma transformação em novos dados:

```python
laps_scaled, scaler = scale_features(
    laps_df,
    numeric_columns=['LapTime_seconds'],
    scaler_type='robust',
    return_scaler=True
)

# Mais tarde, em novos dados
new_laps_scaled = scaler.transform(new_laps_df[['LapTime_seconds']])
```

#### Exemplo Completo

```python
import pandas as pd
from src.preprocessing.feature_engineering import scale_features

laps = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')

# Antes
print(laps['LapTime_seconds'].describe())
print(laps['TyreLife'].describe())

# Escalonar
laps_scaled = scale_features(
    laps,
    numeric_columns=['LapTime_seconds', 'TyreLife', 'Sector1Time_seconds'],
    scaler_type='robust'  # Resistente a outliers
)

# Depois (média ~0, std ~1)
print(laps_scaled['LapTime_seconds'].describe())
print(laps_scaled['TyreLife'].describe())
```

---

### Pipeline Completo de Pré-processamento para ML

Exemplo combinando as 3 etapas:

```python
from src.preprocessing.feature_engineering import (
    impute_missing_values,
    encode_categorical_variables,
    scale_features
)

# Etapa 1: Imputação
laps_imputed = impute_missing_values(
    laps_df,
    numeric_columns=['LapTime_seconds', 'Sector1Time_seconds', 'TyreLife'],
    strategy='median'
)

# Etapa 2: Encoding
laps_encoded = encode_categorical_variables(
    laps_imputed,
    categorical_columns=['Compound'],
    drop_first=True
)

# Etapa 3: Escalonamento
laps_scaled = scale_features(
    laps_encoded,
    numeric_columns=['LapTime_seconds', 'Sector1Time_seconds', 'TyreLife'],
    scaler_type='robust'
)

# Agora está pronto para K-Means, DBSCAN, Isolation Forest!
```

---

## Próximos Passos

Após pré-processar:
1. **Clustering** - K-Means, DBSCAN (agrupar stints, identificar ritmos) → Ver [src/ml/README.md](src/ml/README.md)
2. **Anomaly Detection** - Isolation Forest (detectar eventos raros) → Ver [src/ml/README.md](src/ml/README.md)
3. **Visualização** - Gráficos com dados sincronizados

---

## Referências

- **SciPy Interpolate:** https://docs.scipy.org/doc/scipy/reference/interpolate.html
- **SciPy Signal:** https://docs.scipy.org/doc/scipy/reference/signal.html
- **SciPy Stats:** https://docs.scipy.org/doc/scipy/reference/stats.html
- **Código:** `src/preprocessing/`
- **Testes:** `tests/preprocessing/`
