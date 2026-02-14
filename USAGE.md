# PitWall AI - Guia de Uso

## Visão Geral

PitWall AI é um pipeline completo para extração, pré-processamento e análise ML de dados de corridas de Fórmula 1.

**Um único comando faz tudo:**
```bash
uv run python cli/pipeline.py 2025 1
```

Este comando:
1. ✅ Extrai TODOS os dados da corrida (laps, telemetry, race_control, weather, results)
2. ✅ Pré-processa TODOS os dados com NumPy, Pandas e SciPy
3. ✅ Executa Machine Learning com Scikit-learn (clustering + detecção de anomalias)
4. ✅ Salva dados brutos, processados e resultados de ML organizados

## Instalação

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/pitwall-ai.git
cd pitwall-ai

# Instalar dependências
uv sync
```

## Uso Básico

### Pipeline Completo (Recomendado)

```bash
# Extrair e processar corrida completa
uv run python cli/pipeline.py 2025 1

# Com polling (aguardar disponibilidade dos dados)
uv run python cli/pipeline.py 2025 1 --polling

# Mostrar amostras dos dados processados
uv run python cli/pipeline.py 2025 1 --show-sample
```

**Saída:**
- `data/raw/races/2025/round_01/` - Dados brutos
  - `laps.parquet`
  - `telemetry/*.parquet` (um arquivo por piloto)
  - `race_control.parquet`
  - `weather.parquet`
  - `results.parquet`
  - `metadata.json`

- `data/processed/races/2025/round_01/` - Dados processados
  - `laps_processed.parquet`
  - `telemetry/*_processed.parquet`
  - `race_control_processed.parquet`
  - `weather_processed.parquet`
  - `results_processed.parquet`

- `data/ml/races/2025/round_01/` - Resultados de Machine Learning
  - `laps_clustered.parquet` - Voltas com labels de cluster (ritmos identificados)
  - `laps_anomalies.parquet` - Voltas com detecção de anomalias (eventos raros)
  - `anomalies_summary.parquet` - Sumário de anomalias por piloto

## Comandos Individuais (Opcional)

### Apenas Extração

```bash
# Extrair TODOS os dados de uma corrida
uv run python cli/extract.py 2025 1

# Com polling (para corridas recentes)
uv run python cli/extract.py 2025 1 --polling
```

**Importante:** A extração SEMPRE inclui telemetria de todos os pilotos.

### Apenas Pré-processamento

```bash
# Pré-processar tudo (de dados já extraídos)
uv run python cli/preprocess.py --year 2025 --round 1 --all --save

# Pré-processar apenas laps
uv run python cli/preprocess.py --year 2025 --round 1 --laps --save

# Pré-processar apenas telemetria
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --save
```

## Dados Extraídos (Brutos)

### 1. Laps (Voltas e Estratégia)

**Colunas principais:**
- `LapTime_seconds`: Tempo da volta em segundos
- `Sector1Time_seconds`, `Sector2Time_seconds`, `Sector3Time_seconds`: Tempos por setor
- `Compound`: Tipo de pneu (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
- `TyreLife`: Número de voltas do pneu
- `FreshTyre`: Se é pneu novo (boolean)
- `Stint`: Número do stint
- `PitInTime_seconds`, `PitOutTime_seconds`: Tempos de pit stop
- `Position`: Posição na pista
- `TrackStatus`: Status da pista

### 2. Telemetria (Dados do Carro)

**Colunas principais:**
- `Speed`: Velocidade em km/h
- `RPM`: Rotações do motor
- `Throttle`: Acelerador (0-100%)
- `Brake`: Freio (True/False)
- `nGear`: Marcha atual (0-8)
- `DRS`: DRS ativo (True/False)
- `Distance`: Distância percorrida na pista (metros)
- `Time_seconds`: Tempo em segundos desde o início

### 3. Race Control (Controle de Corrida)

**Colunas principais:**
- `Time_seconds`: Tempo do evento
- `Category`: Categoria da mensagem
- `Message`: Texto da mensagem
- `Status`: Status relacionado
- `Flag`: Tipo de bandeira (se aplicável)

### 4. Weather (Condições Meteorológicas)

**Colunas principais:**
- `Time_seconds`: Tempo da medição
- `AirTemp`: Temperatura do ar (°C)
- `TrackTemp`: Temperatura da pista (°C)
- `Humidity`: Umidade relativa (%)
- `Pressure`: Pressão atmosférica (mbar)
- `WindSpeed`: Velocidade do vento (km/h)
- `Rainfall`: Indicador de chuva

### 5. Results (Resultados Finais)

**Colunas principais:**
- `Position`: Posição final
- `GridPosition`: Posição no grid de largada
- `DriverNumber`: Número do piloto
- `Abbreviation`: Abreviação do piloto (VER, HAM, LEC, etc.)
- `FullName`: Nome completo
- `TeamName`: Nome da equipe
- `Points`: Pontos conquistados
- `Status`: Status final (Finished, +1 Lap, Collision, etc.)

## Dados Pré-processados

### 1. Laps Processados

**Features adicionadas:**
- `z_score`: Score padronizado do tempo de volta
- `is_outlier`: Flag de outlier (|z| > 3)
- `group_mean`, `group_std`: Estatísticas do grupo
- `degradation_slope`: Taxa de degradação do pneu (seg/volta)
- `degradation_r_squared`: Qualidade do ajuste da degradação
- `group_nobs`, `group_skewness`, `group_kurtosis`: Estatísticas descritivas

**Exemplo de uso:**
```python
import pandas as pd

laps = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')

# Filtrar voltas válidas (sem outliers)
clean_laps = laps[~laps['is_outlier']]

# Analisar degradação por piloto
for driver in clean_laps['Driver'].unique():
    driver_laps = clean_laps[clean_laps['Driver'] == driver]
    print(f"{driver}: {driver_laps['degradation_slope'].iloc[0]:.3f} s/volta")
```

### 2. Telemetria Processada

**Features adicionadas:**
- `Speed_derivative`: Aceleração (km/h/s)
- `Throttle_derivative`: Taxa de mudança do acelerador
- `Brake_derivative`: Taxa de mudança do freio
- Todos os canais sincronizados em grid comum de distância
- Ruído removido com filtro mediano
- Outliers detectados e corrigidos

**Exemplo de uso:**
```python
import pandas as pd

ver = pd.read_parquet('data/processed/races/2025/round_01/telemetry/VER_processed.parquet')
ham = pd.read_parquet('data/processed/races/2025/round_01/telemetry/HAM_processed.parquet')

# Comparar velocidades (sincronizadas!)
speed_delta = ver['Speed'] - ham['Speed']

# Analisar aceleração
acceleration = ver['Speed_derivative']
max_accel = acceleration.max()
```

### 3. Race Control Processado

**Features adicionadas:**
- `time_seconds`: Tempo normalizado
- `is_safety_car`: Flag de safety car/VSC
- `is_flag`: Flag de bandeiras
- `is_penalty`: Flag de penalidades
- `is_drs`: Flag de eventos DRS
- `category_encoded`: Categoria codificada (0-4)
- `event_severity`: Severidade (0=info, 1=warning, 2=critical)

**Exemplo de uso:**
```python
import pandas as pd

rc = pd.read_parquet('data/processed/races/2025/round_01/race_control_processed.parquet')

# Encontrar safety car events
safety_cars = rc[rc['is_safety_car']]
print(f"Safety car deployments: {len(safety_cars)}")

# Eventos críticos
critical = rc[rc['event_severity'] == 2]
```

### 4. Weather Processado

**Features adicionadas:**
- `time_seconds`: Tempo normalizado
- `air_temp_normalized`: Temperatura do ar normalizada (Z-score)
- `track_temp_normalized`: Temperatura da pista normalizada
- `temp_delta`: Diferença pista-ar
- `rainfall_indicator`: Indicador binário de chuva
- `temp_trend`: Tendência de temperatura
- `temp_trend_direction`: Direção (1=subindo, -1=descendo, 0=estável)
- `weather_change`: Flag de mudança brusca

**Exemplo de uso:**
```python
import pandas as pd

weather = pd.read_parquet('data/processed/races/2025/round_01/weather_processed.parquet')

# Períodos de chuva
rain_periods = weather[weather['rainfall_indicator'] == 1]

# Mudanças bruscas de temperatura
temp_changes = weather[weather['weather_change'] == 1]
```

### 5. Results Processado

**Features adicionadas:**
- `final_position`: Posição final (numérica)
- `grid_position`: Posição no grid (numérica)
- `position_change`: Mudança de posição (negativo = ganhou)
- `position_gain`: Flag de ganho de posições
- `finish_status`: 1=finished, 0=DNF
- `dnf_category`: Tipo de DNF (collision, mechanical, electrical, finished, other)
- `points_normalized`: Pontos normalizados [0-1]
- `performance_score`: Score de desempenho relativo [0-1]

**Exemplo de uso:**
```python
import pandas as pd

results = pd.read_parquet('data/processed/races/2025/round_01/results_processed.parquet')

# Top performers (por ganho de posições)
best_gainers = results.nsmallest(5, 'position_change')

# Melhores performance scores
best_performers = results.nlargest(5, 'performance_score')

# DNFs por categoria
dnf_stats = results[results['finish_status'] == 0]['dnf_category'].value_counts()
```

## Uso Programático

### Pipeline Completo

```python
from pathlib import Path
import fastf1

# Habilitar cache
cache_dir = Path.home() / ".cache" / "fastf1"
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Executar pipeline completo
from cli.pipeline import run_complete_pipeline

run_complete_pipeline(
    year=2025,
    round_num=1,
    use_polling=False,
    show_sample=True
)
```

### Extração Apenas

```python
from src.extraction.orchestrator import extract_race_complete

race_dir = extract_race_complete(
    year=2025,
    round_number=1,
    use_polling=False,
    output_dir="data/raw/races"
)

print(f"Dados salvos em: {race_dir}")
```

### Pré-processamento Apenas

```python
import pandas as pd
from src.preprocessing.feature_engineering import (
    enrich_dataframe_with_stats,
    preprocess_race_control,
    preprocess_weather,
    preprocess_results
)

# Laps
laps_df = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')
laps_processed = enrich_dataframe_with_stats(
    laps_df,
    value_column='LapTime_seconds',
    group_by=['Driver', 'Compound'],
    include_degradation=True
)

# Race Control
rc_df = pd.read_parquet('data/raw/races/2025/round_01/race_control.parquet')
rc_processed = preprocess_race_control(rc_df)

# Weather
weather_df = pd.read_parquet('data/raw/races/2025/round_01/weather.parquet')
weather_processed = preprocess_weather(weather_df)

# Results
results_df = pd.read_parquet('data/raw/races/2025/round_01/results.parquet')
results_processed = preprocess_results(results_df)
```

## Performance

- **Extração (com telemetria):** ~30-60 segundos
- **Pré-processamento:** ~10-20 segundos
- **Pipeline completo:** ~40-80 segundos
- **Primeira execução:** Mais lento (download inicial)

## Troubleshooting

### Cache do FastF1

O FastF1 usa cache local em `~/.cache/fastf1/`. Se tiver problemas:

```bash
# Limpar cache
rm -rf ~/.cache/fastf1/
```

### Corridas recentes sem dados

Para corridas muito recentes, use `--polling`:

```bash
uv run python cli/pipeline.py 2025 10 --polling
```

### Dados faltantes

A extração SEMPRE inclui todos os 5 tipos de dados. Se algum estiver faltando, verifique:
- Corrida foi realizada?
- Dados disponíveis no FastF1?
- Cache corrompido? (limpe o cache)

## Próximos Passos

Após extrair e pré-processar os dados, você pode:

1. **Análise exploratória:** Usar Jupyter notebooks para explorar os dados
2. **Machine Learning:** Implementar detecção de eventos (Ruptures, Isolation Forest)
3. **Visualização:** Criar gráficos de telemetria, degradação de pneus, etc.
4. **Exportação:** Converter para outros formatos (CSV, JSON)

## Documentação Adicional

- [README.md](README.md) - Visão geral do projeto
- [PREPROCESSING.md](PREPROCESSING.md) - Guia completo de pré-processamento (todos os 5 tipos de dados)
- [src/extraction/README.md](src/extraction/README.md) - Módulo de extração
- [src/preprocessing/README.md](src/preprocessing/README.md) - Módulo de pré-processamento
- [src/ml/README.md](src/ml/README.md) - Módulo de Machine Learning (Scikit-learn)
- [cli/README.md](cli/README.md) - Comandos CLI
