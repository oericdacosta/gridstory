# Guia de Pré-processamento - SciPy

Este guia mostra como usar o módulo de pré-processamento de dados de telemetria F1 usando **SciPy**.

## Visão Geral

O módulo de pré-processamento transforma dados brutos extraídos do FastF1 em features matemáticas limpas e sincronizadas, prontas para análise de Machine Learning. Ele utiliza três submódulos principais do SciPy:

- **`scipy.interpolate`**: Sincronização de telemetria
- **`scipy.signal`**: Processamento de sinal e remoção de ruído
- **`scipy.stats`**: Engenharia de features estatísticas

## Instalação

As dependências já estão incluídas no projeto:

```bash
uv sync  # SciPy, NumPy e Pandas já incluídos
```

## Estrutura dos Dados

### Entrada (Dados Brutos)
```
data/raw/races/2025/round_01/
├── laps.parquet           # Voltas com tempos, compostos, stints
└── telemetry/             # Telemetria por piloto
    ├── VER.parquet        # Speed, RPM, Throttle, Brake, Distance
    └── ...
```

### Saída (Dados Pré-processados)
```
data/processed/
├── laps_processed_2025_01_R.parquet      # Voltas + features estatísticas
└── telemetry_processed_2025_01_R.parquet # Telemetria sincronizada e limpa
```

## Uso Básico

### 1. Listar Dados Disponíveis

```bash
uv run python cli/list_data.py
```

Mostra todos os dados brutos e pré-processados disponíveis.

### 2. Pré-processar Dados de Voltas

```bash
# Ver estatísticas sem salvar
uv run python cli/preprocess.py --year 2025 --round 1 --laps

# Ver estatísticas E salvar dados processados
uv run python cli/preprocess.py --year 2025 --round 1 --laps --save

# Ver amostra dos dados em formato de tabela
uv run python cli/preprocess.py --year 2025 --round 1 --laps --show-sample
```

**O que é calculado:**
- ✅ Z-scores para cada volta
- ✅ Detecção de outliers estatísticos (|z| > 3)
- ✅ Taxa de degradação de pneu (segundos/volta)
- ✅ Estatísticas descritivas por grupo (média, desvio, assimetria, curtose)

### 3. Pré-processar Telemetria

```bash
# Telemetria de todos os pilotos
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry

# Filtrar por piloto específico
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --driver VER

# Filtrar piloto E volta específica
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --driver VER --lap 10 --save
```

**O que é calculado:**
- ✅ Sincronização em grid comum de distância (interpolação cúbica spline)
- ✅ Remoção de ruído de sensores (filtro mediano)
- ✅ Suavização de sinais (Savitzky-Golay)
- ✅ Cálculo de derivadas (aceleração, jerk)

### 4. Pré-processar Tudo

```bash
# Processar voltas + telemetria de uma vez
uv run python cli/preprocess.py --year 2025 --round 1 --all --save
```

### 5. Processar Múltiplas Corridas

```bash
# Loop para processar corridas 1, 2 e 3
for i in 1 2 3; do
  uv run python cli/preprocess.py --year 2025 --round $i --all --save
done
```

## Uso Avançado (Python)

### Engenharia de Features Estatísticas

```python
import pandas as pd
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats

# Carregar dados brutos
laps_df = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')

# Adicionar features estatísticas
enriched = enrich_dataframe_with_stats(
    laps_df,
    value_column='LapTime_seconds',
    group_by=['Driver', 'Compound', 'Stint'],
    include_degradation=True
)

# Filtrar outliers
clean_laps = enriched[~enriched['is_outlier']]

# Analisar degradação
for driver in ['VER', 'HAM', 'LEC']:
    driver_data = enriched[enriched['Driver'] == driver]

    for stint in driver_data['Stint'].unique():
        stint_data = driver_data[driver_data['Stint'] == stint].iloc[0]

        slope = stint_data['degradation_slope']
        r2 = stint_data['degradation_r_squared']
        compound = stint_data['Compound']

        print(f"{driver} - Stint {stint} ({compound}): {slope:.3f}s/volta (R²={r2:.2f})")
```

### Sincronização de Telemetria

```python
import pandas as pd
from src.preprocessing.interpolation import synchronize_telemetry

# Carregar telemetria de dois pilotos
ver_telem = pd.read_parquet('data/raw/races/2025/round_01/telemetry/VER.parquet')
ham_telem = pd.read_parquet('data/raw/races/2025/round_01/telemetry/HAM.parquet')

track_length = 5281.0  # Monaco em metros

# Sincronizar ambos no mesmo grid de distância
ver_sync = synchronize_telemetry(ver_telem, track_length, num_points=1000)
ham_sync = synchronize_telemetry(ham_telem, track_length, num_points=1000)

# Agora pode calcular deltas diretamente
speed_delta = ver_sync['Speed'].values - ham_sync['Speed'].values

print(f"Vantagem máxima VER: {speed_delta.max():.2f} km/h")
print(f"Vantagem máxima HAM: {abs(speed_delta.min()):.2f} km/h")
```

### Processamento de Sinal

```python
import numpy as np
from src.preprocessing.signal_processing import apply_telemetry_pipeline

# Extrair canais de telemetria
telemetry_dict = {
    'Speed': speed_array,
    'Throttle': throttle_array,
    'Brake': brake_array,
    'RPM': rpm_array,
}

# Aplicar pipeline completo
processed = apply_telemetry_pipeline(
    telemetry_dict,
    noise_reduction=True,      # Remove ruído com filtro mediano
    outlier_removal=True,       # Remove spikes de sensores
    calculate_derivatives=True  # Calcula aceleração
)

# Acessar dados processados
clean_speed = processed['Speed']
acceleration = processed['Speed_derivative']

print(f"Aceleração máxima: {np.nanmax(acceleration):.2f}")
print(f"Desaceleração máxima: {np.nanmin(acceleration):.2f}")
```

## Interpretando os Resultados

### Z-Score
- **z_score < -3 ou > +3**: Outlier estatístico
- **-1 < z_score < +1**: Valor normal
- **Exemplo**: `z_score = 2.5` significa 2.5 desvios padrão acima da média

### Taxa de Degradação
- **Valor positivo** (+0.5s/volta): Pneu degradando, tempos piorando
- **Valor negativo** (-0.5s/volta): Tempos melhorando (pista secando, combustível queimando)
- **R² alto** (>0.4): Degradação linear clara
- **R² baixo** (<0.1): Muita variação, sem padrão linear (condições instáveis)

**Exemplo:**
```
NOR, Stint 5 (HARD): -4.911s/volta (R²=0.42)
```
- Lando Norris melhorou 4.9s por volta no stint 5
- 42% da variação é explicada pela tendência linear
- Provavelmente: pista secando ou carro aliviando

### Outliers
- **is_outlier = True**: Volta estatisticamente anormal
- **Causas comuns**: Safety Car, tráfego, erro de piloto, problema técnico
- **Usar para**: Filtrar voltas antes de análises de performance

## Estrutura dos Módulos

### `src/preprocessing/interpolation.py`
**Sincronização de telemetria usando interpolação cúbica spline.**

Funções principais:
- `synchronize_telemetry()`: Sincroniza telemetria em grid comum
- `synchronize_multiple_laps()`: Cria matriz de voltas sincronizadas

**Casos de uso:**
- Comparar telemetria de pilotos diferentes
- Calcular deltas ponto-a-ponto
- Preparar dados para modelos ML

### `src/preprocessing/signal_processing.py`
**Processamento de sinal e remoção de ruído.**

Funções principais:
- `clean_signal()`: Remove ruído (filtro mediano ou Savitzky-Golay)
- `calculate_derivative()`: Calcula derivadas suaves
- `remove_outliers()`: Detecta e remove outliers
- `apply_telemetry_pipeline()`: Pipeline completo

**Casos de uso:**
- Limpar ruído de sensores
- Calcular aceleração/desaceleração
- Remover spikes anormais

### `src/preprocessing/feature_engineering.py`
**Engenharia de features estatísticas.**

Funções principais:
- `calculate_statistical_features()`: Z-scores e outliers
- `calculate_degradation_rate()`: Regressão linear de degradação
- `calculate_descriptive_statistics()`: Estatísticas descritivas
- `enrich_dataframe_with_stats()`: Pipeline completo

**Casos de uso:**
- Detectar voltas anormais
- Calcular degradação de pneu
- Preparar features para ML

## Exemplos Práticos

### Exemplo 1: Análise de Degradação de Pneu

```python
import pandas as pd
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats

# Carregar dados
laps = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')

# Filtrar apenas um piloto
ver_laps = laps[laps['Driver'] == 'VER']

# Calcular degradação
enriched = enrich_dataframe_with_stats(
    ver_laps,
    value_column='LapTime_seconds',
    group_by=['Stint', 'Compound'],
    include_degradation=True
)

# Mostrar degradação por stint
for stint in enriched['Stint'].unique():
    stint_data = enriched[enriched['Stint'] == stint].iloc[0]

    slope = stint_data['degradation_slope']
    compound = stint_data['Compound']

    if slope > 0:
        print(f"Stint {stint} ({compound}): Degradação de {slope:.3f}s/volta")
    else:
        print(f"Stint {stint} ({compound}): Melhora de {abs(slope):.3f}s/volta")
```

### Exemplo 2: Comparação de Pilotos

```python
from src.preprocessing.interpolation import synchronize_telemetry
import matplotlib.pyplot as plt

# Carregar telemetria
ver = pd.read_parquet('data/raw/races/2025/round_01/telemetry/VER.parquet')
lec = pd.read_parquet('data/raw/races/2025/round_01/telemetry/LEC.parquet')

# Sincronizar
track_length = 5281.0
ver_sync = synchronize_telemetry(ver, track_length, num_points=1000)
lec_sync = synchronize_telemetry(lec, track_length, num_points=1000)

# Calcular delta
speed_delta = ver_sync['Speed'] - lec_sync['Speed']

# Plotar
plt.figure(figsize=(12, 6))
plt.plot(ver_sync['Distance'], speed_delta)
plt.xlabel('Distância (m)')
plt.ylabel('Delta de Velocidade (km/h)')
plt.title('VER vs LEC - Delta de Velocidade')
plt.grid(True)
plt.show()
```

### Exemplo 3: Detecção de Anomalias

```python
from src.preprocessing.signal_processing import remove_outliers
import numpy as np

# Extrair RPM de telemetria
rpm_data = telemetry['RPM'].values

# Detectar outliers
clean_rpm, outlier_mask = remove_outliers(
    rpm_data,
    threshold=3.0,
    method='median'  # Mais robusto que z-score
)

print(f"Outliers encontrados: {outlier_mask.sum()} de {len(rpm_data)}")
print(f"Percentual: {outlier_mask.sum() / len(rpm_data) * 100:.2f}%")

# Mostrar onde estão os outliers
outlier_indices = np.where(outlier_mask)[0]
print(f"Índices dos outliers: {outlier_indices}")
```

## Parâmetros Importantes

### Sincronização de Telemetria
- **num_points**: Número de pontos no grid (padrão: 5000)
  - Mais pontos = maior resolução, mas arquivos maiores
  - Menos pontos = menor resolução, processamento mais rápido

### Processamento de Sinal
- **kernel_size**: Tamanho da janela do filtro (padrão: 5)
  - Maior = mais suavização
  - Menor = preserva mais detalhes

- **method**: Tipo de filtro
  - `"median"`: Melhor para remover spikes
  - `"savgol"`: Melhor para suavização geral

### Detecção de Outliers
- **threshold**: Número de desvios padrão (padrão: 3.0)
  - Maior = mais tolerante (menos outliers)
  - Menor = mais rigoroso (mais outliers)

## Troubleshooting

### Erro: "Nenhum valor de distância válido encontrado"
**Causa**: Telemetria sem coluna Distance ou com valores NaN
**Solução**: Verificar se a telemetria foi extraída corretamente

### Warning: "R² muito baixo"
**Causa**: Condições de corrida muito variáveis (chuva, Safety Car)
**Solução**: Normal em condições instáveis, não é erro

### Erro: "Coluna 'LapTime' não encontrada"
**Causa**: Dados usando coluna 'LapTime_seconds'
**Solução**: CLI detecta automaticamente, usar módulo atualizado

## Performance

### Benchmark (Corrida Completa)

| Operação | Tempo | Dados |
|----------|-------|-------|
| Pré-processar voltas | ~2s | 927 voltas |
| Pré-processar telemetria | ~15s | 20 pilotos |
| Salvar em Parquet | <1s | - |

### Dicas de Otimização

1. **Use filtros por piloto** quando possível
2. **Reduza num_points** se não precisa de alta resolução
3. **Processe em lote** múltiplas corridas
4. **Use --save** apenas quando necessário

## Próximos Passos

Após pré-processar os dados:

1. **Análise ML**: Use dados limpos para treinar modelos
2. **Detecção de eventos**: Aplique Ruptures nos dados sincronizados
3. **Clustering**: Agrupe stints similares
4. **Visualização**: Crie gráficos com dados sincronizados

## Referências

- Documentação SciPy Interpolate: https://docs.scipy.org/doc/scipy/reference/interpolate.html
- Documentação SciPy Signal: https://docs.scipy.org/doc/scipy/reference/signal.html
- Documentação SciPy Stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- Código fonte: `src/preprocessing/`
- Testes: `tests/preprocessing/`
- Exemplos: `examples/preprocessing_example.py`

## Suporte

Para problemas ou dúvidas:
1. Verifique os logs de erro
2. Consulte os exemplos em `examples/preprocessing_example.py`
3. Execute os testes: `uv run pytest tests/preprocessing/ -v`
4. Reporte issues no repositório
