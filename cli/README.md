# Ferramentas CLI - PitWall AI

Comandos de linha de comando para extração, pré-processamento e análise de dados F1.

## Comandos Disponíveis

### 1. pipeline.py (Recomendado — ponto de entrada)

**Pipeline completo end-to-end:** Extração + Pré-processamento + Machine Learning em um único comando.

```bash
# Pipeline completo
uv run python cli/pipeline.py 2025 1

# Com polling (aguardar dados)
uv run python cli/pipeline.py 2025 1 --polling

# Mostrar amostras
uv run python cli/pipeline.py 2025 1 --show-sample
```

**O que faz:**
1. ✅ **FASE 1 - Extração**: Coleta TODOS os dados da corrida (laps, telemetry, race_control, weather, results)
2. ✅ **FASE 2 - Pré-processamento**: Processa TODOS os dados (features, normalização, limpeza)
3. ✅ **FASE 3 - Machine Learning**: Clustering (K-Means) e detecção de anomalias (Isolation Forest)
4. ✅ Salva resultados em 3 diretórios organizados

**Estrutura Modular do Pipeline:**

O pipeline é composto por módulos especializados em `cli/pipeline_steps/`:
- `extraction.py` - Gerencia extração de dados via FastF1
- `preprocessing.py` - Coordena pré-processamento de 5 tipos de dados
- `ml.py` - Executa análises de ML (imputação, encoding, clustering, anomalias)
- `reporting.py` - Classe `Reporter` para formatação consistente de saídas

**Opções:**
- `YEAR`: Ano da temporada (ex: 2025)
- `ROUND`: Número da rodada (ex: 1)
- `--polling`: Aguardar disponibilidade dos dados (para corridas recentes)
- `--show-sample`: Mostrar amostras dos dados processados

---

### 2. extract.py

**Apenas extração** de dados (sem pré-processamento).

```bash
# Extrair corrida completa
uv run python cli/extract.py 2025 1

# Com polling
uv run python cli/extract.py 2025 1 --polling
```

**O que extrai (SEMPRE):**
- ✅ Laps (voltas e estratégia)
- ✅ Telemetria (todos os pilotos)
- ✅ Race Control (eventos da corrida)
- ✅ Weather (condições meteorológicas)
- ✅ Results (classificação final)

**Opções:**
- `YEAR`: Ano da temporada
- `ROUND`: Número da rodada
- `--polling`: Aguardar disponibilidade dos dados
- `--output-dir DIR`: Diretório de saída (padrão: data/raw/races)

**IMPORTANTE:** Não há opção para extrair sem telemetria. Tudo é sempre extraído.

---

### 3. preprocess.py

**Apenas pré-processamento** (requer dados já extraídos).

```bash
# Pré-processar tudo
uv run python cli/preprocess.py --year 2025 --round 1 --all --save

# Apenas laps
uv run python cli/preprocess.py --year 2025 --round 1 --laps --save

# Apenas telemetria
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --save

# Mostrar amostras
uv run python cli/preprocess.py --year 2025 --round 1 --laps --show-sample
```

**O que processa:**
- Laps: Features estatísticas, degradação de pneus, outliers
- Telemetria: Sincronização, limpeza, derivadas
- Race Control: Eventos estruturados, severidade
- Weather: Tendências, mudanças bruscas
- Results: Desempenho relativo

**Opções:**
- `--year YEAR`: Ano da temporada
- `--round ROUND`: Número da rodada
- `--laps`: Pré-processar laps
- `--telemetry`: Pré-processar telemetria
- `--all`: Pré-processar tudo
- `--driver DRV`: Filtrar por piloto (ex: VER, HAM)
- `--lap NUM`: Filtrar por volta (apenas telemetria)
- `--save`: Salvar dados processados
- `--show-sample`: Mostrar amostras em formato de tabela

---

### 4. ml_analysis.py

**Análise de ML isolada com tracking MLFlow.** Requer dados já processados pelo `pipeline.py`.

```bash
# Análise completa (clustering + anomaly detection) com tracking
uv run python -m cli.ml_analysis --year 2025 --round 1 --mlflow --show-metrics

# Salvar resultados em data/ml/
uv run python -m cli.ml_analysis --year 2025 --round 1 --mlflow --save

# Apenas clustering
uv run python -m cli.ml_analysis --year 2025 --round 1 --clustering --mlflow

# Apenas detecção de anomalias
uv run python -m cli.ml_analysis --year 2025 --round 1 --anomaly --mlflow

# Piloto específico
uv run python -m cli.ml_analysis --year 2025 --round 1 --driver VER --mlflow

# Comparar runs anteriores
uv run python -m cli.ml_analysis --compare --experiment "F1_2025_Round_01" --max-runs 5

# Mostrar melhor run (maior silhouette)
uv run python -m cli.ml_analysis --compare --experiment "F1_2025_Round_01" --best

# Visualizar no MLFlow UI
uv run mlflow ui   # Acesse http://localhost:5000
```

**Opções principais:**
- `--year YEAR` / `--round ROUND`: Identificar a corrida
- `--driver CODE`: Filtrar por piloto (ex: VER, HAM, LEC)
- `--clustering`: Apenas K-Means por piloto
- `--anomaly`: Apenas Isolation Forest
- `--mlflow`: Habilitar tracking MLFlow (métricas + artefatos CSV)
- `--experiment NAME`: Nome do experimento MLFlow (padrão: `F1_YEAR_Round_XX`)
- `--run-name NAME`: Nome do run (opcional, gerado automaticamente)
- `--save`: Salvar DataFrames de resultado em `data/ml/races/`
- `--show-metrics`: Exibir métricas no terminal

**O que é gerado no MLFlow:**
- Métricas de clustering **por piloto** (silhouette_mean, davies_bouldin_mean, etc.)
- Métricas por piloto individual (driver_VER_silhouette, driver_HAM_silhouette, etc.)
- Métricas de anomaly detection (n_anomalies, anomaly_rate, score_mean)
- Artefatos CSV na aba **Artifacts**: `laps_clustered.csv`, `laps_anomalies.csv`, `per_driver_metrics.csv`

---

### 5. list_data.py

**Listar dados disponíveis** no sistema.

```bash
uv run python cli/list_data.py
```

Mostra:
- Corridas extraídas (brutos)
- Corridas processadas
- Tamanhos dos arquivos
- Estrutura de diretórios

---

## Workflow Recomendado

### Workflow Completo (Recomendado)

```bash
# 1. Pipeline completo: extração + pré-processamento + ML básico
uv run python cli/pipeline.py 2025 1

# 2. Análise de ML com tracking MLFlow
uv run python -m cli.ml_analysis --year 2025 --round 1 --mlflow --show-metrics --save

# 3. Visualizar resultados
uv run mlflow ui   # http://localhost:5000
```

### Workflow em Etapas (Mais controle)

```bash
# 1. Extrair dados
uv run python cli/extract.py 2025 1

# 2. Listar o que foi extraído
uv run python cli/list_data.py

# 3. Pré-processar
uv run python cli/preprocess.py --year 2025 --round 1 --all --save

# 4. Análise ML com tracking
uv run python -m cli.ml_analysis --year 2025 --round 1 --mlflow --save
```

---

## Exemplos Práticos

### Exemplo 1: Processar corrida recente

```bash
# Aguardar dados e processar
uv run python cli/pipeline.py 2025 10 --polling --show-sample
```

### Exemplo 2: Analisar apenas um piloto

```bash
# Extrair tudo
uv run python cli/extract.py 2025 1

# Processar apenas telemetria de Verstappen
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --driver VER --save
```

### Exemplo 3: Verificar degradação de pneus

```bash
# Pipeline completo com amostras
uv run python cli/pipeline.py 2025 1 --show-sample

# Depois usar Python para análise detalhada
python -c "
import pandas as pd
laps = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')
print(laps[['Driver', 'degradation_slope']].drop_duplicates())
"
```

---

## Estrutura de Saída

### Dados Brutos (`data/raw/races/`)

```
data/raw/races/
└── 2025/
    └── round_01/
        ├── metadata.json
        ├── laps.parquet
        ├── race_control.parquet
        ├── weather.parquet
        ├── results.parquet
        └── telemetry/
            ├── VER.parquet
            ├── HAM.parquet
            ├── LEC.parquet
            └── ...
```

### Dados Processados (`data/processed/races/`)

```
data/processed/races/
└── 2025/
    └── round_01/
        ├── laps_processed.parquet
        ├── race_control_processed.parquet
        ├── weather_processed.parquet
        ├── results_processed.parquet
        └── telemetry/
            ├── VER_processed.parquet
            ├── HAM_processed.parquet
            ├── LEC_processed.parquet
            └── ...
```

### Resultados de ML (`data/ml/races/`)

```
data/ml/races/
└── 2025/
    └── round_01/
        ├── laps_clustered.parquet      # K-Means: diferentes ritmos de pilotagem
        ├── laps_anomalies.parquet      # Isolation Forest: voltas anômalas
        └── anomalies_summary.parquet   # Resumo de anomalias por piloto
```

---

## Troubleshooting

### Comando não encontrado

```bash
# Sempre use 'uv run' antes do comando
uv run python cli/pipeline.py 2025 1
```

### Dados não disponíveis

```bash
# Use --polling para corridas recentes
uv run python cli/pipeline.py 2025 10 --polling
```

### Cache corrompido

```bash
# Limpar cache do FastF1
rm -rf ~/.cache/fastf1/
```

### Arquivo não encontrado (pré-processamento)

```bash
# Verificar se dados foram extraídos primeiro
uv run python cli/list_data.py

# Extrair se necessário
uv run python cli/extract.py 2025 1
```

---

## Performance

| Operação | Tempo (com cache) | Tamanho |
|----------|-------------------|---------|
| Extração completa | ~30-60s | ~11-15MB |
| Pré-processamento | ~10-20s | ~8-12MB |
| Machine Learning | ~5-15s | ~2-5MB |
| Pipeline completo | ~45-95s | ~25-35MB |

**Nota:** Primeira execução é mais lenta (download inicial).

## Configuração

Todos os parâmetros do pipeline (num_points, contamination, random_state, etc.) são configuráveis através do arquivo `config.yaml` na raiz do projeto. Veja o arquivo para todas as opções disponíveis.

---

## Documentação Adicional

- [USAGE.md](../USAGE.md) - Guia completo de uso
- [README.md](../README.md) - Visão geral do projeto
- [MLFLOW_SETUP.md](../MLFLOW_SETUP.md) - Documentação do MLFlow e métricas
- [src/extraction/README.md](../src/extraction/README.md) - Módulo de extração
- [src/preprocessing/README.md](../src/preprocessing/README.md) - Módulo de pré-processamento
- [src/ml/README.md](../src/ml/README.md) - Módulo de Machine Learning
