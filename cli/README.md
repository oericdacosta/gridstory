# Ferramentas CLI - PitWall AI

Comandos de linha de comando para executar o pipeline de análise de dados F1.

## Comandos Disponíveis

### 1. pipeline.py — Único ponto de entrada

**Pipeline completo end-to-end:** Extração + Pré-processamento + Machine Learning em um único comando.

```bash
# Pipeline completo
uv run python cli/pipeline.py 2025 1

# Com polling (aguardar dados de corrida recente)
uv run python cli/pipeline.py 2025 1 --polling

# Mostrar amostras dos resultados
uv run python cli/pipeline.py 2025 1 --show-sample
```

**O que faz:**
1. ✅ **FASE 1 - Extração**: Coleta TODOS os dados da corrida (laps, telemetry, race_control, weather, results)
2. ✅ **FASE 2 - Pré-processamento**: Processa TODOS os dados (features, normalização, limpeza)
3. ✅ **FASE 3 - Machine Learning**: Clustering (K-Means/DBSCAN), anomaly detection (Isolation Forest), change point detection (Ruptures/PELT)
4. ✅ Salva resultados em 3 diretórios organizados
5. ✅ Tracking MLFlow automático (config-driven via `mlflow.enabled` em `config.yaml`)

**Estrutura Modular do Pipeline:**

O pipeline delega para módulos especializados em `cli/pipeline_steps/`:
- `extraction.py` - Gerencia extração de dados via FastF1
- `preprocessing.py` - Coordena pré-processamento de 5 tipos de dados
- `ml.py` - Executa análises de ML (clustering, anomalias, change points)
- `reporting.py` - Classe `Reporter` para formatação consistente de saídas

**Opções:**
- `YEAR`: Ano da temporada (ex: 2025)
- `ROUND`: Número da rodada (ex: 1)
- `--polling`: Aguardar disponibilidade dos dados (para corridas recentes)
- `--show-sample`: Mostrar amostras dos dados processados

---

### 2. ruptures_analysis.py — Calibração de penalty

**Ferramenta de calibração de hiperparâmetro** para o algoritmo PELT (change point detection). Use uma vez para descobrir o valor ideal de penalty, depois configure em `config.yaml`.

```bash
# Testar range de penalties (definido em config.yaml > ml.degradation.penalty_search_range)
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow

# Comparar runs anteriores por cliff_rate
uv run python cli/ruptures_analysis.py --compare --experiment "F1_2025_Round_01_Ruptures"

# Analisar piloto específico com penalty atual do config
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --driver VER --show-metrics
```

**Fluxo de calibração:**
1. Rodar `--penalty-search` uma vez → MLFlow loga um run por penalty testada
2. Abrir `uv run mlflow ui` → comparar `cliff_rate` e `cliff_validated_rate`
3. Escolher o melhor valor → setar `ml.degradation.penalty` em `config.yaml`
4. Dali em frente, apenas `pipeline.py` — ele usa o valor do config automaticamente

**Opções:**
- `--year / --round`: Identificar a corrida (lê de `data/ml/races/YEAR/round_XX/laps_anomalies.parquet`)
- `--driver CODE`: Analisar apenas um piloto
- `--penalty-search`: Testar todas as penalties do `penalty_search_range` (config.yaml)
- `--mlflow`: Logar cada penalty como run separado no MLFlow
- `--experiment NAME`: Nome do experimento MLFlow
- `--compare`: Comparar runs anteriores por cliff_rate
- `--save`: Salvar `laps_changepoints.parquet` e `tire_cliffs.parquet`
- `--show-metrics`: Exibir métricas detalhadas no terminal

---

### 3. list_data.py — Utilitário

**Lista dados disponíveis** no sistema de forma organizada.

```bash
uv run python cli/list_data.py
```

Mostra:
- Corridas extraídas (brutos) com tamanhos e número de pilotos
- Corridas pré-processadas
- Estrutura de diretórios

---

## Workflow Recomendado

```bash
# 1. Rodar pipeline completo
uv run python cli/pipeline.py 2025 1

# 2. (Opcional) Verificar MLFlow
uv run mlflow ui   # http://localhost:5000

# 3. (Opcional, uma vez) Calibrar penalty do PELT
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow
# → escolher melhor valor no MLFlow UI → setar em config.yaml

# 4. Verificar dados gerados
uv run python cli/list_data.py
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
            └── ...
```

### Resultados de ML (`data/ml/races/`)

```
data/ml/races/
└── 2025/
    └── round_01/
        ├── laps_clustered.parquet       # K-Means: push/base/degraded por piloto
        ├── laps_anomalies.parquet       # Isolation Forest: voltas anômalas
        ├── anomalies_summary.parquet    # Resumo de anomalias por piloto
        ├── laps_changepoints.parquet    # PELT: regimes de degradação por stint
        ├── tire_cliffs.parquet          # Tire cliffs por (Driver, Stint)
        └── tire_cliffs_summary.parquet  # Sumário de cliffs por piloto
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

### ruptures_analysis sem dados

```bash
# O ruptures_analysis lê laps_anomalies.parquet — rodar pipeline antes
uv run python cli/pipeline.py 2025 1
```

---

## Performance

| Operação | Tempo (com cache) | Tamanho |
|----------|-------------------|---------|
| Extração completa | ~30-60s | ~11-15MB |
| Pré-processamento | ~10-20s | ~8-12MB |
| Machine Learning | ~5-15s | ~2-5MB |
| Pipeline completo | ~45-95s | ~25-35MB |

**Nota:** Primeira execução é mais lenta (download inicial do FastF1).

## Configuração

Todos os parâmetros do pipeline (num_points, contamination, penalty, random_state, mlflow.enabled, etc.) são configuráveis através do arquivo `config.yaml` na raiz do projeto. Veja [docs/configuration.md](../docs/configuration.md) para todas as opções.

---

## Documentação Adicional

- [USAGE.md](../USAGE.md) - Guia completo de uso
- [README.md](../README.md) - Visão geral do projeto
- [MLFLOW_SETUP.md](../MLFLOW_SETUP.md) - Documentação do MLFlow e métricas
- [docs/configuration.md](../docs/configuration.md) - Guia de configuração (config.yaml)
- [src/ml/README.md](../src/ml/README.md) - Módulo de Machine Learning
