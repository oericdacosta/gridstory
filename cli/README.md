# Ferramentas CLI - PitWall AI

Comandos de linha de comando para extração e pré-processamento de dados F1.

## Comandos Disponíveis

### 1. pipeline.py (Recomendado)

**Pipeline completo end-to-end:** Extração + Pré-processamento em um único comando.

```bash
# Pipeline completo
uv run python cli/pipeline.py 2025 1

# Com polling (aguardar dados)
uv run python cli/pipeline.py 2025 1 --polling

# Mostrar amostras
uv run python cli/pipeline.py 2025 1 --show-sample
```

**O que faz:**
1. ✅ Extrai TODOS os dados (laps, telemetry, race_control, weather, results)
2. ✅ Pré-processa TODOS os dados (features, normalização, limpeza)
3. ✅ Salva dados brutos em `data/raw/races/YEAR/round_XX/`
4. ✅ Salva dados processados em `data/processed/races/YEAR/round_XX/`

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

### 4. list_data.py

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

### Workflow Simplificado (Um comando)

```bash
# Tudo em um comando
uv run python cli/pipeline.py 2025 1
```

### Workflow em Etapas (Mais controle)

```bash
# 1. Extrair dados
uv run python cli/extract.py 2025 1

# 2. Listar o que foi extraído
uv run python cli/list_data.py

# 3. Pré-processar
uv run python cli/preprocess.py --year 2025 --round 1 --all --save
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
| Pipeline completo | ~40-80s | ~20-25MB |

**Nota:** Primeira execução é mais lenta (download inicial).

---

## Documentação Adicional

- [USAGE.md](../USAGE.md) - Guia completo de uso
- [README.md](../README.md) - Visão geral do projeto
- [src/extraction/README.md](../src/extraction/README.md) - Módulo de extração
- [src/preprocessing/README.md](../src/preprocessing/README.md) - Módulo de pré-processamento
