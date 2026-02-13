# PitWall AI

**Ferramenta de extração e estruturação de dados de Fórmula 1** usando FastF1, Pandas e NumPy.

## Sobre

PitWall AI é uma ferramenta Python para extrair, processar e estruturar dados de corridas de Fórmula 1. Utiliza a biblioteca FastF1 para acessar dados oficiais de telemetria, cronometragem, estratégia e condições meteorológicas de corridas da F1.

### Funcionalidades Implementadas

**Extração Completa de Dados de Corrida:**
- Dados de voltas e estratégia (tempos, pneus, pit stops, stints)
- Telemetria completa por piloto (velocidade, RPM, aceleração, freio, DRS, marchas)
- Mensagens de controle de corrida (Safety Car, bandeiras, penalidades)
- Dados meteorológicos (temperatura do ar/pista, chuva, vento)
- Resultados finais da corrida

**Sistema de Polling:**
- Aguarda disponibilidade de dados para corridas recentes
- Extração automática assim que os dados são publicados

**Organização Inteligente:**
- Dados salvos em formato Parquet (eficiente e compacto)
- Estrutura hierárquica por temporada e rodada
- Telemetria organizada por abreviação do piloto (VER, HAM, LEC, etc.)

**Gerenciamento de Calendário:**
- Extração do calendário completo da temporada
- Identificação automática da próxima corrida

## Instalação

### Pré-requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes)

### Setup

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/pitwall-ai.git
cd pitwall-ai

# Instale as dependências
uv sync
```

## Uso

### Comandos Principais

#### 1. Extrair Calendário da Temporada

```bash
uv run python cli/extract.py --calendar 2025
```

#### 2. Extrair Dados de uma Corrida

```bash
# Sem telemetria (rápido)
uv run python cli/extract.py --race 2025 1

# Com telemetria completa
uv run python cli/extract.py --race 2025 1 --telemetry
```

#### 3. Extrair Múltiplas Corridas

```bash
# Primeiras 5 corridas
uv run python cli/extract.py --batch 2025 "1,2,3,4,5" --telemetry
```

#### 4. Modo Polling (Corridas Recentes)

```bash
# Aguarda disponibilidade dos dados
uv run python cli/extract.py --race 2025 10 --polling --telemetry
```

### Uso Programático

```python
import fastf1
from pathlib import Path
from src.extraction.orchestrator import extract_race_complete

# Habilitar cache do FastF1
cache_dir = Path.home() / '.cache' / 'fastf1'
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Extrair corrida completa
race_dir = extract_race_complete(
    year=2025,
    round_number=1,
    use_polling=False,
    save_telemetry=True
)

print(f"Dados salvos em: {race_dir}")
```

## Estrutura do Projeto

```
pitwall-ai/
├── cli/                        # Command-line tools
│   └── extract.py             # Data extraction CLI
├── src/                       # Source code
│   ├── extraction/            # ✅ Data extraction (implemented)
│   ├── ml/                    # 🚧 ML pipeline (planned)
│   ├── models/                # 🚧 Pydantic models (planned)
│   ├── api/                   # 🚧 FastAPI (planned)
│   └── llm/                   # 🚧 LLM integration (planned)
├── tests/                     # Test suite
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Raw extracted data
│   ├── processed/             # Processed features
│   ├── timelines/             # ML output (JSON)
│   └── models/                # Trained models
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks
└── main.py                    # Main entry point (future API server)
```

### Estrutura dos Dados Extraídos

```
data/raw/
├── calendar/
│   └── schedule_2025.parquet          # Calendário da temporada
└── races/
    └── 2025/
        └── round_01/                   # Australian Grand Prix
            ├── metadata.json           # Informações do evento
            ├── laps.parquet           # Dados de voltas (~36KB)
            ├── race_control.parquet   # Mensagens de controle (~8KB)
            ├── weather.parquet        # Dados meteorológicos (~9KB)
            ├── results.parquet        # Resultados finais (~7KB)
            └── telemetry/             # Telemetria por piloto (~11MB total)
                ├── VER.parquet        # Max Verstappen
                ├── HAM.parquet        # Lewis Hamilton
                ├── LEC.parquet        # Charles Leclerc
                └── ...                # Todos os 20 pilotos
```

## Dados Disponíveis

### Dados de Voltas (`laps.parquet`)

Informações de cada volta de cada piloto:

- Tempos: `LapTime_seconds`, `Sector1Time_seconds`, `Sector2Time_seconds`, `Sector3Time_seconds`
- Estratégia: `Compound` (tipo de pneu), `TyreLife` (voltas do pneu), `FreshTyre`
- Pit stops: `PitInTime_seconds`, `PitOutTime_seconds`
- Contexto: `Stint`, `Position`, `TrackStatus`

### Telemetria (`telemetry/*.parquet`)

Dados de alta frequência de cada piloto:

- `Speed`: Velocidade (km/h)
- `RPM`: Rotações do motor
- `Throttle`: Acelerador (0-100%)
- `Brake`: Freio (True/False)
- `nGear`: Marcha atual
- `DRS`: DRS ativo (True/False)
- `Distance`: Distância percorrida na pista

### Controle de Corrida (`race_control.parquet`)

Mensagens oficiais da direção de prova:

- Safety Car / Virtual Safety Car
- Bandeiras (amarelas, vermelhas)
- Penalidades aplicadas
- Investigações

### Clima (`weather.parquet`)

Condições meteorológicas durante a corrida:

- `AirTemp`: Temperatura do ar (°C)
- `TrackTemp`: Temperatura da pista (°C)
- `Rainfall`: Indicador de chuva
- `WindSpeed`: Velocidade do vento
- `Humidity`, `Pressure`

### Resultados (`results.parquet`)

Classificação final:

- `Position`: Posição final
- `GridPosition`: Posição no grid de largada
- `Points`: Pontos conquistados
- `Status`: Status final (Finished, Collision, +1 Lap, etc.)

## Stack Tecnológica

| Tecnologia | Função |
|------------|--------|
| **FastF1** | API de dados de F1 (telemetria, tempos, estratégia) |
| **Pandas** | Manipulação e transformação de DataFrames |
| **NumPy** | Cálculos vetoriais e processamento numérico |
| **PyArrow** | Leitura/escrita eficiente em formato Parquet |
| **uv** | Gerenciamento de dependências e ambiente |

## Performance

- **Sem telemetria**: ~5-10 segundos por corrida (com cache)
- **Com telemetria**: ~30-60 segundos por corrida (com cache)
- **Primeira execução**: Mais lento devido ao download inicial dos dados

O FastF1 usa cache local (`~/.cache/fastf1/`) para evitar downloads repetidos.

## Exemplos de Análise

Com os dados extraídos, você pode realizar análises como:

```python
import pandas as pd

# Carregar dados de voltas
laps = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')

# Análise 1: Degradação de pneu médio
medium_laps = laps[laps['Compound'] == 'MEDIUM']
degradation = medium_laps.groupby('TyreLife')['LapTime_seconds'].mean()

# Análise 2: Comparar pit stops
pit_stops = laps[laps['PitOutTime_seconds'].notna()]
pit_duration = pit_stops['PitOutTime_seconds'] - pit_stops['PitInTime_seconds']

# Análise 3: Telemetria de ultrapassagem
ver_telemetry = pd.read_parquet('data/raw/races/2025/round_01/telemetry/VER.parquet')
ham_telemetry = pd.read_parquet('data/raw/races/2025/round_01/telemetry/HAM.parquet')
# Comparar DRS, velocidade, etc.
```

## Documentação Adicional

- [USAGE.md](USAGE.md) - Guia completo de uso e exemplos avançados
- [docs/architecture.md](docs/architecture.md) - Arquitetura detalhada do projeto
- [docs/api.md](docs/api.md) - Documentação da API (planejada)
- [docs/ml_pipeline.md](docs/ml_pipeline.md) - Pipeline de ML (planejado)

## Contribuindo

Contribuições são bem-vindas! Sinta-se livre para:

- Reportar bugs
- Sugerir novas features
- Enviar pull requests