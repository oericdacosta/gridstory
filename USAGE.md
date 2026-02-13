# Guia de Uso - Extração de Dados F1

Este guia mostra como usar o módulo de extração de dados do **F1 Race Intelligence Platform**.

## Instalação

O projeto usa `uv` para gerenciamento de dependências:

```bash
# Instalar dependências (incluindo fastf1, pandas, numpy e pyarrow)
uv sync
```

**Nota**: Não é necessário ativar o ambiente virtual manualmente. Use `uv run` antes dos comandos Python.

## Estrutura dos Dados Extraídos

Os dados são organizados da seguinte forma:

```
data/raw/
├── calendar/
│   └── schedule_2025.parquet          # Calendário da temporada
└── races/
    └── 2025/
        └── round_01/
            ├── metadata.json           # Informações do evento
            ├── laps.parquet           # Dados de voltas e estratégia
            ├── race_control.parquet   # Mensagens de controle de corrida
            ├── weather.parquet        # Dados meteorológicos
            ├── results.parquet        # Resultados finais
            └── telemetry/             # Telemetria por piloto (opcional)
                ├── VER.parquet        # Abreviação do piloto
                ├── HAM.parquet
                ├── LEC.parquet
                └── ...
```

## Uso Básico

### 1. Extrair Calendário da Temporada

```bash
uv run python cli/extract.py --calendar 2025
```

Extrai o calendário completo da temporada 2025, incluindo todas as datas e locais.

### 2. Extrair Dados de uma Corrida

```bash
# Primeira corrida de 2025 (sem telemetria)
uv run python cli/extract.py --race 2025 1

# Com telemetria completa de todos os pilotos
uv run python cli/extract.py --race 2025 1 --telemetry
```

### 3. Extrair Múltiplas Corridas

```bash
# Primeiras 5 corridas de 2025
uv run python cli/extract.py --batch 2025 "1,2,3,4,5"

# Com telemetria
uv run python cli/extract.py --batch 2025 "1,2,3,4,5" --telemetry
```

### 4. Modo Polling (Para Corridas Recentes)

Para corridas que acabaram de acontecer, use o modo polling que aguarda a disponibilidade dos dados:

```bash
uv run python cli/extract.py --race 2026 1 --polling
```

O script tentará carregar os dados a cada 5 minutos (até 10 tentativas).

## Uso Avançado (Python)

### Extração Programática

```python
import fastf1
from pathlib import Path
from src.extraction.orchestrator import extract_race_complete

# Habilitar cache
cache_dir = Path.home() / '.cache' / 'fastf1'
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Extrair corrida
race_dir = extract_race_complete(
    year=2025,
    round_number=1,
    use_polling=False,
    save_telemetry=True
)

print(f"Dados salvos em: {race_dir}")
```

### Usar os Módulos Individualmente

```python
from src.extraction.calendar import get_season_schedule, get_next_race
from src.extraction.polling import quick_load_session
from src.extraction.etl import RaceDataETL

# 1. Calendário
schedule = get_season_schedule(2025)
next_race = get_next_race(schedule)
print(f"Próxima corrida: {next_race['event_name']}")

# 2. Carregar sessão
session = quick_load_session(year=2025, round_number=1)

# 3. ETL
etl = RaceDataETL(session)

# Extrair apenas voltas
laps_df = etl.extract_laps_data()

# Extrair apenas telemetria de pilotos específicos (por abreviação)
telemetry = etl.extract_telemetry_data(drivers=['VER', 'HAM', 'LEC'])

# Extrair tudo
all_data = etl.extract_all(save_telemetry=True)

# Salvar
etl.save_to_parquet(all_data, output_dir="data/raw/races")
```

## Dados Extraídos

### A. Dados de Voltas (`laps.parquet`)

Parâmetros críticos para análise de ritmo e estratégia:

- `LapTime_seconds`: Tempo de volta em segundos
- `Sector1Time_seconds`, `Sector2Time_seconds`, `Sector3Time_seconds`: Tempos por setor
- `Compound`: Tipo de pneu (SOFT, MEDIUM, HARD)
- `TyreLife`: Número de voltas do pneu
- `FreshTyre`: Se o pneu era novo
- `IsAccurate`: Filtro de qualidade
- `PitInTime_seconds`, `PitOutTime_seconds`: Tempos de pit stop
- `Stint`: Número do stint

### B. Telemetria (`telemetry/*.parquet`)

Dados de cada piloto ao longo da corrida:

- `Speed`: Velocidade (km/h)
- `RPM`: Rotações do motor
- `Throttle`: Acelerador (0-100%)
- `Brake`: Freio acionado (True/False)
- `nGear`: Marcha atual
- `DRS`: DRS ativo (True/False)
- `Distance`: Distância percorrida

### C. Controle de Corrida (`race_control.parquet`)

Mensagens da direção de prova:

- Bandeiras amarelas
- Safety Car / Virtual Safety Car
- Penalidades
- Outros eventos

### D. Clima (`weather.parquet`)

Condições meteorológicas:

- `AirTemp`: Temperatura do ar
- `TrackTemp`: Temperatura da pista
- `Rainfall`: Chuva (True/False)
- `WindSpeed`: Velocidade do vento
- `Humidity`: Umidade
- `Pressure`: Pressão atmosférica

### E. Resultados (`results.parquet`)

Classificação final:

- `Position`: Posição final
- `GridPosition`: Posição no grid
- `Points`: Pontos conquistados
- `Status`: Status final (Finished, Collision, etc.)

## Exemplos de Teste

Execute os testes de exemplo:

```bash
uv run python tests/test_extraction/test_basic.py
```

Isso executará:
1. Extração básica da primeira corrida de 2025
2. Extração do calendário completo

## Notas Importantes

### Cache

O FastF1 usa um sistema de cache local para evitar downloads repetidos. O cache é armazenado em:

```
~/.cache/fastf1/
```

### Performance

- **Sem telemetria**: ~5-10 segundos por corrida (com cache)
- **Com telemetria**: ~30-60 segundos por corrida (com cache)
- **Primeira execução**: Mais lento devido ao download inicial dos dados

### Disponibilidade dos Dados

- Os dados ficam disponíveis **imediatamente** após o fim da transmissão oficial
- Para corridas ao vivo, use o modo `--polling`
- Para corridas antigas (2018+), os dados estão sempre disponíveis

### Testando com Dados de 2025

Como sugerido no planejamento, use dados de 2025 para testar enquanto a temporada 2026 não começa. A estrutura dos objetos é consistente entre temporadas.

## Próximos Passos

Após a extração, os dados estão prontos para:

1. **Análise de ML** (Ruptures, Scikit-learn) - Detectar degradação de pneus, undercuts, etc.
2. **Validação** (Pydantic) - Estruturar em JSON para LLM
3. **Geração de Relatórios** (DSPY) - Criar narrativas jornalísticas
4. **Chatbot Interativo** (Agno) - Responder perguntas sobre a corrida

Consulte o `README.md` principal para a arquitetura completa do projeto.
