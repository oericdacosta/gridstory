# Ferramentas CLI

Ferramentas de linha de comando para PitWall AI.

## Comandos Disponíveis

### extract.py

Extrair dados de corridas de F1 usando FastF1.

```bash
# Extrair calendário
uv run python cli/extract.py --calendar 2025

# Extrair uma corrida
uv run python cli/extract.py --race 2025 1 --telemetry

# Extrair múltiplas corridas
uv run python cli/extract.py --batch 2025 "1,2,3,4,5" --telemetry

# Usar modo polling (para corridas recentes)
uv run python cli/extract.py --race 2025 10 --polling --telemetry
```

**Opções**:
- `--calendar ANO`: Extrair calendário da temporada
- `--race ANO RODADA`: Extrair corrida específica
- `--batch ANO RODADAS`: Extrair múltiplas corridas (separadas por vírgula)
- `--telemetry`: Incluir dados completos de telemetria
- `--polling`: Aguardar disponibilidade dos dados (para corridas recentes)
- `--output-dir DIR`: Diretório de saída customizado (padrão: data/raw/races)

### preprocess.py

Pré-processar dados extraídos usando SciPy (interpolação, signal processing, features estatísticas).

```bash
# Listar dados disponíveis
uv run python cli/list_data.py

# Pré-processar dados de voltas
uv run python cli/preprocess.py --year 2025 --round 1 --laps

# Pré-processar telemetria de um piloto específico
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --driver VER --lap 10

# Pré-processar tudo (voltas + telemetria) e salvar
uv run python cli/preprocess.py --year 2025 --round 1 --all --save

# Processar com filtro de piloto
uv run python cli/preprocess.py --year 2025 --round 1 --laps --driver VER --save

# Ver amostra dos dados em formato de tabela
uv run python cli/preprocess.py --year 2025 --round 1 --laps --show-sample
```

**Opções**:
- `--year ANO`: Ano da temporada (obrigatório)
- `--round NUM`: Número da rodada (obrigatório)
- `--session TIPO`: Tipo de sessão - R, Q, FP1, FP2, FP3, S (padrão: R)
- `--laps`: Pré-processar dados de voltas (features estatísticas)
- `--telemetry`: Pré-processar telemetria (sincronização, signal processing)
- `--all`: Pré-processar tudo
- `--driver CÓDIGO`: Filtrar por piloto (ex: VER, HAM, LEC)
- `--lap NUM`: Filtrar por volta específica (apenas telemetria)
- `--save`: Salvar dados processados em `data/processed/`
- `--show-sample`: Mostrar amostra dos dados em formato de tabela (antes/depois)
- `--track-length METROS`: Comprimento da pista (auto-detectado se não fornecido)

**O que cada tipo faz**:
- `--laps`: Calcula Z-scores, detecta outliers, calcula taxa de degradação de pneu
- `--telemetry`: Sincroniza grid de distância, remove ruído, calcula derivadas (aceleração)

### list_data.py

Listar todos os dados disponíveis no projeto (brutos e pré-processados).

```bash
uv run python cli/list_data.py
```

### Ferramentas CLI Futuras

Planejadas:
- `train_ml.py`: Treinar modelos ML com dados pré-processados
- `generate_report.py`: Gerar relatórios de corrida
- `serve.py`: Executar servidor FastAPI

## Exemplos de Fluxo Completo

```bash
# 1. Ver dados disponíveis
uv run python cli/list_data.py

# 2. Extrair dados de uma corrida
uv run python cli/extract.py --race 2025 1 --telemetry

# 3. Pré-processar os dados extraídos
uv run python cli/preprocess.py --year 2025 --round 1 --all --save

# 4. Ver dados pré-processados
uv run python cli/list_data.py

# Extrair temporada completa de 2025 com telemetria
uv run python cli/extract.py --batch 2025 "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24" --telemetry

# Extrair última corrida (modo polling)
uv run python cli/extract.py --race 2025 24 --polling --telemetry

# Pré-processar múltiplas corridas
for i in 1 2 3; do
  uv run python cli/preprocess.py --year 2025 --round $i --all --save
done
```
