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

### Ferramentas CLI Futuras

Planejadas:
- `train_ml.py`: Treinar modelos ML com dados extraídos
- `generate_report.py`: Gerar relatórios de corrida
- `serve.py`: Executar servidor FastAPI

## Exemplos

```bash
# Extrair temporada completa de 2025 com telemetria
uv run python cli/extract.py --batch 2025 "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24" --telemetry

# Extrair última corrida (modo polling)
uv run python cli/extract.py --race 2025 24 --polling --telemetry
```
