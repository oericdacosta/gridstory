# PitWall AI

Plataforma de análise inteligente de corridas de Fórmula 1 usando extração de dados, machine learning e LLM.

## Sobre o Projeto

PitWall AI é uma plataforma completa para análise de corridas de Fórmula 1 que combina:
- Extração de dados detalhados usando FastF1
- Detecção de eventos usando Machine Learning
- Geração de relatórios narrativos usando LLM
- API interativa para consultas sobre corridas

O projeto está sendo desenvolvido em fases modulares, com a fase de extração de dados já implementada.

## Status do Desenvolvimento

| Módulo | Status | Descrição |
|--------|--------|-----------|
| Extração de Dados | ✅ Implementado | FastF1, Pandas, NumPy |
| Pré-processamento | ✅ Implementado | SciPy (interpolação, signal processing, features) |
| Pipeline ML | Planejado | Ruptures, Scikit-learn |
| Validação | Planejado | Pydantic |
| API | Planejado | FastAPI |
| LLM | Planejado | DSPY, Agno |
| Observabilidade | Planejado | MLflow |

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

## Uso Rápido

### 1. Extrair Dados de uma Corrida

```bash
# Extrair primeira corrida de 2025 com telemetria
uv run python cli/extract.py --race 2025 1 --telemetry

# Extrair calendário completo
uv run python cli/extract.py --calendar 2025

# Extrair múltiplas corridas
uv run python cli/extract.py --batch 2025 "1,2,3,4,5" --telemetry
```

### 2. Pré-processar Dados

```bash
# Listar dados disponíveis
uv run python cli/list_data.py

# Pré-processar voltas (features estatísticas)
uv run python cli/preprocess.py --year 2025 --round 1 --laps --save

# Pré-processar telemetria (sincronização, limpeza)
uv run python cli/preprocess.py --year 2025 --round 1 --telemetry --save

# Ver amostra dos dados em tabela
uv run python cli/preprocess.py --year 2025 --round 1 --laps --show-sample
```

### Documentação Completa

- [USAGE.md](USAGE.md) - Guia de extração de dados
- [PREPROCESSING.md](PREPROCESSING.md) - Guia de pré-processamento (SciPy)
- [src/extraction/README.md](src/extraction/README.md) - Documentação do módulo de extração
- [src/preprocessing/README.md](src/preprocessing/README.md) - Documentação do módulo de pré-processamento
- [cli/README.md](cli/README.md) - Documentação dos CLIs

## Estrutura do Projeto

```
pitwall-ai/
├── cli/                    # Scripts de linha de comando
├── src/                    # Código-fonte
│   ├── extraction/         # Extração de dados (✅ implementado)
│   ├── preprocessing/      # Pré-processamento SciPy (✅ implementado)
│   ├── ml/                 # Pipeline ML (planejado)
│   ├── models/             # Modelos Pydantic (planejado)
│   ├── api/                # FastAPI (planejado)
│   ├── llm/                # Integração LLM (planejado)
│   └── utils/              # Utilitários
├── tests/                  # Testes automatizados
├── examples/               # Exemplos de uso
├── data/                   # Dados (não versionado)
├── docs/                   # Documentação
├── notebooks/              # Jupyter notebooks
├── config.yaml             # Configuração centralizada
└── main.py                 # Entry point (futuro: servidor API)
```

## Funcionalidades

### 1. Extração de Dados (✅ Implementado)

A ferramenta extrai dados completos de corridas:

- **Voltas**: Tempos por setor, pit stops, compostos de pneu, desgaste
- **Telemetria**: Velocidade, RPM, aceleração, freio, DRS, marchas
- **Controle de Corrida**: Safety Car, bandeiras, penalidades
- **Clima**: Temperatura, chuva, vento, pressão
- **Resultados**: Classificação final, pontos, status

Os dados são salvos em formato Parquet para eficiência e organizados por temporada e rodada.

### 2. Pré-processamento com SciPy (✅ Implementado)

Motor matemático que transforma dados brutos em features prontas para ML:

- **Sincronização de Telemetria** (`scipy.interpolate`): Alinha dados de diferentes pilotos em grid comum de distância para comparações diretas
- **Processamento de Sinal** (`scipy.signal`): Remove ruído de sensores, calcula derivadas (aceleração), preserva informação importante
- **Features Estatísticas** (`scipy.stats`): Detecta outliers com Z-score, calcula taxa de degradação de pneus, estatísticas descritivas

**Exemplos**:
```python
# Sincronizar telemetria de dois pilotos para comparação
from src.preprocessing.interpolation import synchronize_telemetry
ver_sync = synchronize_telemetry(ver_telemetry, track_length=5281.0)
ham_sync = synchronize_telemetry(ham_telemetry, track_length=5281.0)
speed_delta = ver_sync['Speed'] - ham_sync['Speed']

# Limpar ruído e calcular aceleração
from src.preprocessing.signal_processing import apply_telemetry_pipeline
processed = apply_telemetry_pipeline(telemetry, calculate_derivatives=True)

# Detectar outliers e calcular degradação de pneus
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats
enriched = enrich_dataframe_with_stats(laps_df, group_by=['Driver', 'Stint'])
```

Veja [src/preprocessing/README.md](src/preprocessing/README.md) para documentação completa.

## Arquitetura

O projeto segue uma arquitetura modular com separação clara de responsabilidades:

**Módulo 1: The Engine (Dados & ML)**
- Extração de dados brutos (FastF1)
- Detecção de eventos (Ruptures, Scikit-learn)
- Validação e estruturação (Pydantic)

**Módulo 2: The Application (API & LLM)**
- Servidor REST (FastAPI)
- Geração de relatórios (DSPY)
- Chat interativo (Agno)

Para detalhes completos, veja [ARCHITECTURE.md](ARCHITECTURE.md) e [docs/architecture.md](docs/architecture.md).

## Stack Tecnológica

| Camada | Tecnologia | Status |
|--------|-----------|--------|
| Extração | FastF1, Pandas, NumPy | ✅ Implementado |
| Armazenamento | Parquet (PyArrow) | ✅ Implementado |
| Pré-processamento | SciPy (interpolate, signal, stats) | ✅ Implementado |
| ML | Ruptures, Scikit-learn | Planejado |
| Validação | Pydantic | Planejado |
| API | FastAPI | Planejado |
| LLM | DSPY, Agno | Planejado |
| Observabilidade | MLflow | Planejado |

## Documentação

### Guias de Uso
- [USAGE.md](USAGE.md) - Guia de extração de dados
- [PREPROCESSING.md](PREPROCESSING.md) - Guia de pré-processamento com SciPy
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitetura do projeto

### Documentação dos Módulos
- [src/extraction/README.md](src/extraction/README.md) - Módulo de extração
- [src/preprocessing/README.md](src/preprocessing/README.md) - Módulo de pré-processamento
- [cli/README.md](cli/README.md) - Ferramentas CLI

### Documentação Técnica
- [docs/](docs/) - Documentação detalhada (arquitetura, API, ML pipeline)

## Testes

```bash
# Executar testes de extração
uv run python tests/test_extraction/test_basic.py

# Executar testes de pré-processamento (23 testes)
uv run pytest tests/preprocessing/ -v

# Rodar exemplos práticos
uv run python examples/preprocessing_example.py
```

**Cobertura de Testes:**
- ✅ Extração: Testado manualmente
- ✅ Pré-processamento: 23 testes unitários (100% passando)
- ⏳ ML Pipeline: Planejado
- ⏳ API: Planejado

## Configuração

O arquivo `config.yaml` centraliza todas as configurações do projeto:
- Diretórios de dados
- Parâmetros de extração
- Configurações de ML
- Configurações de API e LLM

## Contribuindo

Contribuições são bem-vindas! Por favor:
- Reporte bugs através das issues
- Sugira novas funcionalidades
- Envie pull requests