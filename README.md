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
| Extração de Dados | Implementado | FastF1, Pandas, NumPy |
| Pipeline ML | Planejado | Ruptures, Scikit-learn, SciPy |
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

### Extrair Dados de uma Corrida

```bash
# Extrair primeira corrida de 2025 com telemetria
uv run python cli/extract.py --race 2025 1 --telemetry

# Extrair calendário completo
uv run python cli/extract.py --calendar 2025

# Extrair múltiplas corridas
uv run python cli/extract.py --batch 2025 "1,2,3,4,5" --telemetry
```

Para mais detalhes, veja:
- [src/extraction/README.md](src/extraction/README.md) - Documentação completa de extração
- [USAGE.md](USAGE.md) - Guia de uso avançado
- [cli/README.md](cli/README.md) - Documentação dos CLIs

## Estrutura do Projeto

```
pitwall-ai/
├── cli/                    # Scripts de linha de comando
├── src/                    # Código-fonte
│   ├── extraction/         # Extração de dados (implementado)
│   ├── ml/                 # Pipeline ML (planejado)
│   ├── models/             # Modelos Pydantic (planejado)
│   ├── api/                # FastAPI (planejado)
│   ├── llm/                # Integração LLM (planejado)
│   └── utils/              # Utilitários
├── tests/                  # Testes automatizados
├── data/                   # Dados (não versionado)
├── docs/                   # Documentação
├── notebooks/              # Jupyter notebooks
├── config.yaml             # Configuração centralizada
└── main.py                 # Entry point (futuro: servidor API)
```

## Dados Extraídos

A ferramenta extrai dados completos de corridas:

- **Voltas**: Tempos por setor, pit stops, compostos de pneu, desgaste
- **Telemetria**: Velocidade, RPM, aceleração, freio, DRS, marchas
- **Controle de Corrida**: Safety Car, bandeiras, penalidades
- **Clima**: Temperatura, chuva, vento, pressão
- **Resultados**: Classificação final, pontos, status

Os dados são salvos em formato Parquet para eficiência e organizados por temporada e rodada.

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
| Extração | FastF1, Pandas, NumPy | Implementado |
| Armazenamento | Parquet (PyArrow) | Implementado |
| ML | Ruptures, Scikit-learn, SciPy | Planejado |
| Validação | Pydantic | Planejado |
| API | FastAPI | Planejado |
| LLM | DSPY, Agno | Planejado |
| Observabilidade | MLflow | Planejado |

## Documentação

- [USAGE.md](USAGE.md) - Guia completo de uso
- [ARCHITECTURE.md](ARCHITECTURE.md) - Visão geral da arquitetura
- [docs/](docs/) - Documentação detalhada
- [src/extraction/README.md](src/extraction/README.md) - Módulo de extração

## Testes

```bash
# Executar testes de extração
uv run python tests/test_extraction/test_basic.py
```

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