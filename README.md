# PitWall AI

**Pipeline completo de análise de corridas de Fórmula 1 usando FastF1, NumPy, Pandas e SciPy.**

## Sobre o Projeto

PitWall AI é um pipeline de engenharia de dados para análise de corridas de Fórmula 1 que:

1. **Extrai TODOS os dados** de uma corrida usando FastF1
2. **Pré-processa TUDO** com NumPy, Pandas e SciPy
3. **Prepara dados estruturados** prontos para análise ML

**Pipeline atual (implementado):**
- ✅ Extração completa de dados (laps, telemetry, race_control, weather, results)
- ✅ Pré-processamento com SciPy (interpolação, signal processing, features estatísticas)
- 🚧 Pipeline ML (próxima fase: Ruptures, Scikit-learn, Pydantic)
- 🚧 Geração de narrativas com LLM (fase futura: DSPY, Agno, FastAPI)

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

### Pipeline Completo (Extração + Pré-processamento)

```bash
# Um único comando que faz TUDO
uv run python cli/pipeline.py 2025 1

# Com polling (aguardar disponibilidade dos dados)
uv run python cli/pipeline.py 2025 1 --polling

# Mostrar amostras dos dados processados
uv run python cli/pipeline.py 2025 1 --show-sample
```

**O que este comando faz:**
1. ✅ Extrai TODOS os dados da corrida (laps, telemetry, race_control, weather, results)
2. ✅ Pré-processa TODOS os dados (features, normalização, limpeza)
3. ✅ Salva dados brutos em `data/raw/races/YEAR/round_XX/`
4. ✅ Salva dados processados em `data/processed/races/YEAR/round_XX/`

### Comandos Individuais (Opcional)

```bash
# Apenas extração (SEMPRE extrai todos os dados)
uv run python cli/extract.py 2025 1

# Apenas pré-processamento (de dados já extraídos)
uv run python cli/preprocess.py --year 2025 --round 1 --all --save
```

### Documentação Completa

- [USAGE.md](USAGE.md) - Guia de extração de dados
- [PREPROCESSING.md](PREPROCESSING.md) - Guia completo de pré-processamento (todos os dados)
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

### 1. Extração Completa de Dados (✅ Implementado)

**SEMPRE extrai TODOS os dados de uma corrida:**

- **Laps**: Tempos por setor, pit stops, compostos de pneu, desgaste de pneu
- **Telemetria**: Velocidade, RPM, aceleração, freio, DRS, marchas (TODOS os pilotos)
- **Race Control**: Safety Car, bandeiras, penalidades, investigações
- **Weather**: Temperatura do ar/pista, chuva, vento, pressão, umidade
- **Results**: Classificação final, grid de largada, pontos, status

**Formato:** Parquet (eficiente e compacto)
**Organização:** `data/raw/races/YEAR/round_XX/`

### 2. Pré-processamento Completo (✅ Implementado)

**Transforma TODOS os dados brutos em features prontas para análise:**

#### **A. Laps (Voltas e Estratégia)**
- Features estatísticas (Z-score, outliers)
- Taxa de degradação de pneus (regressão linear)
- Estatísticas descritivas por grupo (piloto, composto)

#### **B. Telemetria (Dados do Carro)**
- Sincronização em grid comum (`scipy.interpolate`)
- Remoção de ruído (`scipy.signal`)
- Cálculo de derivadas (aceleração, jerk)
- Detecção e correção de outliers

#### **C. Race Control (Eventos da Corrida)**
- Normalização de timestamps
- Indicadores binários (safety car, bandeiras, penalidades)
- Categorização de eventos
- Severidade do evento (info/warning/critical)

#### **D. Weather (Condições Meteorológicas)**
- Interpolação de valores faltantes
- Normalização de temperaturas
- Tendências climáticas (temperatura subindo/descendo)
- Detecção de mudanças bruscas

#### **E. Results (Classificação Final)**
- Mudança de posições (grid → final)
- Status de finalização (finished/DNF)
- Categorização de DNF (collision/mechanical/electrical)
- Score de desempenho relativo

**Formato:** Parquet processado
**Organização:** `data/processed/races/YEAR/round_XX/`

## Arquitetura

O projeto é um **pipeline de engenharia de dados** com fases bem definidas:

### **FASE 1: Extração (✅ Implementado)**
```
FastF1 API → Extração Completa → Parquet (data/raw/)
```
- Laps, Telemetry, Race Control, Weather, Results
- Cache local do FastF1 para eficiência
- Organização hierárquica por temporada/rodada

### **FASE 2: Pré-processamento (✅ Implementado)**
```
Dados Brutos → NumPy/Pandas/SciPy → Parquet (data/processed/)
```
- **Laps:** Features estatísticas, degradação de pneus
- **Telemetria:** Sincronização, limpeza, derivadas
- **Race Control:** Eventos estruturados, severidade
- **Weather:** Tendências, mudanças bruscas
- **Results:** Desempenho relativo, classificação

### **FASE 3: Machine Learning (🚧 Planejado)**
```
Dados Processados → Ruptures/Scikit-learn → Eventos (JSON)
```
- Ruptures: Change Point Detection (degradação de pneus)
- Isolation Forest: Detecção de anomalias
- DBSCAN/K-Means: Agrupamento de stints
- Pydantic: Validação e estruturação de eventos

### **FASE 4: LLM & API (🚧 Planejado)**
```
Eventos (JSON) → DSPY/Agno → Narrativas & Chat
```
- DSPY: Geração de relatórios narrativos
- Agno: Chatbot interativo com contexto
- FastAPI: API REST para consultas
- MLflow: Observabilidade e tracing

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
- [PREPROCESSING.md](PREPROCESSING.md) - Guia completo de pré-processamento
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