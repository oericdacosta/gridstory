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
- ✅ Machine Learning com Scikit-learn (clustering, anomaly detection, pipeline)
- 🚧 Exportação estruturada (próxima fase: Pydantic)
- 🚧 Geração de narrativas com LLM (fase futura: DSPY, Agno, FastAPI)

## Status do Desenvolvimento

| Módulo | Status | Descrição |
|--------|--------|-----------|
| Extração de Dados | ✅ Implementado | FastF1, Pandas, NumPy |
| Pré-processamento | ✅ Implementado | SciPy (interpolação, signal processing, features) + Scikit-learn (imputação, encoding, escalonamento) |
| Machine Learning | ✅ Implementado | Scikit-learn (K-Means, DBSCAN, Isolation Forest, Pipeline) |
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
3. ✅ Executa ML (clustering K-Means, detecção de anomalias Isolation Forest)
4. ✅ Salva dados brutos em `data/raw/races/YEAR/round_XX/`
5. ✅ Salva dados processados em `data/processed/races/YEAR/round_XX/`
6. ✅ Salva resultados de ML em `data/ml/races/YEAR/round_XX/`

### Comandos Individuais (Opcional)

```bash
# Apenas extração (SEMPRE extrai todos os dados)
uv run python cli/extract.py 2025 1

# Apenas pré-processamento (de dados já extraídos)
uv run python cli/preprocess.py --year 2025 --round 1 --all --save
```

### Documentação Completa

- [USAGE.md](USAGE.md) - Guia de uso do pipeline completo
- [PREPROCESSING.md](PREPROCESSING.md) - Guia completo de pré-processamento (todos os dados)
- [docs/configuration.md](docs/configuration.md) - **Guia de configuração** (config.yaml)
- [src/extraction/README.md](src/extraction/README.md) - Documentação do módulo de extração
- [src/preprocessing/README.md](src/preprocessing/README.md) - Documentação do módulo de pré-processamento
- [src/ml/README.md](src/ml/README.md) - Documentação do módulo de Machine Learning (Scikit-learn)
- [cli/README.md](cli/README.md) - Documentação dos CLIs

## Estrutura do Projeto

```
pitwall-ai/
├── cli/                           # Scripts de linha de comando
│   ├── pipeline.py                # Pipeline completo (orquestrador)
│   ├── pipeline_steps/            # Módulos do pipeline
│   │   ├── extraction.py          # Fase 1: Extração
│   │   ├── preprocessing.py       # Fase 2: Pré-processamento
│   │   ├── ml.py                  # Fase 3: Machine Learning
│   │   └── reporting.py           # Formatação de saídas
│   ├── extract.py                 # CLI de extração individual
│   └── preprocess.py              # CLI de pré-processamento individual
├── src/                           # Código-fonte
│   ├── extraction/                # Extração de dados (✅ implementado)
│   ├── preprocessing/             # Pré-processamento (✅ implementado)
│   │   ├── interpolation.py       # Sincronização de telemetria
│   │   ├── signal_processing.py   # Tratamento de sinal
│   │   └── feature_engineering/   # Engenharia de features (modular)
│   │       ├── statistical.py     # Features estatísticas
│   │       ├── domain.py          # Pré-processamento F1
│   │       └── ml_prep.py         # Preparação para ML
│   ├── ml/                        # Machine Learning (✅ implementado)
│   ├── models/                    # Modelos Pydantic (planejado)
│   ├── api/                       # FastAPI (planejado)
│   ├── llm/                       # Integração LLM (planejado)
│   └── utils/                     # Utilitários e configuração
├── tests/                         # Testes automatizados
├── examples/                      # Exemplos de uso
├── data/                          # Dados (não versionado)
│   ├── raw/races/                 # Dados brutos extraídos
│   ├── processed/races/           # Dados pré-processados
│   └── ml/races/                  # Resultados de Machine Learning
├── docs/                          # Documentação
├── notebooks/                     # Jupyter notebooks
├── config.yaml                    # ⚙️ Configuração centralizada
└── main.py                        # Entry point (futuro: servidor API)
```

### Configuração Centralizada

Todos os parâmetros do pipeline estão centralizados em `config.yaml`:
- **Pré-processamento**: num_points, kernel_size, thresholds, etc.
- **Machine Learning**: random_state, contamination, k_range, etc.
- **Diretórios**: Estrutura de dados configurável

Edite `config.yaml` para customizar o comportamento do pipeline sem modificar código.

### Estrutura de Dados Gerada

```
data/
├── raw/races/YEAR/round_XX/              # FASE 1: Extração
│   ├── laps.parquet
│   ├── telemetry/*.parquet
│   ├── race_control.parquet
│   ├── weather.parquet
│   ├── results.parquet
│   └── metadata.json
│
├── processed/races/YEAR/round_XX/        # FASE 2: Pré-processamento
│   ├── laps_processed.parquet
│   ├── telemetry/*_processed.parquet
│   ├── race_control_processed.parquet
│   ├── weather_processed.parquet
│   └── results_processed.parquet
│
└── ml/races/YEAR/round_XX/               # FASE 3: Machine Learning
    ├── laps_clustered.parquet            # Clustering (ritmos)
    ├── laps_anomalies.parquet            # Detecção de anomalias
    └── anomalies_summary.parquet         # Sumário por piloto
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

#### **F. Pré-processamento para Scikit-learn**
- **Imputação**: Preenche valores faltantes (SimpleImputer, KNNImputer)
- **Encoding**: Converte categorias em números (OneHotEncoder para Compound, TrackStatus)
- **Escalonamento**: Normaliza features (StandardScaler, RobustScaler)

**Por quê:** Algoritmos de ML baseados em distância (K-Means, DBSCAN, Isolation Forest) requerem dados completos, numéricos e na mesma escala.

**Formato:** Parquet processado
**Organização:** `data/processed/races/YEAR/round_XX/`

### 3. Machine Learning com Scikit-learn (✅ Implementado)

**Análise não supervisionada para identificar padrões e eventos:**

#### **A. Clusterização (Análise de Ritmo)**
- **K-Means**: Agrupa voltas em ritmos (Puro, Gestão de Pneus, Tráfego)
- **DBSCAN**: Identifica ritmo consistente e detecta ruído automaticamente
- **Aplicações**: Identificar mudanças de estratégia, filtrar tráfego

#### **B. Detecção de Anomalias**
- **Isolation Forest**: Detecta eventos raros e outliers
- **Aplicações**: Erros de piloto, quebras mecânicas, voltas excepcionais
- **Saída**: Flags binários + scores de anomalia

#### **C. Pipeline Integrado**
- **ColumnTransformer**: Pré-processamento em um objeto único
- **Pipeline Scikit-learn**: Encapsula pré-proc + ML
- **run_race_analysis()**: Função de alto nível para análise completa

**Formato:** DataFrames com labels e scores
**Documentação:** [src/ml/README.md](src/ml/README.md)

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
Dados Brutos → NumPy/Pandas/SciPy/Scikit-learn → Parquet (data/processed/)
```
- **Laps:** Features estatísticas, degradação de pneus
- **Telemetria:** Sincronização, limpeza, derivadas
- **Race Control:** Eventos estruturados, severidade
- **Weather:** Tendências, mudanças bruscas
- **Results:** Desempenho relativo, classificação
- **Para ML:** Imputação, Encoding, Escalonamento

### **FASE 3: Machine Learning (✅ Implementado)**
```
Dados Processados → Scikit-learn → DataFrames com Labels/Scores
```
- **K-Means**: Agrupamento de voltas por ritmo
- **DBSCAN**: Detecção de clusters + ruído
- **Isolation Forest**: Detecção de anomalias (eventos raros)
- **Pipeline**: Integração pré-processamento + ML

### **FASE 4: Exportação Estruturada (🚧 Próxima Fase)**
```
DataFrames → Pydantic → JSON Estruturado
```
- Pydantic: Validação e estruturação de eventos
- Schema de eventos (clusters, anomalias, mudanças de ritmo)
- Exportação para consumo downstream

### **FASE 5: LLM & API (🚧 Planejado)**
```
Eventos (JSON) → DSPY/Agno → Narrativas & Chat
```
- DSPY: Geração de relatórios narrativos
- Agno: Chatbot interativo com contexto
- FastAPI: API REST para consultas
- MLflow: Observabilidade e tracing

## Stack Tecnológica

| Camada | Tecnologia | Status | Documentação |
|--------|-----------|--------|--------------|
| Extração | FastF1, Pandas, NumPy | ✅ Implementado | [src/extraction/](src/extraction/README.md) |
| Armazenamento | Parquet (PyArrow) | ✅ Implementado | - |
| Pré-processamento | SciPy (interpolate, signal, stats) | ✅ Implementado | [src/preprocessing/](src/preprocessing/README.md) |
| Pré-proc ML | Scikit-learn (imputers, encoders, scalers) | ✅ Implementado | [PREPROCESSING.md](PREPROCESSING.md) |
| Machine Learning | Scikit-learn (KMeans, DBSCAN, IsolationForest) | ✅ Implementado | [src/ml/](src/ml/README.md) |
| Change Point Detection | Ruptures | 🚧 Próxima Fase | - |
| Validação | Pydantic | 🚧 Próxima Fase | - |
| Observabilidade ML | MLflow | 🚧 Próxima Fase | - |
| API | FastAPI | 📅 Planejado | - |
| LLM | DSPY, Agno | 📅 Planejado | - |

### Legenda
- ✅ Implementado e documentado
- 🚧 Próxima fase (segundo planejamento)
- 📅 Planejado (Módulo 2)

## Documentação

### Guias de Uso
- [USAGE.md](USAGE.md) - Guia de uso do pipeline completo
- [PREPROCESSING.md](PREPROCESSING.md) - Guia completo de pré-processamento (todos os 5 tipos de dados + Scikit-learn)

### Documentação dos Módulos
- [src/extraction/README.md](src/extraction/README.md) - Módulo de extração
- [src/preprocessing/README.md](src/preprocessing/README.md) - Módulo de pré-processamento
- [src/ml/README.md](src/ml/README.md) - Módulo de Machine Learning (Clustering + Anomaly Detection)
- [cli/README.md](cli/README.md) - Ferramentas CLI

### Documentação Técnica
- [docs/](docs/) - Documentação detalhada (arquitetura, API)

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