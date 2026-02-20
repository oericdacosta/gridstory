# Visão Geral da Arquitetura

## Estrutura do Projeto

```
pitwall-ai/
│
├── cli/                        # Ferramentas de linha de comando
│   ├── pipeline.py            # [IMPLEMENTADO] Pipeline completo (único ponto de entrada)
│   ├── pipeline_steps/        # [IMPLEMENTADO] Módulos internos do pipeline
│   │   ├── extraction.py      # Fase 1: Extração
│   │   ├── preprocessing.py   # Fase 2: Pré-processamento
│   │   ├── ml.py              # Fase 3: Machine Learning
│   │   └── reporting.py       # Formatação de saídas
│   ├── ruptures_analysis.py   # [IMPLEMENTADO] Calibração de penalty (penalty-search)
│   ├── list_data.py           # [IMPLEMENTADO] Listagem de dados
│   ├── generate_report.py     # [PLANEJADO] Geração de relatórios
│   └── serve.py               # [PLANEJADO] Servidor API
│
├── src/                        # Módulos de código-fonte
│   ├── extraction/            # [IMPLEMENTADO] Extração de dados
│   │   ├── calendar.py        # Gerenciamento de calendário
│   │   ├── polling.py         # Polling de disponibilidade
│   │   ├── etl.py             # Extract, Transform, Load
│   │   └── orchestrator.py    # Orquestração de extração
│   │
│   ├── preprocessing/         # [IMPLEMENTADO] Pré-processamento SciPy
│   │   ├── interpolation.py   # Sincronização de telemetria
│   │   ├── signal_processing.py # Processamento de sinal
│   │   └── feature_engineering.py # Features estatísticas
│   │
│   ├── ml/                    # [IMPLEMENTADO] Pipeline ML
│   │   ├── pipeline.py        # Orquestração completa (run_race_analysis)
│   │   ├── clustering.py      # K-Means e DBSCAN por piloto
│   │   ├── anomaly_detection.py # Isolation Forest
│   │   ├── change_point.py    # Ruptures/PELT — tire cliffs
│   │   ├── metrics.py         # Métricas de avaliação (silhouette, DB, etc.)
│   │   └── tracking.py        # Integração MLFlow
│   │
│   ├── models/                # [PLANEJADO] Modelos Pydantic
│   │   ├── race.py            # Modelos de dados de corrida
│   │   ├── telemetry.py       # Modelos de telemetria
│   │   ├── events.py          # Modelos de eventos
│   │   └── timeline.py        # Modelo de timeline de saída
│   │
│   ├── api/                   # [PLANEJADO] FastAPI
│   │   ├── app.py             # Aplicação FastAPI
│   │   └── routes/            # Rotas da API
│   │       ├── race.py        # Endpoints de corrida
│   │       └── chat.py        # Endpoints de chat
│   │
│   ├── llm/                   # [PLANEJADO] Integração LLM
│   │   ├── journalist.py      # Geração de relatórios (DSPY)
│   │   ├── chatbot.py         # Chat interativo (Agno)
│   │   └── prompts/           # Templates de prompts
│   │
│   ├── observability/         # [PLANEJADO] Observabilidade
│   │   ├── mlflow_config.py   # Configuração MLflow
│   │   └── tracing.py         # Rastreamento de LLM
│   │
│   └── utils/                 # Utilitários compartilhados
│       ├── config.py          # Configuração
│       └── logger.py          # Logging
│
├── tests/                      # Suite de testes
│   ├── test_extraction/       # [IMPLEMENTADO] Testes de extração
│   ├── preprocessing/         # [IMPLEMENTADO] Testes de pré-processamento (23 testes)
│   ├── test_ml/               # [PLANEJADO] Testes de ML
│   ├── test_api/              # [PLANEJADO] Testes de API
│   └── test_llm/              # [PLANEJADO] Testes de LLM
│
├── data/                       # Diretório de dados (gitignored)
│   ├── raw/                   # Dados brutos extraídos
│   │   ├── calendar/          # Calendários de temporada
│   │   └── races/             # Dados de corrida
│   ├── processed/             # Features processadas para ML
│   ├── timelines/             # Saída do ML (JSON estruturado)
│   └── models/                # Modelos ML treinados
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_prototipagem_ml.ipynb
│   └── 03_engenharia_prompt.ipynb
│
├── docs/                       # Documentação
│   ├── architecture.md        # Arquitetura detalhada
│   ├── api.md                 # Documentação da API
│   └── ml_pipeline.md         # Documentação do pipeline ML
│
├── scripts/                    # Scripts utilitários
│
├── config.yaml                 # Configuração centralizada
├── main.py                     # Entry point principal (futuro: servidor API)
├── README.md                   # Documentação do projeto
├── USAGE.md                    # Guia de uso (extração)
├── PREPROCESSING.md            # Guia de pré-processamento
├── ARCHITECTURE.md             # Este arquivo
├── pyproject.toml             # Dependências
└── .gitignore

```

## Descrição dos Módulos

### Módulos Implementados

#### `src/extraction/`
Extração de dados usando a API FastF1.
- Extrai voltas, telemetria, clima, mensagens de controle
- Salva dados em formato Parquet
- Organiza por temporada/rodada

#### `src/preprocessing/`
Pré-processamento matemático de dados usando SciPy.
- **Interpolação**: Sincroniza telemetria em grid comum (scipy.interpolate)
- **Signal Processing**: Remove ruído, calcula derivadas (scipy.signal)
- **Feature Engineering**: Z-scores, outliers, degradação (scipy.stats)
- Prepara dados para pipeline ML

#### `src/ml/`
Pipeline de machine learning para detecção de eventos.
- **Clustering**: K-Means por piloto (k=3 fixo: push/base/degraded) + DBSCAN complementar
- **Anomaly Detection**: Isolation Forest (contaminação config-driven por perfil de corrida)
- **Change Point Detection**: Ruptures/PELT para detectar tire cliffs por stint
- **Métricas**: Silhouette, Davies-Bouldin, Calinski-Harabasz (por piloto)
- **MLFlow**: Tracking config-driven (parâmetros, métricas, artefatos CSV)

### Módulos Planejados

#### `src/models/`
Modelos Pydantic para validação.
- Garante type safety
- Valida saída do ML
- Serializa para JSON

#### `src/api/`
API REST usando FastAPI.
- Serve timelines de corrida
- Gera relatórios
- Chat interativo

#### `src/llm/`
Funcionalidades baseadas em LLM.
- **Jornalista** (DSPY): Gera relatórios de corrida
- **Chatbot** (Agno): Q&A interativo

#### `src/observability/`
Rastreamento e monitoramento com MLflow.
- Rastreia experimentos de ML
- Monitora chamadas de LLM
- Métricas de performance

## Fluxo de Dados

```
┌─────────────┐
│  API FastF1 │
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│  Extração           │ [IMPLEMENTADO]
│  (src/extraction)   │
│  • FastF1           │
│  • Pandas           │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  Dados Brutos       │
│  (data/raw/)        │
│  • Parquet          │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  Pré-processamento  │ [IMPLEMENTADO]
│  (src/preprocessing)│
│  • SciPy            │
│  • NumPy            │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  Dados Processados  │
│  (data/processed/)  │
│  • Features         │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  Pipeline ML        │ [IMPLEMENTADO]
│  (src/ml/)          │
│  • Ruptures/PELT    │
│  • Scikit-learn     │
│  • MLFlow           │
└──────┬──────────────┘
       │
       ↓
┌─────────────────┐
│  Validação      │ [PLANEJADO]
│  Pydantic       │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  Timeline JSON  │
│  (data/timelines/)│
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  FastAPI        │ [PLANEJADO]
│  (src/api/)     │
└──────┬──────────┘
       │
       ├──→ Relatório DSPY
       │
       └──→ Chat Agno
```

## Stack Tecnológica

| Componente | Tecnologia | Status |
|-----------|-----------|--------|
| Extração de Dados | FastF1, Pandas, NumPy | ✅ Implementado |
| Armazenamento | Parquet (PyArrow) | ✅ Implementado |
| Pré-processamento | SciPy (interpolate, signal, stats) | ✅ Implementado |
| ML | Scikit-learn (KMeans, DBSCAN, IsolationForest) | ✅ Implementado |
| Change Point Detection | Ruptures/PELT (tire cliffs) | ✅ Implementado |
| Observabilidade | MLflow (tracking config-driven) | ✅ Implementado |
| Validação | Pydantic | Planejado |
| API | FastAPI | Planejado |
| LLM | DSPY, Agno | Planejado |
| CLI | argparse | ✅ Implementado |

## Fases de Desenvolvimento

### Fase 1: Extração de Dados [✅ COMPLETA]
- [x] Integração FastF1
- [x] Pipeline ETL
- [x] Armazenamento Parquet
- [x] Ferramentas CLI
- [x] Testes

### Fase 2: Pré-processamento [✅ COMPLETA]
- [x] Sincronização de telemetria (scipy.interpolate)
- [x] Processamento de sinal (scipy.signal)
- [x] Engenharia de features estatísticas (scipy.stats)
- [x] CLI de pré-processamento
- [x] Testes (23 testes, 100% passando)
- [x] Documentação completa

### Fase 3: Pipeline ML [✅ COMPLETA]
- [x] Clustering (K-Means por piloto, k=3, semântica push/base/degraded)
- [x] DBSCAN (análise complementar)
- [x] Detecção de anomalias (Isolation Forest, perfis de contaminação)
- [x] Change Point Detection (Ruptures/PELT, tire cliffs por stint)
- [x] Métricas de avaliação (por piloto: silhouette, Davies-Bouldin)
- [x] MLFlow tracking (config-driven, artefatos CSV)
- [ ] Síntese de eventos estruturados (Pydantic — próxima fase)

### Fase 4: API & LLM [EM BREVE]
- [ ] Servidor FastAPI
- [ ] Geração de relatórios DSPY
- [ ] Chatbot Agno
- [ ] Integração MLflow

### Fase 5: Produção [FUTURO]
- [ ] Deploy
- [ ] Monitoramento
- [ ] CI/CD
- [ ] Documentação

## Princípios de Design

1. **Modularidade**: Cada módulo tem uma responsabilidade única
2. **Testabilidade**: Separação clara facilita testes
3. **Escalabilidade**: Arquitetura suporta adicionar novas funcionalidades
4. **Type Safety**: Pydantic garante consistência de dados
5. **Observabilidade**: MLflow rastreia experimentos e performance

## Detalhamento do Pré-processamento

### Módulos SciPy Utilizados

#### 1. scipy.interpolate
**Sincronização de telemetria em grid comum.**

- **Problema**: Telemetria de pilotos diferentes tem pontos de medição em distâncias diferentes
- **Solução**: Interpolação cúbica spline para criar grid uniforme
- **Benefício**: Permite comparação direta ponto-a-ponto entre pilotos

#### 2. scipy.signal
**Processamento de sinal e remoção de ruído.**

- **Problema**: Sensores têm ruído elétrico e spikes anormais
- **Solução**: Filtros medianos e Savitzky-Golay
- **Benefício**: Dados limpos preservando características físicas reais

#### 3. scipy.stats
**Engenharia de features estatísticas.**

- **Problema**: Identificar voltas anormais e calcular degradação
- **Solução**: Z-scores, regressão linear, estatísticas descritivas
- **Benefício**: Features prontas para ML e detecção de outliers

### Pipeline Típico

```python
# 1. Carregar dados brutos
laps = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')

# 2. Pré-processar (scipy.stats)
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats
enriched = enrich_dataframe_with_stats(laps, include_degradation=True)

# 3. Filtrar outliers
clean = enriched[~enriched['is_outlier']]

# 4. Usar em ML
# ... aplicar Ruptures, Scikit-learn, etc.
```

## Referências

- [Guia de Uso - Extração](USAGE.md)
- [Guia de Pré-processamento](PREPROCESSING.md)
- [Detalhes da Arquitetura](docs/architecture.md)
- [Documentação da API](docs/api.md)
- [Pipeline ML](docs/ml_pipeline.md)
