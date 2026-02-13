# Visão Geral da Arquitetura

## Estrutura do Projeto

```
pitwall-ai/
│
├── cli/                        # Ferramentas de linha de comando
│   ├── extract.py             # [IMPLEMENTADO] CLI de extração de dados
│   ├── train_ml.py            # [PLANEJADO] Treinamento de ML
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
│   ├── ml/                    # [PLANEJADO] Pipeline ML
│   │   ├── degradation.py     # Degradação de pneus (Ruptures)
│   │   ├── anomaly.py         # Detecção de anomalias (Isolation Forest)
│   │   ├── clustering.py      # Clustering de stints (K-Means/DBSCAN)
│   │   ├── events.py          # Detecção de eventos
│   │   └── orchestrator.py    # Orquestração do pipeline ML
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
├── USAGE.md                    # Guia de uso
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

### Módulos Planejados

#### `src/ml/`
Pipeline de machine learning para detecção de eventos.
- **Degradação**: Change Point Detection (Ruptures)
- **Anomalia**: Detecção de outliers (Isolation Forest)
- **Clustering**: Agrupamento de stints (K-Means/DBSCAN)
- **Eventos**: Síntese de undercuts, tire drops, etc.

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
┌─────────────────┐
│  Extração       │ [IMPLEMENTADO]
│  (src/extraction)│
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  Dados Brutos   │
│  (data/raw/)    │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  Pipeline ML    │ [PLANEJADO]
│  (src/ml/)      │
└──────┬──────────┘
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
| Extração de Dados | FastF1, Pandas, NumPy | Implementado |
| Armazenamento | Parquet (PyArrow) | Implementado |
| ML | Ruptures, Scikit-learn, SciPy | Planejado |
| Validação | Pydantic | Planejado |
| API | FastAPI | Planejado |
| LLM | DSPY, Agno | Planejado |
| Observabilidade | MLflow | Planejado |
| CLI | argparse | Implementado |

## Fases de Desenvolvimento

### Fase 1: Extração de Dados [COMPLETA]
- [x] Integração FastF1
- [x] Pipeline ETL
- [x] Armazenamento Parquet
- [x] Ferramentas CLI
- [x] Testes

### Fase 2: Pipeline ML [EM PROGRESSO]
- [ ] Engenharia de features
- [ ] Detecção de degradação (Ruptures)
- [ ] Detecção de anomalias (Isolation Forest)
- [ ] Síntese de eventos
- [ ] Validação Pydantic

### Fase 3: API & LLM [EM BREVE]
- [ ] Servidor FastAPI
- [ ] Geração de relatórios DSPY
- [ ] Chatbot Agno
- [ ] Integração MLflow

### Fase 4: Produção [FUTURO]
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

## Referências

- [Detalhes da Arquitetura](docs/architecture.md)
- [Documentação da API](docs/api.md)
- [Pipeline ML](docs/ml_pipeline.md)
