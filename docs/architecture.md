# Arquitetura

PitWall AI segue uma arquitetura modular com clara separação de responsabilidades.

## Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────┐
│                   Plataforma PitWall AI                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Módulo 1: The Engine (Dados & ML)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Extração     │→ │  Pipeline ML │→ │  Validação   │     │
│  │ (FastF1)     │  │  (Ruptures)  │  │  (Pydantic)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
                   Timeline de Corrida (JSON)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Módulo 2: The Application (API & LLM)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   FastAPI    │→ │  Jornalista  │→ │   Chatbot    │     │
│  │   Servidor   │  │   (DSPY)     │  │   (Agno)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Detalhes dos Módulos

### Módulo 1: The Engine

**Propósito**: Extrair dados brutos e transformá-los em eventos estruturados

**Componentes**:
- `src/extraction/`: Extração de dados FastF1 [IMPLEMENTADO]
- `src/ml/`: Detecção de eventos usando ML [PLANEJADO]
- `src/models/`: Validação Pydantic [PLANEJADO]

**Fluxo**:
1. Extrair dados de corrida (voltas, telemetria, clima)
2. Detectar eventos (degradação de pneu, undercuts, anomalias)
3. Validar e estruturar como timeline JSON

### Módulo 2: The Application

**Propósito**: Servir dados e gerar insights via API e LLM

**Componentes**:
- `src/api/`: Servidor FastAPI [PLANEJADO]
- `src/llm/`: Geração de relatórios e chat [PLANEJADO]

**Fluxo**:
1. API recebe requisições
2. DSPY gera relatórios de corrida
3. Agno gerencia chat interativo

## Fluxo de Dados

```
API FastF1 → Extração → Arquivos Parquet → Pipeline ML → Timeline JSON
                                                              ↓
                                                         FastAPI
                                                              ↓
                                              ┌───────────────┴───────────┐
                                              ↓                           ↓
                                         Relatório DSPY            Chat Agno
```

## Stack Tecnológica

| Camada | Tecnologia | Status |
|--------|-----------|--------|
| Extração de Dados | FastF1, Pandas, NumPy | Implementado |
| Armazenamento | Parquet (PyArrow) | Implementado |
| ML | Ruptures, Scikit-learn, SciPy | Planejado |
| Validação | Pydantic | Planejado |
| API | FastAPI | Planejado |
| LLM | DSPY, Agno | Planejado |
| Observabilidade | MLflow | Planejado |

## Estrutura do Projeto

Veja [Estrutura do Projeto](../README.md#estrutura-do-projeto) no README principal.
