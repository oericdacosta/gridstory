# `src/llm/` — LLM integration (planned)

DSPY-powered report generation and Agno knowledge base chatbot.

## Status

📅 **Planned — Module 2**

## Planned components

### `reporter.py` — DSPY journalistic report

Receives the three JSONs (`timeline.json`, `race_summary.json`, `driver_profiles.json`) and generates a fact-based journalistic article in Portuguese.

```python
class RaceReportSignature(dspy.Signature):
    """
    Você é um jornalista especializado em Fórmula 1.
    Com base nos dados estruturados da corrida, escreva uma matéria
    jornalística objetiva, usando apenas os fatos fornecidos.
    """
    race_summary: str = dspy.InputField()
    timeline: str = dspy.InputField()
    driver_profiles: str = dspy.InputField()
    report: str = dspy.OutputField()
```

Output: `data/timelines/races/YEAR/round_XX/race_report.md`

### `agent.py` — Agno chatbot

Loads the three JSONs as a `JSONKnowledgeBase` and answers user questions about the race.

```python
agent = Agent(
    model=Claude(id="claude-sonnet-4-6"),
    knowledge=JSONKnowledgeBase(path=timeline_dir),
    search_knowledge=True,
)
```

## Dependencies (to install for Module 2)

```bash
uv add dspy-ai agno
```
