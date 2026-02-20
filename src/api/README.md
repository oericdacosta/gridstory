# `src/api/` — FastAPI (planned)

REST API for serving race reports and chat interface.

## Status

📅 **Planned — Module 2**

## Planned endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/race/{year}/{round}/report` | Returns the DSPY-generated journalistic report |
| `POST` | `/race/{year}/{round}/chat` | Interactive chat via Agno knowledge base |
| `GET` | `/health` | Health check |

## Planned usage

```bash
uv run fastapi dev src/api/main.py
# http://localhost:8000/docs
```

## Dependencies (to install for Module 2)

```bash
uv add fastapi uvicorn
```
