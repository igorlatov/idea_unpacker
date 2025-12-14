# Idea Unpacker

Multi-model agentic flow for unpacking topics into compressed, insightful outputs.

## Flow Overview

```
User Input → Claude (ideas) → GPT+DeepSeek (parallel scoring) 
          → User Checkpoint → DeepSeek (format/criteria) 
          → Claude (draft) ↔ DeepSeek (evaluate) [max 3 cycles]
          → Output or Failure Analysis
```

## Setup

```bash
pip install -r requirements.txt
```

Set API keys:
```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export DEEPSEEK_API_KEY="sk-..."
```

## Run

```bash
python main.py
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator and CLI |
| `steps.py` | Individual step functions |
| `llm_clients.py` | API wrappers for each model |
| `schemas.py` | Pydantic data models |
| `config.py` | Settings and API keys |

## Configuration

Edit `config.py` to adjust:
- `MAX_REFINEMENT_CYCLES` — default 3
- `PLATEAU_THRESHOLD` — improvement threshold before early stop
- `SCORE_DIVERGENCE_THRESHOLD` — flags contested ideas
- `WORD_LIMIT` — hard cap on output length

## Key Design Choices

- **Disagreement as signal**: High score delta between models flags genuinely novel territory
- **Compression by default**: Word limits enforced, not suggested
- **Parallel scoring**: GPT + DeepSeek run concurrently to avoid anchoring
- **Plateau detection**: Stops grinding when improvement stalls
- **Provenance trace**: Logs which model contributed what
