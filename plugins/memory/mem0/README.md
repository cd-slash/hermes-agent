# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

## Requirements

- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

## Setup

```bash
hermes memory setup    # select "mem0"
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> "$HERMES_HOME/.env"
```

For proxy or self-hosted deployments, set the Mem0 host in `config.yaml`:

```yaml
memory:
  provider: mem0
  mem0:
    host: https://mem0.example.com
```

## Config

Config file: `$HERMES_HOME/config.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `""` | Optional Mem0 API host override for proxy deployments |
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |

`MEM0_API_KEY` stays in `$HERMES_HOME/.env`. If a legacy `$HERMES_HOME/mem0.json` exists with a `host` value, Hermes still honors that host as a read-only fallback override. If no host is configured, Mem0 uses the default cloud API host.

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
