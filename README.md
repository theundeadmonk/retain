# retain
The memory layer built for real-time agents. Fast enough for voice, smart enough to remember.

## Features

- **<50ms hot-path reads** — single-row DB lookup, no LLM calls
- **<5ms fire-and-forget writes** — facts and tasks written during calls
- **Post-call extraction** — LLM analyses transcripts asynchronously, never during calls
- **Token-budgeted profile synthesis** — pre-computed caller profile fits in any system prompt
- **Task lifecycle** — open/in-progress/resolved tracking for agent workflows
- **Rule-based deduplication** — extracted facts deduped against existing memories

## Quickstart

```python
from retain import Memory

memory = Memory()
```

### Hot path — during a call (<50ms reads, <5ms writes)

```python
# Call start: get everything the agent needs
ctx = await memory.context("guest", "hashed_phone_abc123")
print(ctx.profile_blob)        # "Guest prefers room 412, has two children."
print(ctx.recent_memories)     # [MemoryRecord, ...]
print(ctx.open_tasks)          # [TaskRecord, ...]

# During call: record a fact the agent learned
await memory.remember(
    "guest", "hashed_phone_abc123",
    memory_type="preference",
    value={"room": "412", "pillows": "extra"},
)

# During call: create a tracked task
task = await memory.create_task(
    "guest", "hashed_phone_abc123",
    task_type="amenity_request",
    description="3 bath towels to room 412",
)

# During call: resolve a task
await memory.complete_task(task.id)
```

### Cold path — after the call (async, returns immediately)

```python
from retain import Memory
from retain.llm import OpenAIProvider
from retain.types import EntityRef, ProcessRequest

llm = OpenAIProvider(
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

memory = Memory(llm=llm)

# After call: submit transcript for extraction
event_id = await memory.process(
    ProcessRequest(
        transcript=[
            {"role": "user", "content": "Hi, I'd like room 412 with extra pillows"},
            {"role": "assistant", "content": "Of course. Any other requests?"},
            {"role": "user", "content": "Yes, 3 bath towels please."},
        ],
        entities=[EntityRef(entity_type="guest", entity_id="hashed_phone_abc123")],
        instructions="Focus on guest preferences, requests, and personal details.",
    ),
    extraction_timeout=30.0,  # optional, defaults to 30s
)

# Poll for completion
while True:
    status = await memory.event_status(event_id)
    if status["status"] in ("completed", "failed"):
        break
    await asyncio.sleep(0.5)

if status["status"] == "completed":
    # Profile is now stored — ready for the next call
    ctx = await memory.context("guest", "hashed_phone_abc123")
    assert ctx.profile_blob is not None
```

## Installation

```bash
pip install retain
```

For PostgreSQL + pgvector (production):
```bash
pip install retain[postgres]
```

## API

### Memory

| Method | Latency | Description |
|--------|---------|-------------|
| `context(entity_type, entity_id)` | <50ms | Profile + recent memories + open tasks |
| `remember(entity_type, entity_id, type, value)` | <5ms | Store a typed fact |
| `create_task(entity_type, entity_id, type, description)` | <5ms | Create a tracked task |
| `complete_task(task_id)` | <5ms | Resolve a task |
| `list_tasks(entity_type, entity_id, status?)` | <50ms | List tasks, optionally filtered |
| `process(ProcessRequest)` | <5ms (async) | Submit transcript for background extraction |
| `event_status(event_id)` | <5ms | Poll extraction status |

### LLM Providers

```python
from retain.llm import OpenAIProvider, MockProvider

# Any OpenAI-compatible endpoint (OpenAI, DeepSeek, Grok, etc.)
provider = OpenAIProvider(
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

# For testing
mock = MockProvider(response="deterministic output")
memory = Memory(llm=provider)
```

## License

Apache 2.0
