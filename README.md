# vet-agent

An AI agent that handles customer service for a veterinary clinic: it answers FAQs with RAG, books
and manages appointments through tool calling, and escalates emergencies to a human. Portfolio
project built on FastAPI, LangGraph and Azure OpenAI, using entirely fictional data.

The clinic content, prompts and user-facing messages are in Brazilian Portuguese (pt-BR); code,
tests and documentation are in English.

## Features

- **FAQ with RAG:** answers about opening hours, services, prices, vaccines and policies, grounded
  in retrieved passages and returned with their sources.
- **Appointment management:** book, reschedule and cancel through a tool-calling contract with
  validated arguments and structured errors.
- **Triage and escalation:** emergencies (seizures, poisoning, trauma, etc.) are handed off to a
  human without involving the LLM.
- **Guardrails:** no diagnoses, prescriptions or doses; prompt injection resistance; personal data
  masking (LGPD, the Brazilian data protection law).
- **Evaluation:** a labeled dataset and a runner that report accuracy, precision and recall for
  the guardrails.

## Stack

Python 3.11+, FastAPI, LangGraph, Azure OpenAI, PostgreSQL + pgvector, pytest, ruff, Docker,
GitHub Actions.

## Roadmap

- [x] 1. Skeleton: API, Docker Compose, tests and CI
- [x] 2. Appointment scheduling with tools and tests
- [x] 3. Clinic FAQ with RAG (pgvector)
- [x] 4. LangGraph agent, guardrails and evaluation
- [ ] 5. Observability, cloud deployment and a web front end

## Getting started

```bash
cp .env.example .env   # fill in your Azure OpenAI credentials
pip install -e ".[dev]"
pytest
uvicorn vet_agent.main:app --reload
```

With Docker:

```bash
docker compose up --build
```

The test suite uses scripted LLM responses, so it runs offline without any credentials.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/chat` | Plain chat with the LLM (`{"message": "..."}`) |
| GET | `/appointments/slots?day=YYYY-MM-DD` | Free slots for a day |
| POST | `/appointments` | Book an appointment |
| GET, PATCH, DELETE | `/appointments/{id}` | Fetch, reschedule or cancel |
| POST | `/faq/ingest` | Index the FAQ documents |
| POST | `/faq/ask` | Ask a FAQ question (`{"question": "..."}`) |
| POST | `/agent/chat` | Talk to the agent (`{"message": "...", "session_id": "optional"}`) |

## Agent (LangGraph)

`POST /agent/chat` runs a LangGraph graph with per-session memory (`session_id`).

```
guard_input ─┬─ (emergency or injection) ─────────────────────────────┐
             └─ classify ─┬─ faq ──────────► faq_answer ─────────────►│
                          ├─ medical_advice ► medical_refusal ───────►│
                          ├─ smalltalk ─────► smalltalk ─────────────►├─► finalize
                          └─ scheduling ─► plan_scheduling ⇄ run_tool ►│
```

- **Input (`guard_input`):** masks CPF, email and phone numbers before anything reaches the LLM or
  the conversation history. Emergencies and prompt injection attempts are handled by deterministic
  rules, without calling the LLM. Emergencies are escalated to the human team.
- **Routing:** the LLM classifies the intent as JSON. An invalid response falls back to `smalltalk`.
- **Scheduling:** a planner/tool loop capped at 4 steps. The planner can only use the scheduling
  tools, confirms the details with the customer before booking, rescheduling or cancelling, and
  receives tool errors so it can explain alternatives. Invalid output, an unknown tool or too many
  steps escalate to the front desk.
- **Output (`finalize`):** blocks replies containing doses, prescriptions or diagnoses and masks
  personal data.

### Evaluation

`data/eval/guardrails.json` holds labeled cases (emergency, injection and ordinary messages,
including likely false positives):

```bash
python -m vet_agent.evaluation
```

The report shows accuracy and per-class precision and recall, and `tests/test_guardrails.py`
requires 100% on this set. Evaluating routing and answer quality with the real LLM needs Azure
credentials and is planned for the observability step.

## Scheduling

The clinic is open Monday to Saturday, 08:00 to 18:00, in 30-minute slots. The rules live in
`scheduling/service.py`: no past slots, no double booking, and only active appointments can be
changed. The agent uses the same rules through tool calling: `scheduling/tools.py` exposes
`list_available_slots`, `book_appointment`, `reschedule_appointment` and `cancel_appointment`,
returning `{"ok": false, "error": {"code", "message"}}` on failure.

Appointments are stored in memory behind an interface, so PostgreSQL can replace the store without
touching the service.

## FAQ with RAG

The fictional clinic documents are in `data/faq/*.md`. Each `##` section becomes a passage that is
embedded and stored in a vector store.

- `VECTOR_STORE_BACKEND=memory` (default) or `pgvector` (PostgreSQL with the pgvector extension).
- `EMBEDDING_BACKEND=hashing` (default, local, no credentials) or `azure` (an Azure OpenAI
  embeddings deployment, set in `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`).
- Answers are generated only from the retrieved passages. If none reaches the minimum score
  (`RAG_MIN_SCORE`), the LLM is not called and the API says it could not find the information.
- The question and the context are sent in delimited blocks, and the prompt treats them as data,
  not as instructions.
- Every answer returns the sources used, with a score for each passage.

Index the documents:

```bash
python -m vet_agent.rag.ingest
```

The `hashing` embeddings are lexical and meant for development and offline tests. Use
`EMBEDDING_BACKEND=azure` for real semantic search.

The pgvector integration test runs when a Postgres instance is available:

```bash
docker compose up -d db
PGVECTOR_TEST_URL=postgresql://vet:vet@localhost:5432/vet_agent pytest tests/test_pgvector_integration.py
```

## License

Released under the [MIT License](LICENSE).

## Known limitations

- The agent has only been tested with scripted LLM responses; it has not been run against a live
  Azure OpenAI deployment yet.
- The emergency and injection checks are keyword-based, so they are a first line of defense and do
  not cover every phrasing.
- Appointments and session memory are kept in memory and are lost on restart.
- The pgvector store has not yet been validated against a running PostgreSQL instance.
