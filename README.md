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

With Docker (the API and a PostgreSQL database with pgvector; the FAQ index is stored in the
database by default):

```bash
docker compose up --build
```

The compose project is named `vet`. The host ports default to 8000 for the API and 5432 for the
database, and can be changed with `API_PORT` and `DB_PORT`:

```bash
API_PORT=23000 DB_PORT=23432 docker compose up --build
```

To run without Azure credentials, set `LLM_BACKEND=local`: a deterministic, rule-based stand-in
for the model that is meant for demos and offline checks, not for judging answer quality.

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

The set has 63 cases. The first 30 were written together with the rules. The next 33 were
written afterwards with new phrasings: on those, the rules first scored 68% overall (recall of
53% on emergencies and 50% on injections, plus a false positive on an ordinary question). The
patterns were then extended and the set now passes, so 100% is a regression guard and not an
estimate of how well the rules generalize. Keyword rules stay brittle against phrasings nobody
has thought of, which is why they are only the first layer.

Retrieval has its own set, `data/eval/faq.json`, with 30 questions and the document and section that should answer each one:

```bash
python -m vet_agent.evaluation_retrieval
```

With the offline hashing embedder, the correct section is in the top 3 for 90% of the questions and first for 80%. The misses are paraphrases that share no words with the document, such as "what time do you open and close" against a section titled "Funcionamento da clínica". `tests/test_evaluation_datasets.py` requires at least 85% at top 3. These numbers do not apply to the Azure embeddings, which still have to be measured.

Routing has a third set, `data/eval/routing.json`, with 30 messages and their expected intent. `python -m vet_agent.evaluation_routing` classifies them with the real model, so it needs the Azure credentials and has not been run yet. The tests only check the dataset and the runner's arithmetic with a scripted classifier.

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
PGVECTOR_TEST_URL=postgresql://vet:vet@127.0.0.1:5432/vet_agent pytest tests/test_pgvector_integration.py
```

It has been run against the `pgvector/pgvector:pg16` image from `docker-compose.yml`, and the
compose stack answers the bath price question from `servicos.md` through `/faq/ingest` and
`/faq/ask`.

## License

Released under the [MIT License](LICENSE).

## Known limitations

- The agent has only been tested with scripted LLM responses; it has not been run against a live
  Azure OpenAI deployment yet.
- The emergency and injection checks are keyword-based, so they are a first line of defense and do
  not cover every phrasing.
- Appointments and session memory are kept in memory and are lost on restart.
