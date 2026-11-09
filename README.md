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

Python 3.11+, FastAPI, LangGraph, Azure OpenAI, PostgreSQL + pgvector, Langfuse, pytest, ruff,
Docker, GitHub Actions.

## Roadmap

- [x] 1. Skeleton: API, Docker Compose, tests and CI
- [x] 2. Appointment scheduling with tools and tests
- [x] 3. Clinic FAQ with RAG (pgvector)
- [x] 4. LangGraph agent, guardrails and evaluation
- [x] 5. Observability, PostgreSQL persistence, container hardening and a demo
- [ ] 6. Cloud deployment (target chosen in `docs/adr/0001-deploy-target.md`, nothing deployed yet)
- [ ] 7. Web front end

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

## Demo

The conversation below was recorded from a real run of the Docker Compose stack
(`LLM_BACKEND=local`, PostgreSQL for appointments, sessions and the FAQ index) through
`POST /agent/chat`, with the fictional tutor Ana and her dog Rex. The local backend is a
deterministic rule-based stand-in for the model, so this shows the flow, the tool calls and the
guardrails, not how a real model phrases things. A run against Azure OpenAI has not been done.

```
Tutor: Quanto custa o banho?
Assistente: Segundo os materiais da clínica: Serviços e preços - Banho e tosa O banho custa de
            R$ 60,00 a R$ 110,00, conforme o porte do animal. [...]
            [intent=faq, sources=servicos.md, servicos.md, vacinas.md]
Tutor: Quero agendar um banho
Assistente: Claro! Qual o nome do tutor?
Tutor: Meu nome é Ana
Assistente: Qual o nome do pet?
Tutor: Meu cachorro Rex
Assistente: Para qual dia e horário? Por exemplo: dia 10/11 às 10h.
Tutor: Dia 13/11 às 14h
Assistente: Confirma o agendamento de banho tosa para Rex, tutor Ana, em 13/11/2026 às 14:00?
            (sim/não)
Tutor: sim
Assistente: Agendamento confirmado. Código: 2cd4603c-cfa7-4b78-8c45-5e1b6d679107. Rex, 13/11/2026
            às 14:00.
            [intent=scheduling, tool=book_appointment]
Tutor: Quero remarcar o agendamento 2cd4603c-cfa7-4b78-8c45-5e1b6d679107
Assistente: Para qual novo dia e horário? Por exemplo: dia 10/11 às 10h.
Tutor: Para dia 14/11 às 9h
Assistente: Confirma a remarcação do agendamento 2cd4603c-cfa7-4b78-8c45-5e1b6d679107 para
            14/11/2026 às 09:00? (sim/não)
Tutor: sim
Assistente: Agendamento remarcado para 14/11/2026 às 09:00. Código:
            2cd4603c-cfa7-4b78-8c45-5e1b6d679107.
            [intent=scheduling, tool=reschedule_appointment]
Tutor: Meu cachorro está tendo uma convulsão
Assistente: Isso pode ser uma emergência. Não espere por agendamento: ligue agora para a recepção
            no (11) 4000-0000 e leve o animal à clínica imediatamente. [...]
            [intent=emergency, escalated=true, LLM not called]
```

To reproduce it:

```bash
LLM_BACKEND=local docker compose up --build -d
curl -X POST http://127.0.0.1:8000/faq/ingest
curl -X POST http://127.0.0.1:8000/agent/chat -H 'content-type: application/json'   -d '{"message": "Quanto custa o banho?", "session_id": "demo"}'
```

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
credentials and has not been done.

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

## Observability

Every `/agent/chat` turn can be traced in a self-hosted [Langfuse](https://langfuse.com). It is
opt-in: tracing is on only when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-vet-local LANGFUSE_SECRET_KEY=sk-lf-vet-local   docker compose --profile langfuse up -d --build
```

The interface is at http://127.0.0.1:3000 (`LANGFUSE_PORT` changes it). On first start Langfuse
creates the `vet-agent` project with the keys above and the user `admin@vet-agent.local` with the
password `vet-agent-local`. These are development defaults for a machine-local stack; change them
(`LANGFUSE_USER_PASSWORD`, `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`)
before exposing it anywhere. Langfuse 4 needs six containers (web, worker, PostgreSQL, ClickHouse,
Redis and MinIO) that used about 2 GB of memory when idle, so they sit behind the `langfuse`
profile and plain `docker compose up` does not start them.

What a trace contains:

- one trace per turn, named `agent.chat`, grouped by `session_id` as a Langfuse session;
- one span per graph node (`guard_input`, `classify`, `faq_answer`, `plan_scheduling`, `run_tool`,
  `finalize`, ...), so the duration of each node is visible, with the intent, escalation and
  grounding flags as metadata;
- one generation per LLM call, nested under its node, with the prompt, the reply, the deployment
  name, and input and output tokens read from the Azure OpenAI response.

Node spans do not record their input or output, because `guard_input` receives the raw message
before personal data is masked. LLM prompts are recorded after masking. Embedding calls are not
traced.

Number of LLM calls per turn, which is what drives cost and latency:

| Turn | LLM calls |
|---|---|
| Emergency or prompt injection | 0 |
| Smalltalk or medical advice | 1 (`classify`) |
| FAQ | 2 (`classify`, `faq_answer`), or 1 when no passage passes the minimum score |
| Scheduling | 1 (`classify`) plus 1 per planner step, at most 4 steps; a booking takes 3 |

Not measured yet: tokens per turn, cost per turn and real LLM latency need an Azure OpenAI
deployment, so there is no cost summary to publish. The path was verified end to end with an
Azure-shaped stand-in endpoint that returns token counts: Langfuse showed the tokens and the time
of each node and call. Those figures were artificial and are not reported. Langfuse computes cost
only when the deployment name matches a model it knows, so a custom name needs a model definition
in the project settings.

## Scheduling

The clinic is open Monday to Saturday, 08:00 to 18:00, in 30-minute slots. The rules live in
`scheduling/service.py`: no past slots, no double booking, and only active appointments can be
changed. The agent uses the same rules through tool calling: `scheduling/tools.py` exposes
`list_available_slots`, `book_appointment`, `reschedule_appointment` and `cancel_appointment`,
returning `{"ok": false, "error": {"code", "message"}}` on failure.

Appointments live behind a repository interface with two implementations, selected by
`APPOINTMENT_BACKEND`: `memory` (default outside Docker) and `postgres`. The PostgreSQL schema is
versioned in `migrations/*.sql` and applied by `python -m vet_agent.db.migrate` (the API also
applies pending migrations at startup when the backend is `postgres`). A partial unique index
guarantees that two active appointments never share a slot, and one behavior suite
(`tests/test_appointment_repository.py`) runs against both repositories.

Agent conversations are persisted by the LangGraph PostgreSQL checkpointer when
`SESSION_BACKEND=postgres`, so restarting the API does not lose a conversation in progress. With
`docker compose up` both backends default to `postgres`.

The PostgreSQL tests only run when a database is available:

```bash
docker compose up -d db
POSTGRES_TEST_URL=postgresql://vet:vet@127.0.0.1:5432/vet_agent pytest
```

Each test creates and drops its own database, so the URL only needs a role that can create
databases.

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

- The agent has only been tested with scripted LLM responses and the local rule-based backend; it
  has not been run against a live Azure OpenAI deployment yet, so prompt quality, token usage, cost
  and real latency are unmeasured.
- Two concurrent requests for the same slot are resolved by a unique index in PostgreSQL, but the
  in-memory repository is not meant for concurrent use.
- The emergency and injection checks are keyword-based, so they are a first line of defense and do
  not cover every phrasing.
- Outside Docker, appointments and session memory default to memory and are lost on restart;
  set `APPOINTMENT_BACKEND` and `SESSION_BACKEND` to `postgres` to keep them.
