# ADR 0001: Deployment target

Status: accepted

## Context

The service is a stateless FastAPI container that needs a PostgreSQL database with the pgvector extension and calls Azure OpenAI. Traffic is low and bursty, the project is a portfolio piece that must not generate surprise costs, and the deployment should be reproducible from the repository.

## Options considered

| Option | For | Against |
|---|---|---|
| Azure Container Apps with Azure Database for PostgreSQL Flexible Server | Same cloud as the model endpoint, so traffic and identity stay in one place. Scales to zero, so an idle service costs almost nothing. Managed identity can replace API keys for the model. Container images come straight from the existing Dockerfile. | The managed database has a fixed monthly cost even when the app is idle, and it is the largest line of the bill. |
| A single VM running Docker Compose | Cheapest to reason about, closest to the local setup. | Patching, TLS and restarts become manual work, and nothing scales. |
| A serverless platform from another cloud | Generous free tiers. | The model call leaves the Azure network, and a second provider adds credentials and a second set of tools to maintain. |

## Decision

Deploy to Azure Container Apps, with Azure Database for PostgreSQL Flexible Server using the smallest burstable tier and the `vector` extension enabled. Configuration comes from environment variables and secrets of the platform, never from the image. A budget alert is created before anything is deployed.

## Consequences

- The database is the main cost. If the budget alert fires, the database is stopped between demos.
- The app must stay stateless: sessions and appointments move to PostgreSQL before deployment, which is already on the roadmap.
- The Dockerfile must build a minimal image that runs as a non-root user.
- Switching clouds later means rewriting the deployment definition, not the application.
