# Container Deployment (Postgres + Redis)

This setup runs the API alongside Postgres (users + subscriptions) and Redis (subscription cache).

## Quick start

1) Set your secrets in `.env` (Paystack keys, auth secrets, etc.).
2) Start the stack:
   ```bash
   docker compose up --build
   ```
3) The API will be available at `http://localhost:8000`.

## Environment variables

The compose file sets:
- `FINANCIAL_MODELS_USER_DB_URL` for the user store
- `SUBSCRIPTION_STORE_URL` for subscription persistence
- `SUBSCRIPTION_CACHE_URL` for Redis caching

Override them in `.env` if you want different credentials/hosts.

## Migration notes

User data:
- The `users` table schema is compatible between SQLite and Postgres.
- If you need to migrate existing users, export from SQLite and import into Postgres.

Subscription data:
- Subscriptions are a cached view of Paystack and can be rehydrated via `/subscriptions/check`.
- Migrating subscription rows is optional.

## Backups

For a single-host deployment, schedule regular Postgres dumps (for example, daily `pg_dump` to a mounted volume or object storage).
