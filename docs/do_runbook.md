# DigitalOcean Production Runbook

This runbook covers a production deployment for Angular + API + managed Postgres on DigitalOcean.

## Architecture

- Frontend: Angular build hosted on Spaces + CDN
- API: Droplet running Docker
- Database: Managed Postgres
- TLS: Caddy/Nginx on the Droplet (or DO Load Balancer if multi-node)
- Optional cache: Redis (later)

## 1) Create DigitalOcean resources

- Droplet: Ubuntu 22.04, 2 vCPU / 4 GB RAM to start
- Managed Postgres: basic tier
- Spaces + CDN: static frontend hosting
- Domain + DNS zone

## 2) Set up Postgres

- Create a database and user in DO
- Allow the Droplet IP in trusted sources
- Capture the connection string:
  `postgresql://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require`

## 3) Provision the Droplet

```bash
sudo apt update && sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
```

Re-login to apply docker group membership.

## 4) Deploy the API

- Copy the repo to the Droplet
- Set `.env` with Paystack and auth values
- Set DB variables:
  - `FINANCIAL_MODELS_USER_DB_URL`
  - `SUBSCRIPTION_STORE_URL`

Optional (later):
- `SUBSCRIPTION_CACHE_URL`

Run:
```bash
docker compose up --build -d
```

## 5) TLS and routing

- Create DNS record: `api.yourdomain.com` -> Droplet IP
- Install Caddy or Nginx

Example Caddyfile:
```
api.yourdomain.com {
  reverse_proxy localhost:8000
}
```

## 6) Frontend deployment

- Build Angular: `ng build --configuration production`
- Upload the build output to Spaces
- Enable CDN and configure custom domain `app.yourdomain.com`
- Set Angular API base URL to `https://api.yourdomain.com`

## 7) Backups

- Enable DO Droplet backups or weekly snapshots
- Use DO Managed Postgres backups

## 8) Monitoring

- Enable DO Monitoring (CPU/RAM/Disk)
- Health check: `GET /health`

## Suggested sizing

- Start: 2 vCPU / 4 GB Droplet + small managed Postgres
- Scale API horizontally if report generation load increases
