# Tune Coach

Tune Coach includes two runnable products in one repo:

1. Desktop app (PySide6) for local practice.
2. Web app (FastAPI backend + React frontend) for browser access across devices.

Both are kept in this branch. Desktop behavior is not replaced by the web stack.

## What Is In This Repo

- Desktop entrypoint: `python -m tune_coach`
- Web backend entrypoint: `python -m tune_coach.web.server`
- Web frontend source: `webapp/`
- Docker deployment files: `Dockerfile`, `docker-compose.yml`, `deploy/`

## Prerequisites

- Python `3.10+`
- Node.js `20+` and `npm` (for web frontend build/dev)
- Docker Engine + Docker Compose plugin (for container deployment)

## 1) Desktop App Install And Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m tune_coach
```

Equivalent CLI entrypoint:

```bash
tune-coach
```

## 2) Web App Local Run (No Docker)

This mode is for local testing and UI iteration.

### Backend

```bash
source .venv/bin/activate
python -m tune_coach.web.server
```

Backend serves on `http://localhost:8000`.

### Frontend build for backend static hosting

```bash
cd webapp
npm install
npm run build
cd ..
```

After build, open `http://localhost:8000`.

### Frontend dev server (optional)

```bash
cd webapp
npm run dev
```

Use this for hot reload while developing UI.

## 3) Web App Docker Deploy (LAN + HTTPS)

Docker deploy exposes the app at:

- `https://<server-ip>:18863`

Start or rebuild:

```bash
docker compose up -d --build
```

Stop:

```bash
docker compose down
```

Check running status:

```bash
docker compose ps
curl -k https://<server-ip>:18863/api/health
```

## 4) Update Procedure

Use this when code changed and you want to refresh deployment.

```bash
git pull
docker compose up -d --build --remove-orphans
docker image prune -f
```

If only Python/frontend code changed, this is enough. No manual container cleanup is needed.

## 5) iPhone/iPad HTTPS Certificate Trust

`deploy/Caddyfile` uses `tls internal`. Mobile browsers require trusted HTTPS for microphone APIs.

Export local CA certificate:

```bash
docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt ./deploy/caddy-local-root.crt
```

Install and trust `deploy/caddy-local-root.crt` on iOS/iPadOS, then reopen Safari/Chrome and grant mic permission.

## 6) Auto Start After Linux Reboot

Install service from repo:

```bash
sudo mkdir -p /opt/tune_coach
sudo rsync -a --delete ./ /opt/tune_coach/
sudo cp deploy/tune-coach.service /etc/systemd/system/tune-coach.service
sudo systemctl daemon-reload
sudo systemctl enable --now docker
sudo systemctl enable --now tune-coach
```

Check:

```bash
systemctl status tune-coach
docker compose -f /opt/tune_coach/docker-compose.yml ps
```

## 7) Quick Troubleshooting

- White page on `:8000`: run `npm run build` in `webapp/` first.
- Web keyboard no sound in Safari: click any control once to unlock audio context, then press keys again.
- `python -m tune_coach.web.server` warning about `sys.modules`: use `uvicorn tune_coach.web.server:app --host 0.0.0.0 --port 8000`.
- Mobile mic unavailable: HTTPS or certificate trust is not ready.
