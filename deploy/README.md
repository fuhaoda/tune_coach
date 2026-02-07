# Deploy Notes

## Start Or Rebuild Stack

```bash
docker compose up -d --build
```

Visit:

- `https://<server-ip>:18863`

Check health:

```bash
curl -k https://<server-ip>:18863/api/health
docker compose ps
```

## Update Existing Deployment

```bash
git pull
docker compose up -d --build --remove-orphans
docker image prune -f
```

## Enable Auto Start

```bash
sudo mkdir -p /opt/tune_coach
sudo rsync -a --delete ./ /opt/tune_coach/
sudo cp deploy/tune-coach.service /etc/systemd/system/tune-coach.service
sudo systemctl daemon-reload
sudo systemctl enable --now docker
sudo systemctl enable --now tune-coach
```

## Export Caddy Internal CA Certificate (iPhone/iPad Trust)

```bash
docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt ./deploy/caddy-local-root.crt
```

Install and trust this certificate on mobile devices.
