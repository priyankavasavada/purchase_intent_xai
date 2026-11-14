# Free deployment on one EC2 server

The cheapest way to put the project online: one small EC2 server that runs
`docker compose up`, exactly as on a laptop. No ECR or ECS needed.

`docs/deployment_aws.md` describes the production-style setup (ECR + ECS Fargate). That
one costs money from the first hour; this one can run on AWS free-tier credits.

## What "free" means on AWS (check before you start)

AWS changed its free tier on 15 July 2025, and the rules depend on when your account
was created. Check your own account under **Billing → Free Tier** before launching anything.

| Account created | What you get | Server to use |
|---|---|---|
| On or after 15 July 2025 | **Free plan**: $100 in credits (up to $100 more for completing activities). It ends after 6 months or when the credits run out; nothing is charged unless you upgrade to a paid plan. | `t3.small` (2 GB memory), which is on the free-plan list |
| Before 15 July 2025, still in its first 12 months | 750 hours a month of `t3.micro` | `t3.micro` (1 GB), with the swap file below |

Roughly, a `t3.small` with a public IP and a 20 GB disk uses about $20 of credit a month,
so $100 lasts around five months. Prices vary by region; the Billing page shows real usage.

**When the free plan ends, the server stops.** Record the demo video and take screenshots
while it is live.

Memory: the API uses about 280 MB and the dashboard about 340 MB, so 2 GB is comfortable
and 1 GB works with swap.

## 1. Protect yourself from surprise bills

In the console: **Billing → Budgets → Create budget → Zero spend budget**, with your email.
You get an alert as soon as anything starts costing money.

## 2. Launch the server

**EC2 → Launch instance:**

| Setting | Value |
|---|---|
| Name | `purchase-intent` |
| Image | Ubuntu Server 24.04 LTS (x86_64) |
| Instance type | `t3.small` (or `t3.micro` on an older account) |
| Key pair | Create one, download the `.pem` file |
| Storage | 20 GiB gp3 |
| Security group | SSH (22) from **My IP**; Custom TCP 8501 and 8000 from **Anywhere** |

Launch it and note its **Public IPv4 address**.

The API has no login, so anyone with the address can call it. That is fine for a demo of
anonymous data; don't put anything private behind it.

## 3. Install Docker

```bash
chmod 400 ~/Downloads/purchase-intent.pem
ssh -i ~/Downloads/purchase-intent.pem ubuntu@<public-ip>
```

On the server:

```bash
# Swap: extra memory on disk, needed on 1 GB servers, harmless on 2 GB ones.
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Docker and the compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit
```

Log in again (`ssh ...`) so the Docker group takes effect.

## 4. Get the code and start it

If the GitHub repository is public:

```bash
git clone https://github.com/priyankavasavada/purchase_intent_xai.git
```

If it is private, create a fine-grained personal access token on GitHub with read access
to this repository only, and use it as the password when git asks.

```bash
cd purchase_intent_xai
docker compose up --detach --build     # first build takes about 5-10 minutes
docker compose ps                      # both should say "healthy"
```

Open:

- Dashboard: `http://<public-ip>:8501`
- API docs: `http://<public-ip>:8000/docs`

`restart: unless-stopped` in `docker-compose.yml` brings both back after a reboot.

## 5. Update after a new commit

```bash
cd purchase_intent_xai
git pull
docker compose up --detach --build
```

## 6. Save credits, and clean up

- **Stop** the instance (EC2 → Instance state → Stop) when you're not demoing. A stopped
  server isn't charged for compute, only for its disk. Its public IP changes when you start it again.
- **Terminate** it when you're finished; that deletes the server and its disk.
- If you allocated an Elastic IP, release it too: an unused one is charged.

## Troubleshooting

| Problem | Fix |
|---|---|
| Page doesn't load | Check the security group allows 8501/8000, and `docker compose ps` shows healthy |
| `permission denied` running docker | Log out and back in after `usermod` |
| Build is killed or very slow on `t3.micro` | Make sure swap is on: `free -h` |
| Containers stuck "starting" | `docker compose logs api` shows the error |
