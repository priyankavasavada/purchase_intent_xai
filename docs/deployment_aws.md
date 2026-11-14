# Deploying to AWS

The project is ready to run on AWS with **Amazon ECR** (stores the Docker images) and
**Amazon ECS on Fargate** (runs them without managing servers). Nothing here has been
deployed; following these steps creates resources that cost money, so delete them
afterwards (last section). For a free demo on a single server, see `docs/deployment_free_ec2.md`.

## How it fits together

```
your machine ── docker push ──▶ ECR: purchase-intent-api, purchase-intent-dashboard
                                   │
                                   ▼
                     ECS Fargate task "purchase-intent"
                     ├── api        (port 8000)
                     └── dashboard  (port 8501, calls the api at localhost:8000)
```

Both containers run in **one task**, so the dashboard reaches the API at
`http://localhost:8000`, the same way docker-compose links them locally.

| File | Purpose |
|---|---|
| `.env.example` | The settings to fill in: region, account ID, repository names, image tag |
| `deploy/aws/push_images.sh` | Builds both images for x86_64, pushes them to ECR, writes the task definition |
| `deploy/aws/task-definition.json` | ECS task definition template; `${...}` values come from `.env` |

## Before you start

- An AWS account and the AWS CLI, logged in: `aws configure`
- Docker installed and running
- `envsubst` (Linux: usually installed; macOS: `brew install gettext`)
- The IAM role `ecsTaskExecutionRole`, which lets ECS pull images and write logs.
  The AWS console creates it the first time you make a task definition.

## Steps

**1. Fill in the settings**

```bash
cp .env.example .env
# edit .env: AWS_REGION, AWS_ACCOUNT_ID, and optionally the repository names and IMAGE_TAG
```

`.env` is ignored by git. Don't commit account details.

**2. Build and push the images**

```bash
bash deploy/aws/push_images.sh
```

This creates the two ECR repositories if they don't exist, builds the images for
`linux/amd64` (Fargate's architecture, even when you build on an Apple Silicon Mac),
pushes them, and writes `deploy/aws/task-definition.rendered.json`.

**3. Register the task definition**

```bash
aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.rendered.json
```

**4. Create a cluster and run the service**

```bash
aws ecs create-cluster --cluster-name purchase-intent

aws ecs create-service \
  --cluster purchase-intent \
  --service-name purchase-intent \
  --task-definition purchase-intent \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<security-group-id>],assignPublicIp=ENABLED}"
```

Use a public subnet, and a security group that allows inbound TCP 8000 and 8501 from your
IP address only. For anything beyond a demo, put an Application Load Balancer with HTTPS
in front instead of exposing the ports directly.

**5. Open it**

Find the task's public IP in the ECS console (Cluster → Service → Tasks → Network), then:

- API docs: `http://<public-ip>:8000/docs`
- Dashboard: `http://<public-ip>:8501`

Logs from both containers go to CloudWatch, log group `/ecs/purchase-intent`.

## Updating to a new model

1. Train it (`python -m src.models.train_final`); the new version is saved as `purchase_intent_v2`.
2. Set a new `IMAGE_TAG` in `.env` (for example `v2`), then repeat steps 2 and 3.
3. Point the service at the new task definition revision:
   `aws ecs update-service --cluster purchase-intent --service purchase-intent --task-definition purchase-intent`

To roll back, run the same command with the previous revision, e.g. `--task-definition purchase-intent:1`.

## Sizing

The task asks for 1 vCPU and 3 GB of memory: the dashboard holds the data, both models
and the SHAP and LIME explainers in memory. Adjust `cpu` and `memory` in the task
definition if CloudWatch shows otherwise.

## Clean up

```bash
aws ecs update-service --cluster purchase-intent --service purchase-intent --desired-count 0
aws ecs delete-service --cluster purchase-intent --service purchase-intent
aws ecs delete-cluster --cluster purchase-intent
aws ecr delete-repository --repository-name purchase-intent-api --force
aws ecr delete-repository --repository-name purchase-intent-dashboard --force
aws logs delete-log-group --log-group-name /ecs/purchase-intent
```
