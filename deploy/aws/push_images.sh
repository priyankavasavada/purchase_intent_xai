#!/usr/bin/env bash
# Build both images, push them to Amazon ECR and write the ECS task definition.
#
# Usage, from the project root (values come from .env, see .env.example):
#     bash deploy/aws/push_images.sh
#
# Needs: Docker, the AWS CLI (logged in with `aws configure`) and envsubst
# (part of gettext; on macOS: brew install gettext).
set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${AWS_REGION:?Set AWS_REGION in .env}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID in .env}"
: "${ECR_API_REPOSITORY:?Set ECR_API_REPOSITORY in .env}"
: "${ECR_DASHBOARD_REPOSITORY:?Set ECR_DASHBOARD_REPOSITORY in .env}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
export AWS_REGION AWS_ACCOUNT_ID ECR_API_REPOSITORY ECR_DASHBOARD_REPOSITORY IMAGE_TAG

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Logging in to ${REGISTRY}"
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"

for repository in "$ECR_API_REPOSITORY" "$ECR_DASHBOARD_REPOSITORY"; do
    aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$repository" >/dev/null 2>&1 \
        || aws ecr create-repository --region "$AWS_REGION" --repository-name "$repository" >/dev/null
done

# Fargate runs x86_64. Building on an Apple Silicon Mac would make arm64 images otherwise.
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker compose build

docker tag purchase-intent-api:latest "${REGISTRY}/${ECR_API_REPOSITORY}:${IMAGE_TAG}"
docker tag purchase-intent-dashboard:latest "${REGISTRY}/${ECR_DASHBOARD_REPOSITORY}:${IMAGE_TAG}"
docker push "${REGISTRY}/${ECR_API_REPOSITORY}:${IMAGE_TAG}"
docker push "${REGISTRY}/${ECR_DASHBOARD_REPOSITORY}:${IMAGE_TAG}"

envsubst < deploy/aws/task-definition.json > deploy/aws/task-definition.rendered.json
echo "Images pushed. Task definition written to deploy/aws/task-definition.rendered.json"
echo "Next: register it with"
echo "    aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.rendered.json"
