#!/usr/bin/env bash
# Cloud Run deployment script for TBCare API
# Requires: gcloud auth, billing enabled, Artifact Registry repo
# Usage: ./scripts/deploy_cloudrun.sh [PROJECT_ID] [REGION] [SERVICE_NAME]

set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-asia-southeast2}"
SERVICE_NAME="${3:-tbcare-api}"
REPO_NAME="tbcare-docker"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "ERROR: PROJECT_ID not set. Pass as first arg or run 'gcloud config set project <ID>'"
  exit 1
fi

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Image: $IMAGE_NAME"

# Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project="$PROJECT_ID" --quiet

# Create Artifact Registry repo if not exists
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repository..."
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID"
fi

# Configure Docker auth
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build and push
echo "Building image..."
docker build -t "$IMAGE_NAME" -f apps/api/Dockerfile .

echo "Pushing image..."
docker push "$IMAGE_NAME"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image="$IMAGE_NAME" \
  --platform=managed \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --allow-unauthenticated \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=10 \
  --set-env-vars="DEMO_ENABLED=false" \
  --timeout=300

# Get service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --platform=managed --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')
echo "Deployed: $SERVICE_URL"

# Warm-up
echo "Warming up /health..."
for i in {1..3}; do
  curl -sf "$SERVICE_URL/api/v1/health" >/dev/null && echo "Health check OK" && break
  sleep 3
done

echo "Done. API URL: $SERVICE_URL"
echo "Web deployment (Vercel): set VITE_API_BASE=$SERVICE_URL"