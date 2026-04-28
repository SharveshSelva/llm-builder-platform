#!/usr/bin/env bash
# One-time GCP setup. Run: gcloud auth login && bash deploy/setup.sh
# Prerequisites: gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
set -euo pipefail

REGION="us-central1"
PROJECT_ID="ai-platform-$(date +%s)"

echo "==> Creating project: $PROJECT_ID"
gcloud projects create "$PROJECT_ID" --name="AI Platform"
gcloud config set project "$PROJECT_ID"

echo "==> Linking billing account (required for Cloud Run)"
echo "    Open: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
echo "    Link a billing account, then press Enter to continue..."
read -r

echo "==> Enabling APIs"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com

echo "==> Creating Artifact Registry repository"
gcloud artifacts repositories create ai-platform \
  --repository-format=docker \
  --location="$REGION" \
  --description="AI Platform Docker images"

echo "==> Creating GCS bucket for ChromaDB persistence"
gsutil mb -l "$REGION" "gs://$PROJECT_ID-chroma"

echo "==> Storing secrets in Secret Manager"
echo "Paste your GROQ_API_KEY and press Enter:"
read -r GROQ_API_KEY
printf '%s' "$GROQ_API_KEY" | gcloud secrets create groq-api-key --data-file=-

echo "Paste your REDIS_URL (from Upstash) and press Enter:"
read -r REDIS_URL
printf '%s' "$REDIS_URL" | gcloud secrets create redis-url --data-file=-

echo "Paste your TAVILY_API_KEY (or press Enter to skip):"
read -r TAVILY_API_KEY
if [ -n "$TAVILY_API_KEY" ]; then
  printf '%s' "$TAVILY_API_KEY" | gcloud secrets create tavily-api-key --data-file=-
else
  printf '%s' "none" | gcloud secrets create tavily-api-key --data-file=-
fi

echo "==> Creating GitHub Actions service account"
SA_EMAIL="github-actions@$PROJECT_ID.iam.gserviceaccount.com"
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions deployer"

for role in \
  roles/run.admin \
  roles/artifactregistry.writer \
  roles/cloudbuild.builds.editor \
  roles/secretmanager.secretAccessor \
  roles/storage.objectAdmin \
  roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="$role"
done

# Also allow Cloud Run to access secrets
CR_SA="service-$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')@serverless-robot-prod.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$CR_SA" \
  --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$CR_SA" \
  --role="roles/storage.objectAdmin"

echo "==> Generating service account key for GitHub Actions"
gcloud iam service-accounts keys create /tmp/gcp-sa-key.json \
  --iam-account="$SA_EMAIL"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "PROJECT_ID = $PROJECT_ID"
echo "REGION     = $REGION"
echo ""
echo "Next steps:"
echo "1. Add GitHub secret GCP_SA_KEY with contents of /tmp/gcp-sa-key.json"
echo "2. In deploy/backend.cloudrun.yaml  — replace PROJECT_ID and REGION"
echo "3. In deploy/frontend.cloudrun.yaml — replace PROJECT_ID and REGION"
echo "4. In .github/workflows/ci.yml      — replace PROJECT_ID and REGION"
echo "5. git push main → GitHub Actions deploys automatically"
echo ""
echo "Backend URL will be:"
echo "  https://ai-platform-backend-<hash>-$REGION.a.run.app"
echo "  (shown in Actions log after first deploy)"
echo "============================================================"
