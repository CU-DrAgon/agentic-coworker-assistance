#!/usr/bin/env bash
# Idempotent deploy of Meeting Scribe to Cloud Run (+ Pub/Sub push subscription).
# Usage: scripts/deploy.sh   (needs gcloud logged in; reads DISCORD_WEBHOOK_URL from .env)
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
[ -f "$HOME/google-cloud-sdk/path.bash.inc" ] && source "$HOME/google-cloud-sdk/path.bash.inc"

PROJECT="${GCP_PROJECT:-meeting-scribe-hackathon}"
REGION="${GCP_REGION:-asia-southeast1}"
BUCKET="${AUDIO_BUCKET:-meeting-scribe-hackathon-audio}"
TOPIC="meeting-finalize"
SERVICE="meeting-scribe"
SUBSCRIPTION="meeting-finalize-push"
SECRET="discord-webhook-url"
RUN_SA_NAME="meeting-scribe-run"
PUSHER_SA_NAME="meeting-scribe-pusher"
RUN_SA="$RUN_SA_NAME@$PROJECT.iam.gserviceaccount.com"
PUSHER_SA="$PUSHER_SA_NAME@$PROJECT.iam.gserviceaccount.com"

gcloud config set project "$PROJECT" >/dev/null
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

echo "==> Enabling APIs"
gcloud services enable secretmanager.googleapis.com run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com pubsub.googleapis.com aiplatform.googleapis.com firestore.googleapis.com

ensure_sa() {  # name, display name
  gcloud iam service-accounts describe "$1@$PROJECT.iam.gserviceaccount.com" >/dev/null 2>&1 \
    || gcloud iam service-accounts create "$1" --display-name "$2"
}

echo "==> Service accounts"
ensure_sa "$RUN_SA_NAME" "Meeting Scribe (Cloud Run runtime)"
ensure_sa "$PUSHER_SA_NAME" "Meeting Scribe (Pub/Sub push identity)"
# Newly created service accounts can take a few seconds to become visible to IAM.
for _ in 1 2 3 4 5 6; do gcloud iam service-accounts describe "$PUSHER_SA" >/dev/null 2>&1 && break; sleep 5; done

for role in roles/aiplatform.user roles/datastore.user roles/pubsub.publisher; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$RUN_SA" --role "$role" \
    --condition=None >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$RUN_SA" \
  --role roles/storage.objectAdmin >/dev/null

echo "==> Discord webhook secret"
if ! gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  gcloud secrets create "$SECRET" --replication-policy automatic >/dev/null
fi
if [ -f .env ] && grep -q '^DISCORD_WEBHOOK_URL=.' .env; then
  # The value goes via stdin only; it is never echoed or placed on a command line.
  grep '^DISCORD_WEBHOOK_URL=' .env | head -n1 | cut -d= -f2- | tr -d '\r"'"'" | tr -d '\n' \
    | gcloud secrets versions add "$SECRET" --data-file=- >/dev/null
  echo "    stored a new version of $SECRET"
elif ! gcloud secrets versions list "$SECRET" --limit 1 --format='value(name)' | grep -q .; then
  echo "ERROR: no DISCORD_WEBHOOK_URL in .env and no existing secret version" >&2; exit 1
fi
gcloud secrets add-iam-policy-binding "$SECRET" --member "serviceAccount:$RUN_SA" \
  --role roles/secretmanager.secretAccessor >/dev/null

# Expected Cloud Run URL, so the OIDC audience can be set in the same deploy.
# On re-runs the service already reports its canonical URL.
URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)' 2>/dev/null || true)"
URL="${URL:-https://$SERVICE-$PROJECT_NUMBER.$REGION.run.app}"

echo "==> Deploying Cloud Run service $SERVICE"
gcloud run deploy "$SERVICE" --quiet --source . --region "$REGION" \
  --service-account "$RUN_SA" \
  --set-env-vars "USE_PUBSUB=true,PUBSUB_VERIFY_AUDIENCE=$URL/pubsub/finalize,PUSHER_SA_EMAIL=$PUSHER_SA" \
  --set-secrets "DISCORD_WEBHOOK_URL=$SECRET:latest" \
  --allow-unauthenticated --timeout 900 --memory 1Gi --cpu 1 --concurrency 20 --max-instances 1

ACTUAL_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
if [ "$ACTUAL_URL" != "$URL" ]; then
  echo "    URL differs from the expected one; updating the OIDC audience to $ACTUAL_URL"
  URL="$ACTUAL_URL"
  gcloud run services update "$SERVICE" --region "$REGION" \
    --update-env-vars "PUBSUB_VERIFY_AUDIENCE=$URL/pubsub/finalize"
fi

echo "==> Pub/Sub push subscription"
gcloud run services add-iam-policy-binding "$SERVICE" --region "$REGION" \
  --member "serviceAccount:$PUSHER_SA" --role roles/run.invoker >/dev/null
# Pub/Sub's service agent must be able to mint OIDC tokens for the pusher identity.
gcloud iam service-accounts add-iam-policy-binding "$PUSHER_SA" \
  --member "serviceAccount:service-$PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com" \
  --role roles/iam.serviceAccountTokenCreator >/dev/null

SUB_ARGS=(--push-endpoint "$URL/pubsub/finalize" --push-auth-service-account "$PUSHER_SA"
          --push-auth-token-audience "$URL/pubsub/finalize" --ack-deadline 600 --min-retry-delay 10s)
if gcloud pubsub subscriptions describe "$SUBSCRIPTION" >/dev/null 2>&1; then
  gcloud pubsub subscriptions update "$SUBSCRIPTION" "${SUB_ARGS[@]}" >/dev/null
else
  gcloud pubsub subscriptions create "$SUBSCRIPTION" --topic "$TOPIC" "${SUB_ARGS[@]}" >/dev/null
fi

echo
echo "Deployed: $URL"
