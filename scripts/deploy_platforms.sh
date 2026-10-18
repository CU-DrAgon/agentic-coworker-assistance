#!/usr/bin/env bash
# Deploy `python -m salid.platforms` (LINE / WhatsApp webhooks + Slack / Telegram) as the Cloud Run
# SERVICE salid-platforms, on the salid-bot image (cloudbuild.salid.yaml).
# Usage: scripts/deploy_platforms.sh [--dry-run|--apply]
#   (default)  --dry-run: echo the mutating commands only (nothing is built or changed)
#   --apply    build a new image tag and deploy
# Secrets are NEVER passed here: create them first in Secret Manager (names below) and this script
# only references them with --update-secrets. Plain env is read from the local environment:
#   SALID_WHATSAPP_PHONE_NUMBER_ID, GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_REDIRECT_URI and the
#   optional pass-through SALID_GEMINI_MODEL, SALID_GEMINI_LOCATION, SALID_GUILD_COLLECTION,
#   SALID_FIRESTORE_COLLECTION, SALID_HUB_URL, SALID_SLACK_EXEC_USER_IDS, SALID_LINE_EXEC_USER_IDS,
#   SALID_TELEGRAM_EXEC_USER_IDS, SALID_WHATSAPP_EXEC_NUMBERS, SALID_RECEIPT_BUCKET.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
[ -f "$HOME/google-cloud-sdk/path.bash.inc" ] && source "$HOME/google-cloud-sdk/path.bash.inc"

APPLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) APPLY=false ;;
    --apply) APPLY=true ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PROJECT="${GCP_PROJECT:-meeting-scribe-hackathon}"
REGION="${GCP_REGION:-asia-southeast1}"
SERVICE="salid-platforms"
SA="salid-bot@$PROJECT.iam.gserviceaccount.com"
AR_REPO="cloud-run-source-deploy"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$AR_REPO/salid-bot:$IMAGE_TAG"

if $APPLY; then echo "MODE: APPLY (changes will be made in project $PROJECT)"
else echo "MODE: DRY-RUN (nothing is changed; pass --apply to execute)"; fi

run() {
  printf '    $'; printf ' %q' "$@"; printf '\n'
  if $APPLY; then "$@"; fi
}

echo "==> Building image $IMAGE (Cloud Build, Dockerfile.salid)"
run gcloud builds submit . --project "$PROJECT" --region "$REGION" \
  --config cloudbuild.salid.yaml --substitutions "_IMAGE=$IMAGE"

# Non-secret env. '|' is the pair delimiter because values may contain commas.
ENV_PAIRS=("SALID_STORE=firestore" "SALID_GCP_PROJECT=$PROJECT"
           "SALID_GEMINI_LOCATION=${SALID_GEMINI_LOCATION:-global}" "SALID_TZ=Asia/Bangkok")
for k in SALID_WHATSAPP_PHONE_NUMBER_ID GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_REDIRECT_URI \
         SALID_GEMINI_MODEL SALID_GUILD_COLLECTION SALID_FIRESTORE_COLLECTION SALID_HUB_URL \
         SALID_SLACK_EXEC_USER_IDS SALID_LINE_EXEC_USER_IDS SALID_TELEGRAM_EXEC_USER_IDS \
         SALID_WHATSAPP_EXEC_NUMBERS SALID_RECEIPT_BUCKET; do
  [ -z "${!k:-}" ] || ENV_PAIRS+=("$k=${!k}")
done
ENV_JOINED="$(IFS='|'; echo "${ENV_PAIRS[*]}")"

SECRET_PAIRS=(
  "SALID_SLACK_BOT_TOKEN=salid-platforms-slack-bot-token:latest"
  "SALID_SLACK_APP_TOKEN=salid-platforms-slack-app-token:latest"
  "SALID_LINE_CHANNEL_SECRET=salid-platforms-line-channel-secret:latest"
  "SALID_LINE_CHANNEL_ACCESS_TOKEN=salid-platforms-line-channel-access-token:latest"
  "SALID_TELEGRAM_BOT_TOKEN=salid-platforms-telegram-bot-token:latest"
  "SALID_WHATSAPP_ACCESS_TOKEN=salid-platforms-whatsapp-access-token:latest"
  "SALID_WHATSAPP_APP_SECRET=salid-platforms-whatsapp-app-secret:latest"
  "SALID_WHATSAPP_VERIFY_TOKEN=salid-platforms-whatsapp-verify-token:latest"
  "GOOGLE_OAUTH_CLIENT_SECRET=salid-google-oauth-secret:latest"
  "SALID_TOKEN_KEY=salid-token-key:latest"
)
SECRETS_JOINED="$(IFS=','; echo "${SECRET_PAIRS[*]}")"

# One always-on instance (Slack Socket Mode and Telegram polling must not run twice), CPU always
# allocated (background work after a webhook returns). Unauthenticated: LINE and WhatsApp call it
# from the internet and every webhook verifies its own signature.
echo "==> Deploying Cloud Run service $SERVICE"
run gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" --service-account "$SA" \
  --command python --args=-m,salid.platforms \
  --min-instances 1 --max-instances 1 --no-cpu-throttling --memory 1Gi \
  --allow-unauthenticated \
  --update-env-vars "^|^$ENV_JOINED" --update-secrets "$SECRETS_JOINED" --quiet

echo
if $APPLY; then
  URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
    --format='value(status.url)')"
else URL="https://<service-url>"; fi
echo "LINE webhook:     $URL/line/webhook"
echo "WhatsApp webhook: $URL/whatsapp/webhook"
echo "Health:           $URL/health"
$APPLY || echo "Dry run finished. Secrets must already exist in Secret Manager and be readable by $SA."
