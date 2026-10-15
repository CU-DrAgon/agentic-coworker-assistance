#!/usr/bin/env bash
# Idempotent deploy of the Salid Discord bot as a Cloud Run worker pool.
# Usage: scripts/deploy_salid.sh [--dry-run|--apply] [--rotate-token] [--configure-hub]
#   (default)       --dry-run: only echo the mutating commands (read-only gcloud checks still run)
#   --apply         execute for real
#   --rotate-token  add a new Secret Manager version even if one exists
#   --configure-hub ONLY update env vars on the existing meeting-scribe service (hub settings)
# Reads from the local environment (never from .env, never echoed): SALID_DISCORD_TOKEN.
# Optional pass-through: SALID_RECEIPT_CHANNEL_IDS, SALID_ASK_CHANNEL_IDS, SALID_EXEC_ROLE_NAMES,
# SALID_EXEC_USER_IDS, SALID_DIGEST_TIME, SALID_DIGEST_CHANNEL_ID, SALID_RECEIPT_BUCKET,
# SALID_FIRESTORE_COLLECTION, SALID_GUILD_COLLECTION, SALID_HUB_URL, SALID_GEMINI_MODEL,
# SALID_DEEP_TASK_TIMEOUT, SALID_DEEP_TASK_MODEL. For --configure-hub: HUB_GOOGLE_CLIENT_ID,
# SALID_DISCORD_CLIENT_ID, SALID_GUILD_COLLECTION and (optional super admins) HUB_ADMIN_EMAILS.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
[ -f "$HOME/google-cloud-sdk/path.bash.inc" ] && source "$HOME/google-cloud-sdk/path.bash.inc"

APPLY=false ROTATE=false HUB=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) APPLY=false ;;
    --apply) APPLY=true ;;
    --rotate-token) ROTATE=true ;;
    --configure-hub) HUB=true ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PROJECT="${GCP_PROJECT:-meeting-scribe-hackathon}"
REGION="${GCP_REGION:-asia-southeast1}"
POOL="salid-bot"
SA_NAME="salid-bot"
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
SECRET="salid-discord-token"
AR_REPO="cloud-run-source-deploy"   # existing repo created by `gcloud run deploy --source`
HUB_SERVICE="meeting-scribe"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$AR_REPO/$POOL:$IMAGE_TAG"
BUCKET="${SALID_RECEIPT_BUCKET:-}"

if $APPLY; then echo "MODE: APPLY (changes will be made in project $PROJECT)"
else echo "MODE: DRY-RUN (nothing is changed; pass --apply to execute)"; fi

# run CMD...: print the command, execute it only with --apply.
run() {
  printf '    $'; printf ' %q' "$@"; printf '\n'
  if $APPLY; then "$@"; fi
}

# Read-only probes work in both modes; if gcloud is unusable they report "missing".
gc() { gcloud --project "$PROJECT" "$@"; }

configure_hub() {
  echo "==> Hub settings on existing service $HUB_SERVICE (update only; max-instances untouched)"
  local pairs=("SALID_STORE=firestore") missing=()
  for k in HUB_GOOGLE_CLIENT_ID HUB_ADMIN_EMAILS SALID_DISCORD_CLIENT_ID SALID_GUILD_COLLECTION; do
    if [ -n "${!k:-}" ]; then pairs+=("$k=${!k}"); else missing+=("$k"); fi
  done
  [ ${#missing[@]} -eq 0 ] || echo "    not set locally, skipped: ${missing[*]}"
  # '^|^' makes '|' the pair delimiter, because HUB_ADMIN_EMAILS contains commas.
  local joined; joined="$(IFS='|'; echo "${pairs[*]}")"
  run gcloud run services update "$HUB_SERVICE" --project "$PROJECT" --region "$REGION" \
    --update-env-vars "^|^$joined"
  echo "    meeting-scribe-run needs roles/datastore.user for Firestore (it already has it for meetings)."
}

if $HUB; then configure_hub; echo "Done."; exit 0; fi

if [ -z "${SALID_DISCORD_TOKEN:-}" ] && $APPLY; then
  echo "WARNING: SALID_DISCORD_TOKEN is not set; only an existing secret version can be used." >&2
fi

echo "==> Enabling APIs (only missing ones)"
ENABLED="$(gc services list --enabled --format='value(config.name)' 2>/dev/null || true)"
MISSING_APIS=()
for api in run artifactregistry cloudbuild secretmanager firestore aiplatform; do
  grep -qx "$api.googleapis.com" <<<"$ENABLED" || MISSING_APIS+=("$api.googleapis.com")
done
if [ ${#MISSING_APIS[@]} -gt 0 ]; then run gcloud services enable --project "$PROJECT" "${MISSING_APIS[@]}"
else echo "    all enabled"; fi

echo "==> Service account $SA"
if gc iam service-accounts describe "$SA" >/dev/null 2>&1; then echo "    exists"
else run gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" \
       --display-name "Salid bot (Cloud Run worker pool)"
  if $APPLY; then  # new service accounts can take a few seconds to become visible to IAM
    for _ in 1 2 3 4 5 6; do gc iam service-accounts describe "$SA" >/dev/null 2>&1 && break; sleep 5; done
  fi
fi
for role in roles/datastore.user roles/aiplatform.user roles/logging.logWriter; do
  run gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" \
    --role "$role" --condition=None --quiet
done
if [ -n "$BUCKET" ]; then
  run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member "serviceAccount:$SA" --role roles/storage.objectAdmin
else echo "    SALID_RECEIPT_BUCKET not set: no storage role granted (receipt archive off)"; fi

echo "==> Secret $SECRET"
if gc secrets describe "$SECRET" >/dev/null 2>&1; then echo "    exists"
else run gcloud secrets create "$SECRET" --project "$PROJECT" --replication-policy automatic; fi
HAS_VERSION=false
if gc secrets versions list "$SECRET" --limit 1 --format='value(name)' 2>/dev/null | grep -q .; then
  HAS_VERSION=true
fi
if ! $HAS_VERSION || $ROTATE; then
  if [ -z "${SALID_DISCORD_TOKEN:-}" ] && $APPLY; then
    echo "ERROR: need SALID_DISCORD_TOKEN in the environment to add a secret version" >&2; exit 1
  fi
  # The token goes via stdin only; it is never echoed or placed on a command line.
  echo '    $ printf %s "$SALID_DISCORD_TOKEN" | gcloud secrets versions add' \
       "$SECRET --project $PROJECT --data-file=-   # value hidden"
  if $APPLY; then
    printf '%s' "$SALID_DISCORD_TOKEN" | gcloud secrets versions add "$SECRET" \
      --project "$PROJECT" --data-file=- >/dev/null
    echo "    stored a new version of $SECRET"
  fi
else echo "    a version exists; keeping it (use --rotate-token to add a new one)"; fi
run gcloud secrets add-iam-policy-binding "$SECRET" --project "$PROJECT" \
  --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor

echo "==> Building image $IMAGE (Cloud Build, Dockerfile.salid)"
run gcloud builds submit . --project "$PROJECT" --region "$REGION" \
  --config cloudbuild.salid.yaml --substitutions "_IMAGE=$IMAGE"

echo "==> Deploying Cloud Run worker pool $POOL (1 fixed instance)"
ENV_PAIRS=("SALID_STORE=firestore" "SALID_GCP_PROJECT=$PROJECT"
           "SALID_GEMINI_LOCATION=${SALID_GEMINI_LOCATION:-global}" "SALID_TZ=Asia/Bangkok")
for k in SALID_RECEIPT_CHANNEL_IDS SALID_ASK_CHANNEL_IDS SALID_EXEC_ROLE_NAMES SALID_EXEC_USER_IDS \
         SALID_DIGEST_TIME SALID_DIGEST_CHANNEL_ID SALID_RECEIPT_BUCKET SALID_FIRESTORE_COLLECTION \
         SALID_GUILD_COLLECTION SALID_HUB_URL SALID_GEMINI_MODEL SALID_DEEP_TASK_TIMEOUT \
         SALID_DEEP_TASK_MODEL; do
  [ -z "${!k:-}" ] || ENV_PAIRS+=("$k=${!k}")
done
ENV_JOINED="$(IFS='|'; echo "${ENV_PAIRS[*]}")"   # '|' delimiter: values contain commas
# --instances=1 = manual scaling with a fixed count: the Discord gateway needs exactly one
# always-on connection (two would answer every message twice). 1 vCPU / 2 GiB: the Gemini CLI
# (Node) runs next to the Python bot during deep tasks.
run gcloud run worker-pools deploy "$POOL" --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" --service-account "$SA" --instances 1 --cpu 1 --memory 2Gi \
  --set-env-vars "^|^$ENV_JOINED" --set-secrets "SALID_DISCORD_TOKEN=$SECRET:latest" --quiet

echo
if $APPLY; then echo "Deployed worker pool $POOL ($IMAGE)."
else echo "Dry run finished: nothing was changed. Re-run with --apply to deploy $POOL ($IMAGE)."; fi
echo "Logs:  gcloud logging read 'resource.type=cloud_run_worker_pool' --project $PROJECT --limit 20"
echo "Stop:  gcloud run worker-pools update $POOL --region $REGION --instances 0"
echo "Hub:   scripts/deploy_salid.sh --configure-hub   (optional, separate step)"
