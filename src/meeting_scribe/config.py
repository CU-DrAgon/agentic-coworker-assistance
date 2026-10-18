"""Runtime settings, loaded from environment variables and `.env`."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gcp_project: str = "meeting-scribe-hackathon"
    gcp_region: str = "asia-southeast1"  # Cloud Run, GCS, Firestore
    gemini_location: str = "global"  # Vertex AI endpoint for summary_model (Pro is global-only)
    # Regional endpoint for the flash models: own quota pool (the global flash quota was
    # exhausted on 13 Oct) and the same region as Cloud Run.
    flash_location: str = "asia-southeast1"

    audio_bucket: str = "meeting-scribe-hackathon-audio"
    firestore_collection: str = "meetings"
    pubsub_topic: str = "meeting-finalize"
    # When false, /finish runs the finalize step in-process instead of via Pub/Sub.
    use_pubsub: bool = False

    transcribe_model: str = "gemini-3.5-flash"
    summary_model: str = "gemini-3.1-pro-preview"
    extract_model: str = "gemini-3.5-flash"

    # Language of the meeting audio; transcript stays in this language.
    meeting_language: str = "Thai"

    discord_webhook_url: str = ""

    # Public Discord application id of the Salid bot. Not a secret; it only builds the
    # "Add Salid to Discord" link on the home page. Empty hides the link.
    salid_discord_client_id: str = ""

    # Salid hub (/hub). Empty client id = every /api/hub call fails closed.
    # OAuth 2.0 Web client id (public, not a secret) used to verify Google ID tokens. Any
    # verified Google account may sign in; it only sees guilds it administers.
    hub_google_client_id: str = ""
    # Optional "super admins" (comma-separated Google accounts) who see every guild.
    hub_admin_emails: str = ""
    # Firestore collection with one settings doc per Discord guild (shared with the bot).
    salid_guild_collection: str = "salid_guilds"

    # Command center connectors (`salid.connectors`); each is disabled unless its credentials are
    # set. Secrets: never logged or returned by the API.
    slack_client_id: str = ""
    slack_client_secret: str = ""
    slack_signing_secret: str = ""
    clickup_client_id: str = ""
    clickup_client_secret: str = ""

    # Google Calendar + Meet per member (`salid.google_oauth`). All four must be set or the feature is
    # off (routes answer 503). Secrets: never logged or returned by the API.
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = Field(default="", repr=False)
    google_oauth_redirect_uri: str = ""
    salid_token_key: str = Field(default="", repr=False)  # Fernet key for stored refresh tokens


@lru_cache
def get_settings() -> Settings:
    return Settings()
