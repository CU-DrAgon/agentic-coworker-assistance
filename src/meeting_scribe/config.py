"""Runtime settings, loaded from environment variables and `.env`."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gcp_project: str = "meeting-scribe-hackathon"
    gcp_region: str = "asia-southeast1"  # Cloud Run, GCS, Firestore
    gemini_location: str = "global"  # Vertex AI Gemini endpoint location

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
