"""Salid runtime settings, loaded from `SALID_*` environment variables and `.env`."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    # Every variable is namespaced with SALID_ so it cannot clash with meeting_scribe's
    # settings (e.g. STORE, GCP_PROJECT) when both packages share one `.env`.
    model_config = SettingsConfigDict(env_prefix="SALID_", env_file=".env", extra="ignore")

    discord_token: str = ""
    # Extra voice-only bot tokens (comma-separated, may be empty): "Salid · 2", "Salid · 3"...
    # Secrets: never logged, never in repr.
    voice_tokens: Annotated[list[str], NoDecode] = Field(default_factory=list, repr=False)
    # Comma-separated in the environment (NoDecode: skip pydantic-settings' JSON parsing).
    receipt_channel_ids: Annotated[set[int], NoDecode] = set()
    ask_channel_ids: Annotated[set[int], NoDecode] = set()
    exec_role_names: Annotated[set[str], NoDecode] = {"ผู้บริหาร", "Executive"}
    exec_user_ids: Annotated[set[int], NoDecode] = set()
    digest_time: str = "18:00"  # HH:MM in `tz`
    digest_channel_id: int | None = None
    store: str = "sqlite"  # sqlite | firestore
    firestore_collection: str = "salid_receipts"
    # Optional original-receipt archive: bucket name (empty = disabled), object prefix.
    receipt_bucket: str = ""
    receipt_prefix: str = "salid-receipts/"
    db_path: str = "./data/ledger.db"
    gcp_project: str = "meeting-scribe-hackathon"
    gemini_location: str = "global"
    gemini_model: str = "gemini-3.5-flash"
    tz: str = "Asia/Bangkok"
    # Google Maps Platform (Routes, Places, Weather). Off until routes/places/weather.googleapis.com
    # are enabled. Auth is ADC; the API key is only a fallback if the Weather API rejects OAuth.
    maps_enabled: bool = False
    maps_api_key: str = Field(default="", repr=False)  # secret: never logged
    # Optional Google Safe Browsing Lookup key for security_check / the link guard (empty = off).
    safe_browsing_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("SAFE_BROWSING_API_KEY", "SALID_SAFE_BROWSING_API_KEY"),
    )  # secret: never logged
    # WeatherNext 2 ensemble in BigQuery (empty = disabled): "<project>.<dataset>"
    weathernext_bq_dataset: str = ""
    weathernext_bq_table: str = "weathernext_2_0_0"
    # Deep tasks (Gemini CLI, an external Node binary). Empty path = feature disabled.
    gemini_cli_path: str = ""
    deep_task_timeout: int = 600  # seconds
    deep_task_model: str = "gemini-3-flash-preview"  # see deep_task.build_cli_settings
    research_daily_cap: int = 10  # deep_research runs per guild per day (salid.research)
    # Cheap model for the "is this addressed to Salid?" yes/no gates (see salid.gate).
    gate_model: str = "gemini-3.5-flash-lite"
    low_confidence: float = 0.7
    # Spoken replies in voice meetings (see salid.speech). The TTS call reuses the bot's Vertex
    # client, so it runs in `gemini_location` (global; the Gemini TTS models work there).
    voice_reply: bool = True
    # "cloud": Cloud Text-to-Speech Chirp 3 HD (~1.5 s). "gemini": Gemini TTS (~6 s, also the
    # fallback when Cloud TTS fails).
    tts_backend: str = "cloud"
    cloud_tts_voice: str = "Leda"  # brightest Thai Chirp 3 HD voice, for the cute-cat persona
    tts_speaking_rate: float = 1.08
    tts_pitch: float = 1.06  # ffmpeg pitch-up factor at playback (1.0 = unchanged)
    tts_model: str = "gemini-2.5-flash-tts"
    tts_voice: str = "Leda"
    # How Salid talks in a voice meeting (needs voice_reply). "live": a Gemini Live native-audio
    # session hears the room and answers ~1.5 s after someone stops talking (see salid.live_voice).
    # "classic": wake word -> transcription -> text answer -> spoken rewrite -> TTS (~11 s).
    voice_mode: str = "live"  # live | classic
    live_model: str = "gemini-live-2.5-flash-native-audio"
    live_location: str = "us-central1"
    live_voice: str = "Leda"
    # Live turn-taking (automatic activity detection): short greetings are picked up quickly (start
    # sensitivity high), a pause is not an end of turn (end sensitivity low + silence_duration_ms).
    live_silence_ms: int = 1200
    live_prefix_padding_ms: int = 300
    live_start_sensitivity: str = "high"  # high | low
    live_end_sensitivity: str = "low"  # high | low
    live_proactive_audio: bool = True  # the model may stay silent when not addressed
    # Barge-in: "intentional" (default) = background noise or someone else talking never cuts Salid
    # off; she stops only when told to ("สลิด หยุด", "เดี๋ยวก่อน", "stop", "wait", or her name).
    # "any" = the old behaviour: any detected speech interrupts her (good in quiet rooms only).
    live_barge_in: str = "intentional"  # intentional | any
    # Voice meetings stay voice: no "🔊" echoes, per-answer text or join announcement in the text
    # chat (the end-of-meeting summary and replies to messages typed to Salid are always posted).
    meeting_text_echo: bool = False
    # ONE person, ONE voice: the politeness particle of Salid's typed replies AND spoken lines
    # (female: ค่ะ, questions use คะ; ครับ also works). SALID_VOICE_PARTICLE is the old name.
    persona_particle: str = Field(
        "ค่ะ",
        validation_alias=AliasChoices(
            "SALID_PERSONA_PARTICLE", "SALID_VOICE_PARTICLE", "persona_particle", "voice_particle"
        ),
    )

    @property
    def voice_particle(self) -> str:  # old name, same setting
        return self.persona_particle

    # Joined by itself / by schedule and nobody addressed Salid `consent_delay_seconds` after the
    # first speech: it asks (once) whether it may take notes. A "no" is honoured for
    # `consent_window_seconds` after the question; silence or an OK means it carries on.
    consent_delay_seconds: float = 20.0
    consent_window_seconds: float = 90.0
    notes_batch_seconds: float = 60.0  # classic mode: how often the gate looks for decisions
    # Two states: listening (speaks only when its name is called) and engaged (answers every
    # utterance). Engaged falls back to listening after this long without anyone addressing it
    # (or sooner: three utterances in a row that are not addressed to Salid).
    engaged_timeout_seconds: float = 45.0
    # Engaged by the name alone (no question): back to listening after this long unless someone
    # addresses Salid (SALID_ENGAGED_BARE_TIMEOUT_SECONDS).
    engaged_bare_timeout_seconds: float = 12.0
    # "Be quiet" / "เงียบก่อน": Salid says nothing and does not wake on a bare name for this long
    # (her name with words clears it) (SALID_HUSH_HOLD_SECONDS).
    hush_hold_seconds: float = 90.0
    # Every listening <-> engaged change (and the join) is shown as one short line in the meeting's
    # text chat; a flip back within 15 s edits the previous line instead of posting a new one.
    voice_state_posts: bool = True
    # Live ask_salid: hard limit for one job. `live_slow_answer_seconds` is no longer used (the
    # background hand-off replaced the "taking longer" line); kept so old env files still load.
    live_ask_timeout_seconds: float = 150.0
    live_slow_answer_seconds: float = 40.0
    # Live background jobs (v0.9.0): ask_salid answers within `live_foreground_seconds`, else the
    # job runs on, the room keeps talking, and the result is announced at the next pause
    # (`live_result_pause_seconds` of no room audio, waited for at most `live_result_wait_seconds`;
    # it is always posted to the text chat). At most `live_max_jobs` run at once per session.
    live_foreground_seconds: float = 12.0
    # How a finished background job is announced by voice: "short" = one sentence ("the result is in
    # the chat"; an answer of <= 160 chars is said in full), "full" = the whole result, "off" =
    # chat only (SALID_LIVE_RESULT_VOICE). Progress is never spoken.
    live_result_voice: str = "short"
    live_max_jobs: int = 3
    live_result_pause_seconds: float = 1.5
    live_result_wait_seconds: float = 90.0
    # Live: the name was heard but the model has not started to answer after this long (counted from
    # the end of the utterance): force a turn. The model's own first audio takes 2.3-4.7 s there
    # (VAD silence + thinking); a cue sent earlier replaced a real answer with a generic greeting.
    live_force_reply_seconds: float = 5.0
    # The name seen in the Live session's own input transcript (fast path, no classic transcription
    # needed): force a turn this long after the speaker went quiet if the model has not answered.
    live_name_reply_seconds: float = 1.5
    # Mouth gate (code, not prompt): the Live model's voice is played only while the session is
    # engaged (someone called the name) or answers a tool call made while engaged. A name call can
    # land in the transcript a moment after the model starts answering, so the first
    # `live_call_grace_seconds` of an unauthorised turn are held back and released if the name shows
    # up in that window. Off = v0.8.0 behaviour.
    live_mouth_gate: bool = True  # SALID_LIVE_MOUTH_GATE
    live_call_grace_seconds: float = 2.5
    # Engaged this recently counts as engaged for the mouth gate and the tool gate (SALID_LIVE_REENGAGE_GRACE_SECONDS).
    live_reengage_grace_seconds: float = 60.0
    # A name hit (either detector) counts only if real speech is behind it: noise / an unmuted mic
    # makes the transcriber hallucinate "หวัดดีครับ สลิด". Frames with RMS >= `name_min_rms` are
    # voiced. Both detectors (bare greeting / name only) apply the same speech-likeness rule:
    # `name_min_voiced_seconds` of voiced audio (0 = gate off), 30% voiced frames, frame-RMS
    # coefficient of variation >= `name_min_cv` (stationary noise is flat), overall RMS >= floor.
    name_min_voiced_seconds: float = 0.35  # SALID_NAME_MIN_VOICED_SECONDS
    name_min_cv: float = 0.25  # SALID_NAME_MIN_CV
    name_min_rms: float = 250.0  # SALID_NAME_MIN_RMS
    # A clip that is ONLY the name ("สลิด") must have this much voiced audio: saying "Salid" takes
    # >= 0.45 s, while one-word clips (so / sure / slide) the transcriber turned into the name are
    # shorter (SALID_NAME_BARE_MIN_VOICED_SECONDS).
    name_bare_min_voiced_seconds: float = 0.45
    # A spoken "one moment" clip while a slow tool (ask_salid) runs. If it is still running after
    # `live_filler_again_seconds` ONE typed note goes to the meeting chat (not spoken).
    live_filler: bool = True
    live_filler_again_seconds: float = 8.0
    # "One moment" lines (chat, Live voice, classic meeting) are written by the cheap model so they
    # fit the request; this long is the most we wait for one before a hand-written line goes out.
    ack_llm: bool = True  # SALID_ACK_LLM
    # Reusable ("generic") lines are kept in a pool (DocStore) with their voice clips on local disk;
    # below `filler_pool_min` lines every request generates a fresh line, after that only
    # `filler_fresh_ratio` of them do and the rest pick the least-used pooled line.
    filler_pool_min: int = 20
    filler_pool_max: int = 200  # lines per (kind, language, stage)
    filler_fresh_ratio: float = 0.25
    filler_cache_dir: str = ""  # SALID_FILLER_CACHE_DIR; empty = <db dir>/fillers (sqlite) or /tmp
    ack_llm_timeout_s: float = Field(
        2.0, validation_alias=AliasChoices("SALID_ACK_LLM_TIMEOUT", "ack_llm_timeout_s")
    )
    # Voice waits a little longer: the written line also needs a Cloud TTS call (~1.5 s), and the
    # tool it covers takes several seconds anyway.
    voice_ack_timeout_s: float = Field(
        3.0, validation_alias=AliasChoices("SALID_VOICE_ACK_TIMEOUT", "voice_ack_timeout_s")
    )
    # After the cue: no audio within this long -> the cue is sent once more; still nothing after
    # `live_fallback_seconds` more -> the classic path (agent + TTS) answers that utterance.
    live_cue_retry_seconds: float = 6.0
    live_fallback_seconds: float = 4.0
    # Voice-meeting transcription runs every few seconds: use the Singapore endpoint (own quota,
    # close to Discord's sin voice servers). Empty = reuse the bot's client (gemini_location).
    transcribe_location: str = "asia-southeast1"
    # Meaning-based memory search (salid.embed): empty model = keyword-only, no embedding calls.
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 256
    embed_location: str = ""  # empty = gemini_location
    # Organisational memory (salid.memory): raw chat log + per-channel day summaries.
    memory_retention_days: int = 90
    memory_update_every: int = 30  # new messages in a channel that trigger a summary update
    memory_quiet_minutes: int = 20  # quiet time after which any new messages are summarised
    memory_messages_collection: str = "salid_messages"
    memory_summaries_collection: str = "salid_day_summaries"
    memory_channels_collection: str = "salid_memory_channels"  # history-import state per channel
    memory_state_collection: str = "salid_memory_guilds"  # import budgets per guild
    # History import (see salid.memory_import)
    memory_import_guild_cap: int = 20000  # messages imported per guild per run
    memory_import_images_per_day: int = 300  # historical images described per guild per day
    memory_import_summaries_per_day: int = 200  # past-day summaries generated per guild per day
    # Audio/video understanding (salid.media): minutes of unaddressed audio transcribed per guild/day.
    media_minutes_per_day: int = 120
    # Documents (salid.docs): pages of PDF/office/text files read into memory per guild/day.
    doc_pages_per_day: int = 500
    # Promo post kits (salid.promo): image model for the AI scene (location `global`) and AI scenes
    # per guild/day; 0 = AI scenes off (kits use the original photo).
    promo_image_model: str = "gemini-nano-banana-2.1"
    promo_kits_per_day: int = 20
    # Concept studio (salid.concept): concept images per guild/day (own counter, 0 = off) and size.
    concept_images_per_day: int = 20
    concept_image_size: str = "1K"
    # Per-guild settings document `{guild_collection}/{guild_id}` (see salid.guild_settings).
    guild_collection: str = "salid_guilds"
    # PDF documents (salid.pdf_report): regular TTF/OTF with Thai glyphs; empty = search known paths.
    pdf_font: str = ""
    # Scheduled meetings (salid.scheduling)
    meeting_remind_minutes: int = 10  # reminder before the start (and again at the start)
    meeting_noshow_minutes: int = 5  # DM attendees missing from the voice channel after this
    meeting_empty_minutes: int = 2  # an auto-joined meeting nobody ever came to ends after this
    # A meeting where people were present ends only after this long with no humans in the channel.
    meeting_leave_empty_seconds: float = 30.0  # covers a brief disconnect, then Salid leaves
    meeting_leave_empty_minutes: float | None = None  # deprecated: when set, overrides the seconds
    # Auto-join (salid.voice_pool): Salid joins a voice channel as soon as `auto_join_min_humans`
    # people are in it. `auto_join_delay_seconds` > 0 adds a stability wait (default: none).
    auto_join_min_humans: int = 1
    auto_join_delay_seconds: float = 0.0
    auto_join_cooldown_minutes: int = 10  # no auto-join where someone told Salid to leave
    voice_connect_timeout_seconds: float = 20.0  # a voice handshake that hangs is abandoned
    # Email (salid.email_out): Gmail sends only when all three are set (refresh token = secret)
    gmail_client_id: str = ""
    gmail_client_secret: str = ""
    gmail_refresh_token: str = ""
    gmail_sender: str = ""  # the From address (the dedicated Google account)
    # Flights (salid.flights): live status/tracking only with one of these provider keys (secrets)
    aerodatabox_key: str = Field(default="", repr=False)  # RapidAPI key
    aviationstack_key: str = Field(default="", repr=False)
    email_daily_limit: int = 20  # emails per guild per day
    # Briefings (salid.briefings): documents above this total (invoice / quotation) are "important",
    # an unanswered request to a member is reminded after this many hours.
    doc_alert_min_amount: float = 50000.0
    ask_reminder_hours: float = 2.0
    doc_alerts_per_day: int = 10  # real-time document alerts per person per day
    # Where `/ตั้งค่า` links point: the web service that serves /hub.
    # Licensing (salid.licensing): unlicensed guilds get no ingestion and no answers. Existing
    # guilds (installed before the first licensing lookup) get a free trial until `grandfather_until`.
    license_enforce: bool = True
    license_grace_days: int = 3
    grandfather_until: str = "2026-12-31"
    # Web uploads of the command center: GCS (`receipt_bucket` + this prefix) or a local folder.
    docs_prefix: str = "salid-docs/"
    docs_local_dir: str = "./data/uploads"
    hub_url: str = "https://meeting-scribe-638191001920.asia-southeast1.run.app"

    @field_validator("receipt_channel_ids", "ask_channel_ids", "exec_user_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v):
        if isinstance(v, str):
            return {int(x) for x in v.replace(" ", "").split(",") if x.strip().isdigit()}
        return v

    @field_validator("voice_tokens", mode="before")
    @classmethod
    def _parse_tokens(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("exec_role_names", mode="before")
    @classmethod
    def _parse_names(cls, v):
        if isinstance(v, str):
            return {x.strip() for x in v.split(",") if x.strip()}
        return v

    @field_validator("digest_channel_id", mode="before")
    @classmethod
    def _parse_optional_id(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return int(v) if v.isdigit() else None
        return v

    @property
    def leave_empty_seconds(self) -> float:
        """How long a meeting waits with nobody in the room (old minutes setting still honoured)."""
        if self.meeting_leave_empty_minutes is not None:
            return float(self.meeting_leave_empty_minutes) * 60.0
        return float(self.meeting_leave_empty_seconds)

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    def now(self) -> datetime:
        return datetime.now(self.zone)

    def today(self) -> date:
        return self.now().date()

    @classmethod
    def from_env(cls) -> Settings:
        return cls()

    def is_exec(self, user_id: int, role_names: list[str] | set[str] = ()) -> bool:
        return user_id in self.exec_user_ids or bool(self.exec_role_names & set(role_names))
