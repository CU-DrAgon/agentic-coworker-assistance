# Voice behaviour (one design)

Source of truth for how Salid behaves in a voice room. Code, labels, prompts and tests follow this
file; change it first. Live mode = Gemini Live (hears the room). Classic mode = wake word, transcription,
text answer, TTS (fallback when Live cannot connect; its keyword rules stay and are marked CLASSIC).

## The rules

| # | Rule | Decided by | Where |
|---|------|-----------|-------|
| 1 | Two human switches only: `voice_talk` may_talk / listen_only (chair or executive, remembered per room, never an agent tool); `hush` ("Salid, be quiet", anyone, silent until her name comes with words, no timer) | people | `MeetingManager.set_voice_talk`, `_hush` / `_clear_hush` |
| 2a | WAKE: only when someone says her name (any language). Code keeps the mouth gate (model audio while listening is held up to `live_call_grace_seconds` for a name in the Live transcript or an ASR `name_called`, else dropped) and a backup cue when ASR hears the bare name twice in 15 s | Live model + code gate | `LiveVoice._gate_audio`, `_on_heard`, `MeetingManager._bare_name_backup` |
| 2b | STAY: the exchange belongs to the caller. What the caller says is for her unless clearly to someone else; she answers without the name; when the caller talks to others she waits quietly and answers when they speak to her again. Others count only when they address her. No keyword rules in live mode (`is_dismissal`, `_ASK_RE`, `_SELF_TALK_RE`, 3-strike, "next word after a bare name") | Live model | instruction + `STATE_ENGAGED*` |
| 2c | END: (a) the caller closes it in words (thanks / that's all / bye / talk later): the model calls `end_conversation`, code accepts reasons thanks / closed / hushed only and only when the caller has been quiet >= 2 s; (b) hush; (c) `engaged_timeout_seconds` with nobody talking to her and her not talking. The window is renewed by her own audio and by any utterance of the caller (`MeetingManager.renew_engaged`); someone else renews it only by saying her name | model + one timer | `end_conversation`, `_consent_tick`, `renew_engaged` |
| 2d | A bare name ("Salid?") is answered "Yes?" and she waits `engaged_bare_timeout_seconds` | code | `_set_engaged(bare=True)` |
| 3 | Turn-taking: she may start after `live_silence_ms` of silence; never starts while a person spoke < 0.5 s ago (hold until 0.5 s quiet, drop the answer after 6 s); in the first 2.5 s of her answer the caller's voice stops her after 0.3 s; later the caller stops her after 0.8 s of clear speech. Side talk, notifications, whispers never stop her. Explicit "stop" words / her name while she speaks also stop her | code | `build_realtime_input`, `_on_audio`, `_yield_until_quiet`, `_check_talkover`, `_on_heard` |
| 4 | Nothing about settings is spoken (mode, language, connection lines are text only). Every spoken answer is also posted ("💬 **Salid:** ..."). Status lines show the reason: called by name / said thanks / closed it / told to be quiet / quiet N min (N built from `engaged_timeout_seconds`) | code | `_post_state`, `_echo_answer`, i18n `voice_why_*` |
| 5 | Language: answers in the company language (+ the personal languages of the people in the room). The Live instruction, the spoken hint to the text agent and a translate-if-wrong-script safety net all use the session's resolved language (`MeetingSession.lang`, never "auto"; the server default is English until chosen, there are no per-channel overrides) | code | `build_instruction`, `_ask_salid`, `_qa_setting` |
| 6 | listen_only: ASR notes + captions + text answers to "name + request"; no engaged state, no state lines, no Live session | code | `_listen_only` |
| 7 | Leaving: never while people are in the room. "the meeting is over" / leave command leaves. Empty room: scheduled end, or `meeting_empty_minutes`. No "nothing to summarize" line | code | `_ticker`, `_leave_politely` |
| 8 | A spoken question posts "⏳ Looking into it: q · 🔎 ..." at once, edited per tool, then "✅ Done". Results up to 700 chars (`ANSWER_MAX_CHARS` = `SHORT_ANSWER_CHARS`) are spoken in full | code | `LiveVoice._one_brain`, `_status`, `_announce` |

## Timers and thresholds

| What | Config name (default) or constant |
|------|-----------------------------------|
| Quiet window after the last talk with her (renewed by her answer or the caller) | `engaged_timeout_seconds` (300) |
| Wait after a bare name | `engaged_bare_timeout_seconds` (300) |
| Hush | no timer |
| `end_conversation` needs the caller quiet for | 2 s (`MeetingManager.end_conversation`) |
| She may start after silence | `live_silence_ms` (1000) |
| Yield: a person spoke less than / hold until quiet / drop after | `YIELD_ACTIVE_S` 0.5 / `YIELD_QUIET_S` 0.5 / `YIELD_MAX_S` 6 |
| Early cut (first seconds of her answer) / speech needed | `EARLY_CUT_S` 2.5 / `EARLY_CUT_SPEECH_S` 0.3 |
| Barge-in by the caller / loud frame level / allowed gap | `TALKOVER_S` 0.8 / `TALKOVER_RMS` 500 / `TALKOVER_GAP_S` 0.35 (`live_barge_in` intentional) |
| Mouth gate: hold for a name | `live_call_grace_seconds` (4) |
| Gate stays open after an exchange ended without `forget_grace` | `live_reengage_grace_seconds` (60) |
| Name in Live transcript but no answer: force a turn after | `live_name_reply_seconds` (1.0); name from ASR: `live_force_reply_seconds` (2.0) |
| Cue retry / then classic fallback | `live_cue_retry_seconds` (4) / `live_fallback_seconds` (4) |
| ASR bare name twice within | `NAME_TWICE_S` 15 |
| Name from the same call seen by both detectors is one call | `NAME_DEDUPE_S` 10 |
| Name hit needs real speech | `name_min_voiced_seconds` 0.35, `name_bare_min_voiced_seconds` 0.45, `name_min_cv`, `name_min_rms` |
| ask_salid answers in the conversation within / hard limit | `live_foreground_seconds` (25) / `live_ask_timeout_seconds` (150) |
| Background result: pause needed / max wait / jobs at once | `live_result_pause_seconds` 1.5 / `live_result_wait_seconds` 90 / `live_max_jobs` 3 |
| One state line flips (edits the previous line) within | `STATE_FLIP_S` 15 |
| Empty room | `meeting_empty_minutes` (10) (`Settings.leave_empty_seconds`) |
| Meeting hard cap when empty | `MEETING_MAX_S` 6 h |

Defaults in `getattr(settings, name, default)` inside the code equal the `Settings` defaults (one
source of truth: `config.py`).

## Notes

- `wake_policy` strict (default) | natural (hub setting): natural also lets a direct reply within 8 s of
  her own words be heard without the name (`NATURAL_WINDOW_S`).
- Classic mode (not Live) keeps its keyword rules: `is_dismissal`, `_ASK_RE`, `_SELF_TALK_RE`, three
  unaddressed utterances, `pending_wake` (the words after a bare name). They never run in live mode.
