# Meeting Scribe — project context for people and AI agents

Read this first, then `README.md` (product, architecture, how to use it with Google Meet).

## What this is

A Google AI Builder Cup prototype (theme: Future of Work & Enterprise Productivity).
A host opens the web app next to a Google Meet, clicks Record and shares the Meet tab with
"Share tab audio". The browser mixes tab audio + mic and uploads ~45 s audio segments.
The server transcribes each segment in Thai with speaker labels (Gemini on Vertex AI), and on
Stop an ADK multi-agent summarizer produces decisions, action items and speaker names, saves
them to Firestore and posts a Discord embed with the transcript attached.

- Code, comments, docs and UI are **English**. Meeting audio, transcripts and summaries are **Thai**.
- Submission deadline: **18 Oct 2026**.
- Judging: Gen AI technical merit 40%, problem fit 25%, innovation 25%, UX 10%.
- Repo: https://github.com/CU-DrAgon/agentic-coworker-assistance

## Pipeline in one line

Browser chunk → `POST /api/meetings/{id}/chunks` → ffmpeg → 16 kHz FLAC → Cloud Storage →
Gemini (`gemini-3.5-flash`, transcribe + diarize) → Firestore. On Stop: `/finish` → Pub/Sub →
`/pubsub/finalize` (OIDC-verified) → ADK `[decisions ‖ action items ‖ speaker names]` on
`gemini-3.5-flash` → synthesizer on `gemini-3.1-pro-preview` → Firestore → Discord webhook.

## Code map (`src/meeting_scribe/`)

| File | Role |
| --- | --- |
| `config.py` | `Settings` from env / `.env` (unknown keys ignored) |
| `models.py` | Shared contract: `Utterance`, `ChunkTranscript`, `Meeting`, `Summary`, `ActionItem`. **Do not break it.** |
| `audio.py` | `to_flac` (ffmpeg) |
| `transcribe.py` | `transcribe_chunk`: prompt, label rules, retry. With a participant list every label is a listed name or `Unknown`; otherwise `Speaker A/B/C`. |
| `agents.py` | ADK pipeline `summarize_meeting`, `root_agent`; `unresolved_labels`, `resolve_speaker_names` (never invents names), `build_summary` (applies names to action-item owners, drops recorder instructions) |
| `storage.py` / `store.py` | GCS upload; Firestore `MeetingStore` and `InMemoryMeetingStore` |
| `discord_post.py` | Embed + transcript attachment; never logs the webhook URL |
| `pipeline.py` | `process_chunk`, `finalize` (idempotent, per-process lock) |
| `app.py` | FastAPI routes; health check is `/health` (Cloud Run reserves `/healthz`) |
| `static/index.html` | Recorder UI, live transcript, summary |
| `hub.py` / `static/hub.html` | Salid command center, multi-guild: `/hub` (Thai) + `/api/hub/{me,guilds,link}` and `/api/hub/guilds/{id}/{settings,activity,ask}`. Google ID token as Bearer, verified against `HUB_GOOGLE_CLIENT_ID` (empty = 503); any verified account signs in, guild routes need the email in that guild's `admin_emails` (else 403); `HUB_ADMIN_EMAILS` = optional super admins (all guilds + meeting activity). `POST /api/hub/link` redeems a `/ตั้งค่า` code in a Firestore transaction. No server secret, no Discord secret or token in the web service. |
| `scripts/deploy.sh` | Idempotent Cloud Run deploy + IAM + Secret Manager + Pub/Sub push subscription |
| `scripts/resummarize_dry_run.py` | Re-run the summarizer on a stored meeting and print before/after. Read-only, but calls Vertex AI. |

## Salid (`src/salid/`) — second package

Discord bot "สลิด" (display name Salid), **multi-tenant**: tenant key = Discord guild id; any server adds it via the public invite and gets an isolated ledger and settings with no setup. Receipt photos to an expense ledger, Thai executive Q&A. Env vars are all
`SALID_*` (see `.env.example`). `SALID_STORE=sqlite` (default) or `firestore`. Deployment is prepared but **not run**: `Dockerfile.salid`, `cloudbuild.salid.yaml`, `scripts/deploy_salid.sh`
(dry-run by default, `--apply` to execute; README "Deploying Salid"). Never use `--apply` without human approval. Bot-facing replies and prompts are Thai by default and **bilingual**: every typed command has Thai and English forms, and a per-channel reply language `th`/`en`/`auto` (`/ภาษา` `/language`, typed `speak English`, hub; guild `default_language`) switches everything Salid says. All fixed user-facing strings live in `src/salid/i18n.py` (`t(key, lang, **fmt)`; Thai text is the original wording, add both languages for every new string); LLM prompts take a `language` argument.

| File | Role |
| --- | --- |
| `config.py` | `Settings` (pydantic-settings, `SALID_` prefix, `.env`) |
| `models.py` | `Receipt`, `ReceiptExtraction`, categories |
| `store.py` | `Store` protocol (reads take `guild_id` first; dedupe is per guild), `SQLiteStore` (old DBs rebuilt with `guild_id`, rows = guild 0, UNIQUE(guild_id, sha256)), `make_store` (`SALID_STORE`) |
| `firestore_store.py` | `FirestoreStore`: doc id = `{guild_id}_{sha256}`, transactional int id counter in `<collection>_meta`, every query starts with `guild_id ==` server-side, date range and the rest in Python (no composite index; shop scale only) |
| `archive.py` | Optional GCS archive of original receipts (`SALID_RECEIPT_BUCKET`, objects `salid-receipts/{guild_id}/{sha256}.ext`), best effort |
| `llm.py` | Vertex Gemini client + retry on transient errors |
| `extract.py` | Receipt extraction, dedupe, ingest, corrections |
| `qa.py` | Expense Q&A as a Google ADK `LlmAgent` + `InMemoryRunner`; `build_tools(LedgerTools)` makes per-request FunctionTools bound to the guild (`guild_id`) and the caller (`forced_submitter_id`), neither is in a tool schema, `ask`/`ask_async` return `Answer` (with token counts and per-tool ms); max 4 tool rounds. Addressed messages with attachments pass `files: list[FileInput]` (multimodal parts, limits in `select_files`) + `FileActions` (bot-bound `record_receipt` / `start_deep_task` tools; `Answer.receipts` lets the bot link the reply message). `MemoryTools`: `chat_days`, `chat_messages`, hybrid `chat_search` (keyword + meaning, RRF). System prompt = `stable_prompt` (cacheable) + `volatile_prompt` |
| `deep_task.py` | Executive "งานใหญ่:" / `/deep` tasks: guild- and caller-scoped CSV export + TASK.md in a temp dir, Gemini CLI headless with isolated HOME and a pinned model, collects `report.md`, always deletes the temp dir |
| `digest.py` | Daily digest text (`lang`) |
| `cards.py` | Image cards with correct Thai: Pillow RAQM only (`thai_ready()` needs system libfribidi; Docker `libfribidi0`), bundled Kanit (`fonts/`, SIL OFL) else `pdf_report.find_fonts`, `fit_text` (shrink, wrap at spaces, balance two lines, split Thai only at syllable boundaries via `break_ok`), `draw_lines` (optional soft shadow), scrims, badges / pills, `cover` crop, JPEG ladder under 1.5 MB, WCAG `contrast` |
| `promo.py` / `promo_bot.py` | Promo post kits (flag `promo`): `make_promo_kit` tool turns the message's product photo into `promo-1x1/4x5/9x16.jpg`. One Nano Banana edit (`SALID_PROMO_IMAGE_MODEL`, `global`, 9:16 2K, no sampling params) restyles only the scene, never draws text; Thai text is typeset by `cards`. Daily unit taken before the call (`SALID_PROMO_KITS_PER_DAY`, `take_unit`); blocked / failed / over the cap -> original-photo `fallback_scene`. `PromoMemory` (process only, 30 min, 50 entries) keeps the last photo + request per (guild, channel, user) for revisions. `PromoMixin._promo_tools` offers the tool only for chat answers with a photo or a remembered kit (voice passes no files) |
| `deck.py` | Slide decks, pure (no network): `DeckSpec` / `Slide` / `ChartQuery` outline models with limits, `chart_data` (ledger summary -> top groups + "Other", one series per currency), `render_deck` (python-pptx, 16:9 theme, Thai in the `a:cs` font slot, adds the `p:notesMasterIdLst` python-pptx omits), `deck_filename`. Used by `qa`'s `create_deck` tool (bound to the request's scoped `LedgerTools`; files come back in `Answer.files`, the bot attaches them; feature flag `decks`). `scripts/salid_sample_deck.py` renders a sample offline |
| `i18n.py` | Reply language: `PRIMARY` (th, en: hand-written) + `EXTRA` (zh hi ms id vi ko ja: second tier) = `LANGS` + auto; `resolve_lang` (auto = detect by script: Thai/kana/Hangul/Han/Devanagari/Vietnamese letters, else en), `base_lang` (th or en: use it for prompt blocks, particles, formats), `lang_name`, `reply_language_line`, `resolve_language` (channel override -> guild default), `parse_language_command` (+ server form), the `CATALOG` of fixed strings and `t()` (extras read `locales/<lang>.json`, fall back to English, never Thai) |
| `locales/*.json` / `scripts/salid_translate_catalog.py` | Generated catalogs of the seven second-tier languages (`{"_meta": {"source_hash"}, "strings"}`), committed. Regenerate offline (Gemini on Vertex via ADC, cheap `gate_model`): `uv run python scripts/salid_translate_catalog.py [--langs ja,ko] [--only-missing]`; only new/changed English keys are translated, placeholders are validated, failures stay out (English fallback). Re-run it after adding catalog keys. |
| `guild_settings.py` | Firestore `salid_guilds/{guild_id}` (features, receipt/ask/digest channels, `channel_modes`, `default_mode`, `channel_languages`, `default_language`, digest time, `exec_role_names`, `admin_emails`, bot-synced name/icon/`channels`/`installed_at`/`active`) and one-time `salid_link_codes/{code}`; `GuildStore`, in-memory `MemoryGuildStore` (local dev, tests), `BotGuildSettings` (60 s cache per guild, fail-open) |
| `bot.py` / `__main__.py` | Discord client / `python -m salid`. Join/remove/channel events sync the guild doc; intake by CONTENT (describer / PDF intake; configured or learned receipt channels = fast path; never channel names), or @mention + file; instant acknowledgement edited into the answer (`ack.py`), `@[name]` tagging (`mentions.py`), settings and DM tools (`chat_tools.py`, `dm.py`), exact missing-permission replies (`permissions.py`); per-channel modes (`modes.py`: off/listen/call/join/chat, `/โหมด`, hub; `naming.py` name detection; `gate.py` cheap Gemini yes/no gates, fail quiet; snooze `@Salid เงียบก่อน`); executives = owner, Administrator/Manage Server, exec role names, `SALID_EXEC_USER_IDS`; per-guild digest; global slash commands `/ตั้งค่า` + `/setup` (ephemeral hub link) |
| `meeting.py` | Voice meetings: `/เข้าประชุม` `/ออกประชุม` (+ "@Salid เข้าประชุม"). `MeetingManager` (one meeting per guild), `MeetingSink` (voice-recv sink: removes DAVE layer with `dave_session.decrypt`, decodes Opus per user), `Segmenter` (silence 1.2 s / max 20 s / min 0.6 s), Gemini transcription per segment, `Transcript` (memory only, 3 h), wake word `parse_wake` -> `qa.ask` with transcript history, summary on leave. Audio/transcript never persisted. Feature flag `voice_meetings`; invite permissions 1166416 (adds Manage Channels + Connect). |
| `coo.py` / `coo_tools.py` / `coo_store.py` | Coordinator ("COO") role as a `SecretaryBot` mixin: role-hierarchy adapter (`_ranks`, `_coo_scope` -> `Visibility` for the memory tools), disclosure DMs with persistent `DisclosureButton` (DynamicItem) + `handle_disclosure`, DM notes (`kind="dm"`, scope `dm_scope(user_id)` = `-user_id`, `visibility` normal/private), daily follow-up DMs, evening report + private channel `รายงานจากสลิด` (`plan_report_overwrites`), `/รายงาน` `/report`; agent tools `list_tasks` `update_task` `request_disclosure` `my_day`; `DocStore` (SQLite / Firestore) for `salid_tasks`, `salid_disclosures`, `salid_coo_state` |
| `tasks.py` / `disclosure.py` / `reports.py` | Pure-ish logic: task list + `TaskTracker` (memory scheduler `on_summary` hook: action items -> tasks, batched Flash-Lite done check), disclosure grants (`DisclosureRepo.unlocked_ids`), report input assembly (`collect_company`, `compact_text` cap, plain fallback) and `person_day` |
| `briefings.py` / `briefing_bot.py` / `deliver.py` | Personal morning / evening brief DMs (`briefings` flag, per-user prefs in `salid_notif_prefs`, replaces the follow-up DM), pending asks (`AskBook`, regex only), important-document alerts (`DocInfo`, `alert_targets`, `AlertLog`) as `BriefingMixin`; `deliver_file` tool (`match_channels`, `DeliverMixin`: access rules in code, confirm gate, `salid_deliveries` log) |
| `catchup.py` / `catchup_bot.py` | "What did I miss?": `catch_up(since?)` agent tool (registered in `_agent_kwargs`, so voice has it too), `/ตามทัน` `/catchup` (ephemeral) and a short section in the morning brief after a working day away. Start = explicit (`parse_since`) or the asker's last message before their latest 4 h silence; reads only the asker's readable channels (memory-tools scope), own tasks, invited meetings, alerts already addressed to them. Feature flag `catchup`. |
| `translate.py` / `freetime.py` / `polls.py` / `forms.py` / `approvals.py` / `broadcast.py` / `captions.py` / `voice_actions.py` (+ `*_bot.py` mixins) | Feature modules, each with an agent-tool function and a per-guild flag (`translate`, `freetime`, `polls`, `approvals`...). All tools are appended in ONE list in `SecretaryBot._agent_kwargs` (COO first), so chat and voice share them; `qa.ask_async` picks prompt sections from the tool names present (`list_tasks` -> COO, `find_free_time`, `wiki_search`). `voice_actions.ACTION_TOOL_NAMES`: tools re-bound to the speaker in a voice room (parity test checks the names exist). Freetime reads approved leave via `approved_leave`. |
| `markets.py` / `markets_bot.py` | Stock / gold / news tools (feature flag `markets`): `stock_quote` `stock_history` (Yahoo Finance chart endpoint, Thai names -> `.BK`, 60 s cache), `gold_price` (Gold Traders Association classic page + Yahoo `GC=F`/`THB=X`), `news` (Google News RSS, 10 min cache, near-duplicate titles merged), `market_watch` price/% alerts (DocStore `salid_market_watch`, polled from the bot tick in market hours, once per condition per day, DM via `send_dm`; caps 20 per user / 50 symbols per guild), `market_brief` (opt-in line in the morning brief, `NotifPrefs.markets`), `track_news_topic` (exec only; up to 3 labelled headlines in the evening report). Sources fail soft (`unavailable` + read_web hint); no numbers are ever invented. Voice: `market_watch` `market_brief` `track_news_topic` are in `ACTION_TOOL_NAMES`. |
| `ideas.py` / `frameworks.py` / `ideas_pdf.py` / `ideas_bot.py` / `longtext.py` | Idea / project workspace (flag `ideas`): living project document per idea in DocStore `salid_ideas` (sections problem/customers/BMC/SWOT/competitors/GTM/pricing/next steps), agent tools `idea_save` `idea_get` `idea_list` `idea_export` (pdf/docx/md) `idea_share` and `business_framework(kind, idea, context)` (bmc, lean_canvas, swot, pestel, porter5, value_proposition, pricing_tiers with THB/VAT defaults, gtm, pitch_outline, plus the concept kinds product_concept, brand_theme, website_copy, marketing_plan, ui_concept: complete headings validated, one re-ask, PDF attached with a 9-box BMC page, <= 8 line chat summary). Concept images (`IdeaDoc.images`, JPEG <= 700 KB each as base64 in DocStore `salid_idea_images`, max 12 per idea, oldest dropped) are listed in the Markdown export and embedded one per page in the PDF; section `sourcing` holds dated price checks. Visibility in code: owner; default also people who outrank the owner (like DM notes); `private` owner only; `team`; named people; in a shared room only `team` ideas are shown. Library category `ideas`. `longtext.py`: `bot.chunks` / `_deliver` split on headings and paragraphs, never mid-section; over 3 messages -> short summary + full text as PDF/.md. Tools expose files through a function attribute `outbox` (`qa.tool_files`). |
| `concept.py` / `concept_bot.py` | Concept studio (flag `concept`, needs `ideas`; `price_check` also needs `web`): `concept_image(brief, kind=product|scene|ui|logo|moodboard, views=front|turnaround|set, style, idea, revise, file_index)` = one Nano Banana text-to-image call (`SALID_PROMO_IMAGE_MODEL`, 1K, `SALID_CONCEPT_IMAGE_SIZE`; brief in English, prompt always forbids text/logos/faces; 1:1 product/logo/moodboard, 16:9 ui/scene/sets), `turnaround` = ONE image with 4 views on a 2x2 grid, labels drawn by `cards` (a concept render, not a 3D model). Own daily counter `SALID_CONCEPT_IMAGES_PER_DAY` (20 per guild, 0 = off, `promo.take_unit` before the call); over cap / blocked / timeout / error -> i18n line, no image. `ConceptMemory` (process, 30 min) keeps the last concept per (guild, channel, user) for `revise=true` (previous image = reference); a photo of the message is the reference otherwise. With `idea` the image is stored in the idea document. `price_check(item, country=TH, quantity, purpose=buy|procure, idea)` = one Google-grounded search (`web_read.grounded_search`), the table is rebuilt in code (rows without a number dropped, <= 8, range only for one currency and unit, grounding sources numbered), `procure` adds MOQ / lead time, result appended to the idea's `sourcing` section. `ConceptMixin._concept_tools`: chat gets both tools, voice (`room_message` stand-in) only `price_check`. Catalogues: no tool, `web_search` / `read_web` for a supplier page and `search_everything` for uploaded catalogue PDFs |
| `data_studio.py` / `data_bot.py` / `skills/data_analysis.md` | Data studio (flag `data`, default on): Excel / CSV / JSON analysis. `data_studio`: `load_tables` (xlsx/xlsm each sheet = table `t_<slug>`, xls via xlrd, csv/tsv sniffed + utf-8/utf-8-sig/cp874, json array; header row detected below title rows; Thai numbers `1,234` `฿1,200` `12%`=0.12 `(40)`=-40, day-first dates and Buddhist years 2400-2700 minus 543; columns `slug` or `col_N` with the original `label`; caps 50 MB / 200k cells per table / 20 tables / 200 cols, first rows + a note when cut), `profile` / `profile_md` (<= 40 lines; issues: duplicate header, mixed types, empty column, likely total row), `run_sql` (DuckDB in memory, single SELECT/WITH only, regex + parser check, external access off, 512 MB, 10 s timeout with `interrupt`, 200 rows x 40 cols; hints list the real columns), `result_md` / `result_xlsx`, `chart_png` (matplotlib Figure+Agg, 1200x700, bar/line/pie/scatter/hist, 29 categories + other, Thai/CJK/Devanagari through `pdf_report.find_fonts`, <= 1 MB), `DatasetRepo` on the DocStore (`salid_datasets` record + `salid_dataset_data` gzip-CSV base64 chunks <= 600 kB each so Firestore's 1 MiB holds; <= 3 MB compressed else not kept; 30 per guild, oldest unpinned pruned). Visibility `can_read`: channel datasets only when the asker AND everyone who reads the answer can read the source channel (owner too); DM / private-chat datasets (channel 0) only for the owner in a private conversation. `data_bot`: tools `data_load(file_index, dataset)` `data_sql` `data_chart(query, kind, x, y, title)` `data_export(query, file_format, name)` `data_save` `data_list` `data_forget` (owner or exec), per-request state in the `DataCtx` closure (the dataset loaded last is the default), files out through `outbox`; `data_chart` `data_export` `data_save` `data_forget` are voice action tools. Raw bytes reach the tools through `qa.FileInput.raw` (set by `media_bot._fill_office_input` for xlsx/xls/csv/tsv/json up to 25 MB; the text part gets "(use data_load for analysis: N sheets, M rows)"). `skills/data_analysis.md` is the prompt playbook (`qa.stable_prompt(data=True)` when `data_sql` is a tool). |
| `company_brain.py` / `company_brain_bot.py` / `domain_packs.py` | Company learning (flag `company_learning`): per-guild `CompanyPlaybook` (industry, about, persona notes, rules, glossary, SOPs, FAQ, up to 3 domain packs, pending suggestions, history) in DocStore `salid_company`. Agent tools `teach_company` `propose_lesson` `show_company_playbook` `forget_company_item` `list_pending_suggestions` `review_suggestion` `company_domain` (registered in `_agent_kwargs`; voice re-binds the action ones). Admins/executives apply at once; others create a `Suggestion` DM'd to admins with persistent `PlaybookButton` Approve/Decline. `injection_reason` refuses items touching permissions/secrets/persona particle. `company_block` = prompt data block (<= 1,500 chars, keyword-ranked glossary/FAQ/SOP, pack summary <= 600) passed as `company=` to `qa.ask`; once-a-week onboarding nudge to admins when the playbook is empty. 14 built-in packs in `domain_packs.py`. Personal style stays in `profiles.py`. |
| `profiles.py` | Personal preferences (`Profile` per guild+user in `salid_profiles`, `ProfileTools` bound to the speaker, `profile_block` prompt data block, `effective_language`, `ProfileMixin`) |
| `tone.py` / `tone_bot.py` | Switchable tone (personality) presets, TH + EN, no feature flag: `cheerful` (default, today's wording) `friendly` `professional` `concise` `calm`. `qa.PERSONA_IDENTITY` (who she is: สลิด, ค่ะ/คะ, never ผม/ครับ; never changes) + `tone.tone_block(tone, lang, note)` (emoji, exclamation, length, small talk, greeting) = `qa.persona()`; `stable_prompt` / `system_prompt` / `build_agent` / `ask` take `tone` + `tone_note` (the tone is in the text, so the prefix cache is per tone). State: guild `default_tone`, `channel_tones`, `tone_note` (`guild_settings.py`, hub selects) and `Profile.tone` (only in the person's private channel and DMs). `effective_tone(guild, channel, profile, private)`; `SecretaryBot._tone_for` feeds `_agent_kwargs` (chat and voice). Commands: `/โทน` `/tone` (choices + scope channel/server + note), typed "เปลี่ยนโทนเป็นทางการ" / "change tone to professional" / "โทนอะไร" (`parse_tone_command`, `parse_server_tone_command`; shared room = channel for executives/managers, private channel/DM = the person), agent tool `set_tone(tone, scope, note)` (voice action tool; confirmation text comes back already in the new tone). Notes go through `company_brain.injection_reason`. Live voice persona line (`tone.voice_line`), briefs and company reports take the tone too; "one moment" fillers/acks stay tone-neutral. |
| `wiki.py` / `wiki_bot.py` | Company knowledge base (feature flag `wiki`, needs `memory`): pages in DocStore `salid_wiki` (last 5 versions, source channels + authors per channel, never touched by memory retention), agent tools `wiki_write` `wiki_get` `wiki_search` `wiki_list` `wiki_delete` `wiki_publish`; a page is visible only if the reader can read ALL source channels (or outranks the authors used from the ones they cannot); `WikiMixin` posts to the remembered knowledge channel (forum or text) after confirmation and only where the audience can read every source |
| `search.py` / `web_search.py` / `research.py` / `search_bot.py` | Information search (flag `search`): `search_everything` (chat, files, wiki, receipts for executives, tasks, meetings, places, in parallel through the scoped tool objects of `_agent_kwargs`; merge/rank/dedupe; Thai<->English query expansion with the cheap model + glossary), `web_search` (Google-grounded answer with numbered citations, Bangkok date in the request) and `deep_research` (background plan -> parallel web_search -> cited report + PDF, one per guild at a time, `SALID_RESEARCH_DAILY_CAP` per day, cancelled with `cancel_deep_task`); the last two need `web` |
| `orgs.py` / `licensing.py` / `sharing.py` / `library.py` / `connectors/` + `meeting_scribe/center.py` | Command center backend (`/center`, `/api/center/*`): orgs, members, workspaces (Discord / Slack / ClickUp / web), ShareLinks (public items only, chat memory never crosses), license gate (`Licensing.is_licensed`, 5 min cache, grandfathering of existing guilds to a trial until 2026-12-31, `SALID_LICENSE_ENFORCE`; `BotGuildSettings.license_gate` turns every feature off for an unlicensed guild, on_message / voice / DMs / slash commands are gated in bot.py except `/ตั้งค่า`), documents library + web uploads, web chat via `qa.ask`, Slack events / ClickUp import. Store: `OrgStore` on the DocStore. API contract: see the feature brief. |
| `votes.py` / `votes_bot.py` | Meeting votes (flags `votes`, `vote_offers`): `call_vote` `close_vote` `vote_status` agent tools (in `_agent_kwargs`; `call_vote` / `close_vote` are voice action tools). Each voter gets one button ballot (`VoteButton` DynamicItem) in their private channel via `send_dm`; one tally message in the meeting chat edited per ballot (counts only); closes when all voted / deadline (`_votes_due` tick) / on request; result line posted, added to the minutes (`MeetingManager.record_vote`) and announced in voice via `LiveVoice.announce_cue`. Majority of ballots cast, tie = no conclusion. Offers from untagged talk: `looks_like_vote_call` + cheap gate, one text offer per channel per 10 min, `VoteOfferButton`. DocStore kinds `salid_votes`, `salid_vote_offers`. |
| `reminders.py` / `reminders_bot.py` | Personal reminders, once and recurring (flag `reminders`): `set_reminder(message, at, every, weekdays, until, for_user, where, quiet_hours)` `list_reminders` `cancel_reminder(which)` agent tools (in `_agent_kwargs`; all three are voice action tools bound to the speaker). The model turns natural language into `YYYY-MM-DD HH:MM` / `HH:MM` / `+2h`; code does limits (min 15 min interval, 20 active per person, no past, <= 1 year), quiet hours 22:00-08:00 for `every`, exec-only `for_user`, `where` = `me` (`deliver_to_person`) or `here`. `ReminderBook` on the DocStore (kind `salid_reminders`, ids `R-0001`). `_reminders_due` (bot tick, every 30 s): ONCE late > 30 min still sent with a note, recurring skip missed slots (no burst), member gone -> deactivated. |
| `projects.py` / `projects_bot.py` / `meeting_scribe/pm_api.py` | PM dashboard (`/pm`, `/api/center/orgs/{org}/pm/*`, flag `projects`): projects in DocStore `salid_projects` (guild = workspace memory id; Discord workspace preferred, else web), one owner each, templates `standard` / `hackathon` (4 default deliverables), tasks ARE Salid's `Task`s (`project_id` `assignee_email` `milestone_id`), status / owner / deadline / priority changes need a reason (`ReasonRequired`, logged in `changes`), `project_stats` + `needs_review` (high priority and late), finance (`budget`) only for admins and `Member.finance`, `Member.department`. Pure `can_read` `can_contribute` `can_edit` decide access in code. Agent tools `project_create` `project_update` `project_list` `project_show` `project_add_task` `my_projects_and_tasks` (in `_agent_kwargs`; all but list/show are voice action tools); `TaskTracker.ingest` links a task to a project named in its text ("โปรเจกต์ X" / "project X"). `/pm` serves `static/pm.html` (503 until built); risk proposals (`risks/assess`) use the cheap model and fail soft. |
| `memory.py` | Team memory layers 1-2: `MemoryService` (record / edit / delete), `MemoryScheduler` (batched day-summary updates, daily purge by each guild's retention, vector backfill), `vectorize` / `vectors_usable` (embeddings of summary items, unchanged items reuse their vector) |
| `memory_store.py` / `firestore_memory.py` | `MemoryStore` protocol (every read takes `guild_id` + allowed channels), SQLite and Firestore stores for the raw log, `DaySummary` (with optional `vec_model` / `vec_hashes` / `vecs`) and import state; Firestore uses equality filters only (no composite index), per-day reads in parallel |
| `memory_summary.py` | Cheap-model day summary (`update_summary`), `summary_items` (the searchable lines), `render_summary` |
| `memory_access.py` | Who may read which channel's memory: asker can read it, and so can everyone reading the answer's channel |
| `memory_import.py` / `image_memory.py` | History import (resumable, capped) and image descriptions into the log |
| `embed.py` | `VertexEmbedder` (`SALID_EMBED_*`, retry), `NullEmbedder` (kill switch), packed float32 vectors, cosine via `math.sumprod` |
| `memory_index.py` | `IndexedMemoryStore`: per-guild cache of summaries + vectors (64 MB LRU, one load query); every memory write goes through it, so it assumes **one bot process** |
| `recent.py` | `RecentMessages`: last messages per channel from gateway events (short-term context), primed by one Discord read, reset on reconnect; also assumes one bot process |
| `cli.py` | `python -m salid.cli extract\|ask\|deep\|digest` for offline testing |
| `scripts/salid_seed_demo.py` | Seed a demo SQLite ledger |

Salid tests live in `tests/salid/` (a scripted ADK `BaseLlm` and a fake Firestore; the `store` fixture runs
every store test on SQLite and Firestore); `tests/salid/test_salid_live.py` is marked `live` and hits real
Vertex, real Firestore (throwaway `salid_receipts_test_<random>` collection, deleted afterwards) and GCS
(`salid-receipts/test-<random>/`).

Salid env vars beyond the basics: `SALID_EMBED_MODEL` (`gemini-embedding-001`; empty = keyword-only memory
search), `SALID_EMBED_DIM` (256), `SALID_EMBED_LOCATION` (empty = `SALID_GEMINI_LOCATION`), `SALID_MEMORY_*`,
`SALID_STORE`, `SALID_FIRESTORE_COLLECTION`, `SALID_RECEIPT_BUCKET`
(empty = no archive), `SALID_GEMINI_CLI_PATH` (empty = deep tasks disabled), `SALID_DEEP_TASK_TIMEOUT`
(600), `SALID_DEEP_TASK_MODEL` (`gemini-3-flash-preview`), `SALID_GUILD_COLLECTION` (`salid_guilds`),
`SALID_PROMO_IMAGE_MODEL` (`gemini-nano-banana-2.1`), `SALID_PROMO_KITS_PER_DAY` (20 per guild; 0 = AI scenes off), `SALID_CONCEPT_IMAGES_PER_DAY` (20 per guild; 0 = concept images off), `SALID_CONCEPT_IMAGE_SIZE` (`1K`),
`SALID_HUB_URL` (Cloud Run URL; target of `/ตั้งค่า` links). Hub (web service) env: `HUB_GOOGLE_CLIENT_ID`
(required), `HUB_ADMIN_EMAILS` (optional super admins), `SALID_DISCORD_CLIENT_ID`, `SALID_GUILD_COLLECTION`;
setup in README "Salid command center". Meetings are not per guild yet (hub shows them to super admins only).

**Gemini CLI is an external Node binary, not installed by uv** (tested 0.64.0). Never run it in a directory
with untrusted `.gemini/` or `.env`; `deep_task.py` uses an empty temp workspace and an isolated
`HOME`/`GEMINI_CLI_HOME`. Pinning findings: `-m gemini-3.5-flash` is silently rewritten to
`gemini-3.8-flash` on Vertex (hard-coded, not configurable), so we pin `gemini-3-flash-preview` and also
set `experimental.dynamicModelConfiguration` + `modelConfigs.modelIdResolutions` for every alias,
`disableLoopDetection`, `topicUpdateNarration=false` and `tools.core` (file tools only). `DeepResult.pinned_ok`
reports whether only the pinned model ran.

Tests in `tests/` are fully offline (every external service is mocked). Fixture:
`tests/fixtures/thai_meeting.json`.

### Languages (Salid)

Two tiers. **Primary** `th`, `en`: every string hand-written in `i18n.CATALOG`. **Second tier** `zh hi ms id vi ko ja`:
fixed strings come from generated `src/salid/locales/<lang>.json`; the model is told `Reply ONLY in <language>` at the
top of the persona (`i18n.reply_language_line`, `qa.persona`) and otherwise gets the English prompt blocks. Code that
branches on the language must use `base_lang(lang)` / `lang == "th"` (never `lang == "en"`, which would send `ja` to Thai)
and `lang_name(lang)` for prompts that name the language. Thai particles only apply to `th`. Voice: `speech._LANG_CODES`
(+ `_VOICE_FALLBACK`: ms -> id-ID, no ms-MY Chirp 3 HD voices), `live_voice.build_instruction_en(reply_lang=)`. Documents:
`pdf_report.find_fonts(lang=)` / `script_fonts` (Noto CJK faces extracted from the .ttc into the user cache, Devanagari),
`font_note` when the fonts are missing (labels fall back to English), `cards.card_fonts(lang)`, `deck.DECK_FONTS`. The hub UI
stays Thai/English but its language selects list all ten. Adding a catalog key: add th + en, then re-run the translate script.

## Infrastructure that already exists — never recreate

- GCP project `meeting-scribe-hackathon`, region `asia-southeast1`, Vertex Gemini location `global`
- Bucket `gs://meeting-scribe-hackathon-audio`, Firestore `(default)` native DB
- Pub/Sub topic `meeting-finalize` + push subscription `meeting-finalize-push`
- Cloud Run service `meeting-scribe` (**max-instances=1 on purpose**: the finalize lock is per-process)
- Service accounts `meeting-scribe-run` (runtime) and `meeting-scribe-pusher` (Pub/Sub push)
- Secret Manager secret `discord-webhook-url`
- Deployed URL: `gcloud run services describe meeting-scribe --region asia-southeast1 --format='value(status.url)'`
  (keep it private: there is no auth on the app)

## Credentials and safety rules

- `.env` holds `DISCORD_WEBHOOK_URL`, a GitHub token (`GH_TOKEN`) and GCP settings. **Never** print,
  log, echo, paste into chat, commit, or put any of these values on a command line. Refer to them by
  key name only.
- Before any commit: `git status` and `git diff --cached`. `.env`, `samples/` and `*.json` key files
  must never be staged.
- Do not create service-account key files. Use `gcloud auth login` + `gcloud auth application-default login`.
- `gcloud run deploy --source .` uploads the directory. There is no `.gcloudignore`, so `.gitignore`
  applies and excludes `.env`. Confirm before deploying.
- `scripts/deploy.sh` pushes the webhook from `.env` into Secret Manager as a **new version**. Run it
  only intentionally.
- **Never set `USE_PUBSUB=true` locally**: it routes finalize to the production push endpoint.
  Local runs use `STORE=memory` and the default `USE_PUBSUB=false`.
- Re-transcribing chunks costs Vertex quota. 3–4 chunks is fine; ask before a whole meeting.
- Ask a human before anything destructive, costly or production-affecting: deploys, new secret
  versions, deleting data, posting to Discord, changing Firestore records.

## Local development

```bash
uv sync
uv run pytest -q                                  # offline, all mocked (live tests are skipped; `-m live` runs them)
uvx ruff check . && uvx ruff format --check .     # lint + format (line length 100)
STORE=memory uv run uvicorn meeting_scribe.app:app --port 8080
curl -s localhost:8080/health
```

Requires Python 3.12 (uv installs it), ffmpeg, and ADC for anything that touches GCP.
Salid PDF tests (`tests/salid/test_secretary.py`) need a Thai TrueType font: `pdf_report.find_fonts`
only knows the Linux paths that `Dockerfile.salid` installs (`fonts-thai-tlwg`). On macOS run them with
`SALID_PDF_FONT=/System/Library/Fonts/Supplemental/Ayuthaya.ttf` (built in); without it 6 PDF tests fail
with "no TrueType font found for PDFs". Promo-kit text needs the system `libfribidi`
(`brew install fribidi`, and `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` when uv runs pytest).

## Known limitations — do not "fix" without asking

- No auth on the web app or API; keep the URL private.
- About 0.1 s of audio is lost at each 45 s segment boundary.
- No retention policy on audio or Firestore (deleting data needs approval).
- Old Cloud Run logs may contain the webhook URL; rotate it after the hackathon.
- Desktop Chrome/Edge only (tab-audio capture).

## Reference data

Firestore meeting `e6379be86dfd` ("Meet1", 25 min, 34 chunks, 309 utterances, participants
Peat / Toey / Jems) is the first real recording. It showed the two bugs fixed on 13 Oct 2026:
speaker labels drifting between real names and `Speaker B/C/D`, and a spoken instruction to the
recorder ("หยุดบันทึกและสรุปข้อมูลการประชุม") extracted as an action item. Use
`uv run python scripts/resummarize_dry_run.py e6379be86dfd` to check regressions without writing
anything.

## How to work here

Small surgical changes. Write a failing offline test first, keep the suite green, match the existing
style. State what you verified and what you could not. Work on a feature branch and open a PR;
redeploy with `scripts/deploy.sh` only after a human approves, then smoke `/health`.

- `dm_guild.py`: which company a DM is about (sticky pick, confirm-to-switch, reply tags, /บริษัท). `private_chat.py` + `private_chat_bot.py`: a private Salid channel per member; `deliver_to_person` / `dm.send_dm` route proactive messages there (DM fallback, profile `delivery=dm` opt-out).
