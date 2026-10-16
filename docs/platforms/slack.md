# Salid on Slack

Salid answers in Slack with the same brain as in Discord: general chat, links and web search,
questions about the receipt ledger, files you send, and team memory. It runs in **Socket Mode**, so
there is no public endpoint to expose and no request URL to configure.

Run it: `uv run python -m salid.platforms` (it starts every platform that has its secrets set).

## What it does

- **Answers** when you DM it, @mention it, say its name ("สลิด", "Salid"), reply in a thread it
  started or answered in, or follow up within ~3 minutes of its answer to you (a cheap model checks
  that the follow-up is meant for Salid). Everything else in the channel is only remembered.
- **Replies in a thread** under your message (no broadcast), split into chunks of at most 3000
  characters, with Markdown mapped to Slack's mrkdwn (tables become code blocks). If an answer is
  slow it posts a short "one moment" line first and edits it into the answer.
- **Files**: send images and PDFs (up to 4 files, 15 MB each, 20 MB total) with an @mention or in a
  DM. Receipts are recorded in the ledger of your workspace.
- **Ledger questions** ("how much did we spend on ingredients this month?"). Executives see
  everyone's receipts, everyone else only their own. Executives are the people in
  `SALID_SLACK_EXEC_USER_IDS` plus the workspace owners and admins.
- **Team memory**: every message in channels Salid is in (and its own replies) is remembered, with
  day summaries and meaning search. Salid reads a channel's memory for you only if you are in that
  channel **and** everyone who can read the channel the answer is posted in can read it too.
  If Slack cannot enumerate every reader of the answering room (a public channel, an API error, or
  over 5000 members), Salid uses that room alone. Optionally it imports the history of channels it
  joins, within the retention window.
- **Per-workspace settings**: the same settings document as a Discord server (feature switches,
  per-channel modes `off / listen / call / join / chat`, reply language, memory retention), keyed
  by the workspace. Until someone saves settings, Salid answers in the language it is spoken to.

Not in phase 1 (Discord-only for now): scheduling and meetings, PDF reports and email, deliver-file,
executive-assistant tools (tasks, briefings, reports), broadcasts, polls, voice, deep tasks, audio
and video files, image descriptions in memory, receipt forwarding, profiles.

## 1. Create the app

1. Open <https://api.slack.com/apps> → **Create New App** → **From a manifest**, pick the workspace.
2. Paste this manifest (change the names if you like) and create the app:

```yaml
display_information:
  name: Salid
  description: "Team assistant: questions, receipts and team memory"
  background_color: "#2d3142"
features:
  app_home:
    home_tab_enabled: false
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false  # without these two, a DM shows "Sending messages to this app has been turned off"
  bot_user:
    display_name: Salid
    always_online: true
oauth_config:
  scopes:
    bot:
      - app_mentions:read
      - chat:write
      - channels:history
      - groups:history
      - im:history
      - mpim:history
      - channels:read
      - groups:read
      - im:read
      - mpim:read
      - files:read
      - users:read
settings:
  event_subscriptions:
    bot_events:
      - app_mention
      - message.channels
      - message.groups
      - message.im
      - message.mpim
      - member_joined_channel
  interactivity:
    is_enabled: false
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
```

3. **Install App** → **Install to Workspace** → copy the **Bot User OAuth Token** (`xoxb-…`).
4. **Basic Information** → **App-Level Tokens** → **Generate Token** with the scope
   `connections:write` → copy it (`xapp-…`).
5. Optional: **App Home** → turn on **Messages Tab** and *Allow users to send Slash commands and
   messages from the messages tab*, so people can DM Salid.

Scope notes: `users:read` gives names and the admin/owner flag; `*:read` scopes let Salid list the
conversations it is in and who is in them (for memory scope). `reactions:write` is not needed.

## 2. Configure and run

Put the secrets in `.env` (never in a command line or in git):

```bash
SALID_SLACK_BOT_TOKEN=xoxb-...
SALID_SLACK_APP_TOKEN=xapp-...
SALID_SLACK_EXEC_USER_IDS=U012ABCDEF,U034GHIJKL   # optional: extra executives (Slack member ids)
# SALID_SLACK_IMPORT=false                         # do not import channel history
```

Check the tokens without starting anything (read-only `auth.test`, prints no secrets):

```bash
uv run python scripts/platforms_smoke.py
```

Start it (needs the same Google Cloud credentials as the Discord bot, for Vertex AI):

```bash
uv run python -m salid.platforms
```

Invite Salid to a channel with `/invite @Salid` (private channels need the invite too). A Slack
member id is under **profile → ⋮ → Copy member ID**.

## 3. Behaviour you can change

Settings live in the document `salid_guilds/{tenant id}` (Firestore, or in memory with
`SALID_STORE=sqlite`). The tenant id of a workspace is stable and printed in the startup log as
`slack ready team=T…`; see "Tenant ids" in [README](../../README.md#salid-on-slack-and-line). Tune
memory retention, language or a channel's mode there. Executives can also keep Salid out of a
channel by setting its mode to `off` (it then records and answers nothing).

## Limits worth knowing

- One bot token is one workspace. For several workspaces run another process per workspace (each
  with its own tokens), or build OAuth distribution later.
- Salid reads history only in conversations it is a member of. It never reads people's DMs with
  each other.
- Slack rate limits apply; Salid retries once or twice when Slack asks it to wait.
