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
  `SALID_SLACK_EXEC_USER_IDS` plus the owners and admins **of this workspace**. A Slack Connect
  guest who is an admin of their own workspace is not an executive here (Salid checks that the
  user's `team_id` is this workspace; bots, deleted users and unknown teams never count).
- **Team memory**: every message in channels Salid is in (and its own replies) is remembered, with
  day summaries and meaning search. Salid reads a channel's memory for you only if you are in that
  channel **and** everyone who can read the channel the answer is posted in can read it too.
  If Slack cannot enumerate every reader of the answering room (a public channel, a Slack Connect or
  org-shared channel, an API error, or over 5000 members), Salid uses that room alone. Optionally it imports the history of channels it
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
      - users:read.email
settings:
  event_subscriptions:
    bot_events:
      - app_mention
      - message.channels
      - message.groups
      - message.im
      - message.mpim
      - member_joined_channel
      - member_left_channel
      - channel_rename
      - group_rename
      - channel_archive
      - channel_unarchive
      - channel_deleted
      - channel_left
      - group_archive
      - group_unarchive
      - group_deleted
      - group_left
      - app_uninstalled
      - tokens_revoked
  interactivity:
    is_enabled: false
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
```

3. **Install App** → **Install to Workspace** → copy the **Bot User OAuth Token** (`xoxb-…`).
4. **Basic Information** → **App-Level Tokens** → **Generate Token and Scopes**, add the scope
   `connections:write`, generate and copy it (`xapp-…`). The manifest cannot create this token; it
   is always a manual step. Socket Mode needs it, the bot token alone is not enough.
5. **Direct messages**: the manifest turns on **App Home → Messages Tab** and *Allow users to send
   Slash commands and messages from the messages tab*. If either is off, a DM shows "Sending
   messages to this app has been turned off". Check both under **App Home** if people cannot DM
   Salid.

**After any change to scopes or events, reinstall the app** (**Install App → Reinstall to
Workspace**). Until you do, Slack keeps the old token and the old subscriptions. Do **not** turn on
**token rotation**: Salid keeps one long-lived bot token, and a rotating token would expire after
12 hours and stop it.

Event notes: `app_uninstalled` and `tokens_revoked` tell Salid the workspace removed it (see
Privacy). The channel events (`channel_rename`, archive, delete, `member_left_channel`, ...) only
refresh cached names and member lists sooner; without them the caches expire after 10 minutes.

Scope notes: `users:read` gives names and the admin/owner flag; `users:read.email` lets Salid read the profile email to verify members of an organization (optional, see Linking accounts); `*:read` scopes let Salid list the
conversations it is in and who is in them (for memory scope). `reactions:write` is not needed.

## Linking accounts to your organization

When the Slack workspace is attached to an organization (the command center's **Connect Slack**
button, or an attach code, below), Salid verifies people **automatically**: the Slack profile email
(`users.info` → `profile.email`, scope `users:read.email`) of a real member of this workspace
that equals the email of an organization member links that Slack account to the member. Not
accepted: bots, deleted users, guests (single / multi-channel), people from another workspace
(Slack Connect) and anyone whose address is hidden. A member's existing Slack link is never replaced
this way, and an account another member already holds is never taken. Slack is trusted to have
confirmed the address; an admin who can change members' emails can therefore act as them. Salid
asks Slack at most every 10 minutes per person, and only for workspaces that belong to an organization.

Fallback without the scope: the same one-time code as the other platforms
(`POST /api/center/orgs/{org}/identities/slack/code`, then `link 1234-5678` in a DM; never in a
channel). Uninstalling the app (`app_uninstalled` / `tokens_revoked`) forgets the organization's
Slack links.

**After adding the `users:read.email` scope to an existing app, reinstall it** (Install App →
Reinstall to Workspace); until then `profile.email` is absent and Salid falls back to codes.


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
  with its own tokens).
- Salid reads history only in conversations it is a member of. It never reads people's DMs with
  each other.
- Replies are posted at most about once a second per channel. When Slack answers 429, Salid waits
  the `Retry-After` it is given (up to a minute), tries up to five times, and still posts the
  remaining chunks if one fails.
- Before answering, Salid reads the recent thread or channel from Slack for up to 3 seconds. If
  that is slow or rate limited it uses the messages it already saw live, and it does not read again
  until the `Retry-After` has passed. An answer is never delayed by it.

## Slack's 2025 rate-limit change

Since 2025-05-29, `conversations.history` and `conversations.replies` are limited to **1 request per
minute and 15 messages per request** for **commercially distributed apps that are not in the Slack
Marketplace** and were created on or after that date. **Internal customer-built apps** (installed
in your own workspace, not distributed) keep the normal Tier 3 limit (about 50 requests per
minute, up to 1000 messages per request).

That is why Salid is an **internal, per-workspace app**: you create it in your own workspace and
install it there, as in this guide. Do not enable public distribution for it. Distributing the same
app from a non-Marketplace listing would put its history import on the slow tier.

If the import detects the restricted tier (pages stop at exactly 15 messages, or Slack keeps asking
for waits of 30 s or more) it logs one warning, imports slowly in the background and keeps going.
Answers are not affected, because they only use the live events plus a short live read. The import
resumes where it stopped after a restart or after a run that was rate limited. Importing one
workspace's retention window can take hours on the restricted tier.

## Phase 2: a distributed app

Not built yet. Serving many workspaces with one install link needs: OAuth install with a token
store per workspace, the **HTTP Events API** (a public HTTPS request URL with signature
verification) instead of Socket Mode, and a **Slack Marketplace** listing, which also lifts the
history rate limit for a distributed app. Marketplace rules require deleting a workspace's data
within **14 business days** of uninstall. Salid already does the erasing part: on `app_uninstalled`
(or `tokens_revoked` for the bot token) it stops answering that workspace at once and deletes its
remembered messages, summaries and embeddings. Answers still in flight at that moment are
not posted and not remembered. Receipts in the ledger are **not** deleted yet: a phase 2 tenant
purge must also remove them before Marketplace rules can be met.

## Privacy

- Salid remembers messages of the channels it is a member of (and its own replies), with the
  author's display name and Slack user id, for the retention period of the workspace (90 days by
  default). Set a channel's mode to `off` to keep it out. Private channels and group DMs are
  remembered only if someone invited Salid there.
- 1:1 DMs between people are never visible to Salid. A DM to Salid itself is handled like a message
  to a bot: answered, and remembered for the retention period.
- Memory is read back only with the access rule described above (the asker and every reader of the
  answering room must be able to read the source channel).
- Logs carry ids, counts and timings only. Message text, file contents and tokens are never logged.
  Tokens stay in `.env` or the secret manager and are handed only to Slack's SDK; files are
  downloaded only from Slack's file hosts.
- On `app_uninstalled` or a revoked bot token the workspace's memory is erased and Salid stops
  answering. Receipts already in the ledger are not deleted on uninstall yet (phase 2 tenant
  purge): delete them from the ledger store if you need them gone too.
