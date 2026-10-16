# Salid on LINE

Salid answers in LINE 1:1 chats and groups with the same brain as in Discord: general chat, links
and web search, questions about the receipt ledger, images and PDFs you send, and team memory of
the room. LINE delivers messages to a **webhook**, so Salid needs a public HTTPS URL.

Run it: `uv run python -m salid.platforms` (it starts every platform that has its secrets set).

## What it does

- **Answers** every message in a 1:1 chat. In a group or multi-person chat only when you @mention
  Salid, say its name ("สลิด", "Salid"), or follow up within ~3 minutes of its answer to you (a cheap
  model checks the follow-up is meant for Salid). Everything else in a group is only remembered.
- **Replies** with the reply token first (it expires quickly) and falls back to push messages when
  the answer is slow or too long: at most 5 messages per call, 5000 characters per message. In 1:1
  chats it shows LINE's typing animation while it works. Markdown is turned into plain text (LINE
  renders no markup).
- **Files**: images and PDFs sent to Salid (1:1, or with an @mention in a group), up to 15 MB each.
  Receipts are recorded in the ledger of your LINE channel.
- **Ledger questions**. Executives (`SALID_LINE_EXEC_USER_IDS`) see everyone's receipts, everyone
  else only their own.
- **Team memory** of each room. LINE does not tell a bot who is in a group (that needs a verified
  account), so memory is **never shared between rooms**: a group's memory is used only inside that
  group, a 1:1 chat uses only its own history. This is on purpose (fail closed). A person who is in
  a group still gets answers about that group only when asking in the group.

Not in phase 1 (Discord-only for now): scheduling and meetings, PDF reports and email, deliver-file,
executive-assistant tools, broadcasts, polls, voice, deep tasks, audio and video, image descriptions
in memory, receipt forwarding, profiles. There is also no chat history import: LINE's API cannot read
past messages, so memory starts when Salid joins.

## 1. Create the channel

1. <https://developers.line.biz/console/> → create a **Provider** → **Create a Messaging API channel**.
2. **Messaging API** tab: issue a **Channel access token (long-lived)** and note the **channel secret**
   (**Basic settings** tab).
3. Under **LINE Official Account features** (link on the same tab), set:
   - *Auto-reply messages*: **Disabled**
   - *Greeting messages*: **Disabled** (optional)
   - *Allow bot to join group and multi-person chats*: **Enabled**
4. Scan the QR code on the **Messaging API** tab to add Salid as a friend, and invite it to a group.

## 2. Configure

```bash
SALID_LINE_CHANNEL_SECRET=...
SALID_LINE_CHANNEL_ACCESS_TOKEN=...
SALID_LINE_EXEC_USER_IDS=Uxxxxxxxx,Uyyyyyyyy   # optional: LINE user ids of executives
# SALID_LINE_HOST=127.0.0.1                    # 0.0.0.0 in a container
# SALID_LINE_PORT=8091
```

A LINE user id (`U…`, 33 characters) is a per-channel id; it is not visible in the app. To find
yours, run Salid with `SALID_LINE_EXEC_USER_IDS` empty and read the LINE console's webhook log, or
send a message and read `source.userId` in a test webhook payload.

Check the tokens without starting anything (read-only "get bot info", prints no secrets):

```bash
uv run python scripts/platforms_smoke.py
```

Start it:

```bash
uv run python -m salid.platforms
```

It serves `POST /line/webhook` and `GET /health` on `SALID_LINE_HOST:SALID_LINE_PORT`. Requests
without a valid `X-Line-Signature` are rejected with 401. The reply is 200 at once and the message
is handled in the background, so LINE never times out.

## 3. Expose the webhook (HTTPS required by LINE)

**Testing**: a Cloudflare quick tunnel.

```bash
cloudflared tunnel --url http://localhost:8091
# prints https://something.trycloudflare.com
```

**Production**: run the process in a container (Cloud Run, a VM) behind a reverse proxy or load
balancer that terminates TLS and forwards `/line/webhook` to port 8091. Cloud Run needs a service
(not a worker pool) with `SALID_LINE_HOST=0.0.0.0`; keep one instance (`--max-instances 1`) while
conversation windows and rate limits live in memory.

Then in the LINE console → **Messaging API** → **Webhook URL**: `https://<your host>/line/webhook`,
press **Verify** (must say Success), and turn **Use webhook** on.

## Limits worth knowing

- Push messages count against your LINE monthly message quota; replies do not. Salid replies with the
  token whenever it can.
- The tenant of LINE is the channel (the bot), so everyone talking to the same bot shares one ledger
  and one settings document (`salid_guilds/{tenant id}`).
- Salid sees all messages in groups it was added to (that is how LINE works). Tell your members it is
  there; remove it from the group to stop recording.
