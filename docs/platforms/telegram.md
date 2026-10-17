# Salid on Telegram

Salid answers in Telegram private chats and groups with the same brain as in Discord: general chat,
links and web search, questions about the receipt ledger, photos and PDFs you send, and team memory
of the room. It uses **long polling** (`getUpdates`), so it needs **no public URL, no webhook and no
open port**: the process only makes outbound HTTPS calls to `api.telegram.org`.

Run it: `uv run python -m salid.platforms` (it starts every platform that has its secrets set).

## What it does

- **Answers** every message in a private chat. In a group only when you @mention the bot, reply to one
  of its messages, say its name ("สลิด", "Salid"), or follow up within ~3 minutes of its answer to you
  (a cheap model checks the follow-up is meant for Salid). Everything else in a group is only remembered.
  When the bot is added to a group it posts a short Thai/English notice (what is remembered, that
  content is processed by Google Vertex AI, how to stop it). If it is removed from a group or blocked by
  a user (`my_chat_member`), it stops sending there.
  Forum topics are separate threads: Salid answers inside the topic it was asked in.
- **Commands**: `/start` and `/help` (registered with `setMyCommands`, Thai and English descriptions).
  `/help` is answered with a short usage text without calling the model. In a group a command only
  counts when it names the bot (`/help@YourBot`, `/start@YourBot`); the `@YourBot` suffix is removed
  before Salid reads it, and commands for other bots are ignored.
- **Albums**: photos sent together (same `media_group_id`) are collected for ~1.5 s and handled as one
  message with all files and the caption. In a group the album is addressed if any caption tags Salid.
- **Replies** in plain text (Telegram's formatting modes reject a whole message on one stray
  character), at most 4096 characters per message, longer answers are split. In groups the first
  message replies to the one that triggered it. A "typing…" indicator shows while it works.
  Sending is paced to Telegram's limits: at least 1 s between messages to one chat, at most 20 a minute
  to a group, and a `429` is waited out (`retry_after`, up to 120 s) before the next chunk goes. If the
  wait is longer, or the chat is gone, the failure is logged; messages are never dropped silently.
  In a private chat with topics enabled Salid answers inside the topic the message came from.
- **Files**: photos (largest size) and documents (PDF, images) sent to Salid, up to 15 MB each (the Bot
  API itself serves at most 20 MB). They are downloaded only when Salid is actually addressed.
  Receipts are recorded in the ledger of this bot.
- **Edits** (`edited_message`) update the remembered text.
- **Ledger questions**. Executives (`SALID_TELEGRAM_EXEC_USER_IDS`) see everyone's receipts,
  everyone else only their own. **Group admins are not executives.**
- **Team memory** of each room. A bot has no call that lists all members of a group (it can only read
  the admins with `getChatAdministrators` and the count with `getChatMemberCount`; `getChatMember` is
  reliable for others only if the bot is an admin), so Salid cannot know who may read a room and memory
  is **never shared between rooms**: a group's memory is used only inside that group, a private chat uses only its own
  history. This is on purpose (fail closed).

Not in phase 1 (Discord-only for now): scheduling and meetings, PDF reports and email, deliver-file,
executive-assistant tools, broadcasts, polls, voice, deep tasks, audio and video, image descriptions in
memory, receipt forwarding, profiles.

## 1. Create the bot

1. In Telegram open **@BotFather** → `/newbot` → pick a name and a username (ends in `bot`). Copy the
   **token** (`123456:ABC…`). Treat it like a password.
2. `/setprivacy` → choose your bot → **Disable**. With privacy mode on (the default) a bot in a group only
   receives commands, @mentions and replies to itself, so it could not remember the room. Disabling it
   lets Salid read all group messages. (Alternative: make the bot a group admin, which also lifts the
   restriction.) Apply this **before** adding the bot to a group. Changing the setting later only takes
   effect after you **remove the bot from the group and add it again**. At start Salid reads `getMe`
   (`can_read_all_group_messages`) and logs a clear warning if privacy mode is still on.
3. `/setjoingroups` → **Enable** if you want people to add it to groups, **Disable** to keep it private.
4. Set a **privacy policy** with BotFather (see [Compliance](#compliance)). Optional: `/setdescription`,
   `/setuserpic`. The command list (`/start`, `/help`) is registered by Salid itself at start, so
   `/setcommands` is not needed.

## 2. Configure

```bash
SALID_TELEGRAM_BOT_TOKEN=123456:ABC...
SALID_TELEGRAM_EXEC_USER_IDS=123456789,987654321   # optional: numeric Telegram user ids of executives
```

A Telegram user id is a number (not the @username). Message **@userinfobot** or read `from.id` in an
update to find yours.

Check the token without starting anything (read-only `getMe`, prints only the bot's username and id):

```bash
uv run python scripts/platforms_smoke.py
```

Start it:

```bash
uv run python -m salid.platforms
```

At start Salid calls `deleteWebhook` (a webhook and `getUpdates` exclude each other), then polls with a
30 s timeout. A failed poll is retried with a 1 s → 60 s backoff and re-reads the same updates, so none
are lost. Each batch is handled concurrently (an album needs that) and awaited for up to 180 s; the
offset only moves past updates whose handler finished, so an update that failed on a network error,
429 or 5xx is redelivered (at most 3 tries, and updates already handled are skipped). Run **one** instance per token: a second poller (or a webhook) gets `409 Conflict`; Salid logs
a clear error and keeps retrying with at least 5 s between tries. After a week without any update
Telegram may number the next `update_id` at random, so after more than 6 idle days Salid logs it and
resets its offset.

## Limits worth knowing

- **No history.** The Bot API cannot read past messages: memory starts when Salid joins, and there is no
  history import.
- **No member list.** A bot can read the group admins and the member count, but not the full list of
  members, hence the per-room memory rule above.
- **No delete events.** Telegram never tells a bot that a message was deleted, so a deleted message stays
  in Salid's memory until the retention window expires. Edits are tracked, deletions cannot be.
- The tenant of Telegram is the bot (its id from `getMe`), so everyone talking to the same bot shares
  one ledger and one settings document (`salid_guilds/{tenant id}`). Use one bot token per customer.
- The bot token is part of every Bot API URL. Salid never logs URLs or request lines, but do not paste
  the token into chats or commit it; chat ids are logged only as a short hash; rotate it with BotFather's `/token` if it leaks.
- With privacy mode off Salid sees all messages in groups it was added to. It posts a notice when it
  joins; remove it from the group to stop recording.

## Compliance

Telegram's [Bot Developer Terms](https://telegram.org/tos/bot-developers) apply to whoever runs the bot
(the operator), not to this code. Before using Salid with real groups:

- **Privacy policy.** Every bot needs an easily accessible privacy policy saying what it stores, how and
  why. Telegram's Standard Bot Privacy Policy says user data is never shared with third parties, which is
  **not true for Salid**: message and file content is sent to Google Vertex AI (Gemini) to answer and to
  summarise team memory. Write your own policy that discloses this and any other processor (Firestore,
  Cloud Storage, Cloud Run), and set it in @BotFather (Bot Settings → Edit Bot → Privacy Policy). Your
  policy then overrides the standard one.
- **Consent and data minimisation (§4.3).** Do not collect more than the service needs, and never build
  datasets or models from group content. With privacy mode off Salid records what people say in a group
  without them talking to the bot, so tell the members: Salid's join notice does this, and you should
  agree its wording with the group admins (they decide whether the bot stays). Removing the bot stops
  recording.
- **Deletion (§4.2).** Delete a user's data on their or Telegram's request, when it is no longer needed,
  and everything when the bot stops running. Memory expires after the retention window
  (`SALID_MEMORY_RETENTION_DAYS`, per tenant); for earlier deletion use the tenant's memory tools or
  delete the records in Firestore, and say in your policy how users ask for it. Telegram sends no event
  when a user deletes a message, so those copies stay until then.
- **Security (§4.4, §4.5).** Keep the token secret (it is attributed to you), encrypt data at rest
  (Firestore and Cloud Storage do by default), and tell users about a breach.
- **Conduct (§5.2).** No spam, no circumventing rate limits (Salid paces its sends), and tell users when
  the bot's scope changes significantly.
- Telegram may remove chats or data with your bot at any time, so do not treat Telegram as the only copy.
