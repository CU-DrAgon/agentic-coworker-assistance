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
  Forum topics are separate threads: Salid answers inside the topic it was asked in.
- **Replies** in plain text (Telegram's formatting modes reject a whole message on one stray
  character), at most 4096 characters per message, longer answers are split. In groups the first
  message replies to the one that triggered it. A "typing…" indicator shows while it works.
- **Files**: photos (largest size) and documents (PDF, images) sent to Salid, up to 15 MB each (the Bot
  API itself serves at most 20 MB). They are downloaded only when Salid is actually addressed.
  Receipts are recorded in the ledger of this bot.
- **Edits** (`edited_message`) update the remembered text.
- **Ledger questions**. Executives (`SALID_TELEGRAM_EXEC_USER_IDS`) see everyone's receipts,
  everyone else only their own. **Group admins are not executives.**
- **Team memory** of each room. A bot cannot list the members of a group, so memory is **never shared
  between rooms**: a group's memory is used only inside that group, a private chat uses only its own
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
   restriction.) Apply this **before** adding the bot to a group; if it is already there, remove and
   re-add it for the change to take effect.
3. `/setjoingroups` → **Enable** if you want people to add it to groups, **Disable** to keep it private.
4. Optional: `/setdescription`, `/setuserpic`, `/setcommands` (Salid needs no commands; `/start` works).

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
are lost. Run **one** instance per token: a second poller gets `409 Conflict`.

## Limits worth knowing

- **No history.** The Bot API cannot read past messages: memory starts when Salid joins, and there is no
  history import.
- **No member list.** Bots cannot see who is in a group, hence the per-room memory rule above.
- **No delete events.** Telegram never tells a bot that a message was deleted, so a deleted message stays
  in Salid's memory until the retention window expires. Edits are tracked, deletions cannot be.
- The tenant of Telegram is the bot (its id from `getMe`), so everyone talking to the same bot shares
  one ledger and one settings document (`salid_guilds/{tenant id}`). Use one bot token per customer.
- The bot token is part of every Bot API URL. Salid never logs URLs or request lines, but do not paste
  the token into chats or commit it; rotate it with BotFather's `/revoke` if it leaks.
- With privacy mode off Salid sees all messages in groups it was added to. Tell your members it is
  there; remove it from the group to stop recording.
