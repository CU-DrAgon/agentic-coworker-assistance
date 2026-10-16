# Salid on WhatsApp

Salid answers WhatsApp chats with the same brain as in Discord: general chat, links and web search,
questions about the receipt ledger, photos and PDFs you send, and the chat's own history. It uses the
**WhatsApp Business Platform (Cloud API)** from Meta, which delivers messages to a **webhook**, so
this one needs a **public HTTPS URL** (a reverse proxy or a tunnel in front of port 8092).

Run it: `uv run python -m salid.platforms` (it starts every platform that has its secrets set).

## What it does

- **Answers** every message sent to the business number (there are no groups, see limitations).
  Each customer is one room; the tenant is the business phone number.
- **Replies** in WhatsApp formatting (`*bold*`, `_italic_`, `~strike~`, `` ```code``` ``), at most 4096
  characters per message, longer answers are split. The first message quotes the question. The message
  is marked read and a "typing…" indicator shows while Salid works (it lasts up to 25 s).
- **Files**: images and documents (PDF, ...) sent to Salid, up to 15 MB each. Downloaded only when
  Salid answers. Receipts are recorded in the ledger of this number. Audio, video, stickers and
  locations are ignored.
- **Ledger questions**. Executives (`SALID_WHATSAPP_EXEC_NUMBERS`) see everyone's receipts, everyone
  else only their own.
- **Team memory**: each chat sees only its own history, never another customer's.

Salid **only replies to inbound messages**, so it is always inside the 24-hour customer service
window. It never sends a proactive message (a briefing, a reminder, a broadcast): outside the window
WhatsApp only allows pre-approved **message templates**, which are not implemented.

Not in phase 1 (Discord-only for now): scheduling and meetings, PDF reports and email, deliver-file,
executive-assistant tools, broadcasts, polls, voice, deep tasks, audio and video, image descriptions
in memory.

## 1. Meta developer app

1. Go to <https://developers.facebook.com/apps> → **Create app** → use case **Other** → type
   **Business**.
2. In the app dashboard **Add product → WhatsApp** (it creates or links a WhatsApp Business Account).
3. **WhatsApp → API Setup** gives you a **test number** and its **Phone number ID**
   (`SALID_WHATSAPP_PHONE_NUMBER_ID`; not the phone number itself). The test number can only message up
   to **5 verified recipient numbers**: add yours under "To" in the same page and confirm the code.

## 2. Access token

- **Temporary token** (API Setup page): valid 24 hours, fine for a first try.
- **Permanent token**: Business Settings → Users → **System users** → add a system user (Admin) →
  assign the app and the WhatsApp account → **Generate token** with the permissions
  `whatsapp_business_messaging` and `whatsapp_business_management`, no expiry.

This is `SALID_WHATSAPP_ACCESS_TOKEN`. Treat it like a password.

## 3. App secret

App settings → **Basic** → **App secret** → Show. This is `SALID_WHATSAPP_APP_SECRET`: Meta signs
every webhook with it (`X-Hub-Signature-256`) and Salid rejects anything unsigned or wrongly signed
before reading it.

## 4. Webhook

1. Choose any long random string as the **verify token** (`SALID_WHATSAPP_VERIFY_TOKEN`).
2. Start Salid and make `http://127.0.0.1:8092/whatsapp/webhook` reachable over HTTPS (for a quick
   test: `cloudflared tunnel --url http://127.0.0.1:8092`). In a container set `SALID_WHATSAPP_HOST=0.0.0.0`.
3. WhatsApp → **Configuration** → Webhook → **Edit**: callback URL `https://<your host>/whatsapp/webhook`,
   the verify token → **Verify and save**. Meta calls `GET` with `hub.challenge`; Salid echoes it only
   for the right token.
4. In the same section **Manage** → subscribe to the **`messages`** field.

## 5. Environment (`.env`)

```
SALID_WHATSAPP_ACCESS_TOKEN=EAAG...
SALID_WHATSAPP_APP_SECRET=...
SALID_WHATSAPP_VERIFY_TOKEN=a-long-random-string
SALID_WHATSAPP_PHONE_NUMBER_ID=105555000111222
SALID_WHATSAPP_EXEC_NUMBERS=66811112222,66833334444   # optional, digits only (+ and spaces are ignored)
# SALID_WHATSAPP_API_VERSION=v26.0
# SALID_WHATSAPP_HOST=127.0.0.1
# SALID_WHATSAPP_PORT=8092
```

All four secrets/ids are needed for the platform to start. Check them without sending anything:
`uv run python scripts/platforms_smoke.py` (looks up the number and prints only its verified name).

## Security

- The webhook verifies `X-Hub-Signature-256` (HMAC-SHA256 of the raw body, constant time) **before
  parsing**, caps the body at 1 MB and answers 200 at once; Meta retries are de-duplicated by message id.
- Tokens, the verify token, the app secret, media URLs, message text and phone numbers are never
  logged (a `wa_id` is a phone number; only the internal room id appears in logs).
- Put TLS in front. Meta only calls HTTPS URLs.

## Limitations

- **No groups**: the official API only has 1:1 chats with the business number.
- **24-hour window**: free-form replies only within 24 hours of the customer's last message; no
  proactive messages without templates.
- **Test number**: 5 recipients, "sandbox" branding. For production you need your own number, a
  verified **business** (Meta Business verification) and, for higher limits, a display name approval.
- **Cost**: Meta charges per conversation/message category (service replies inside the window are
  currently free, templates are billed). Check Meta's current pricing page; Gemini usage is separate.
