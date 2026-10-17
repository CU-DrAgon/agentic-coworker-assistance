# Salid on WhatsApp

Salid answers WhatsApp chats with the same brain as in Discord: general chat, links and web search,
questions about the receipt ledger, photos and PDFs you send, and the chat's own history. It uses the
**WhatsApp Business Platform (Cloud API)** from Meta, which delivers messages to a **webhook**, so
this one needs a **public HTTPS URL** (a reverse proxy or a tunnel in front of port 8092).

Run it: `uv run python -m salid.platforms` (it starts every platform that has its secrets set).

## What it does

- **Answers** every message sent to the business number (1:1 chats only, see limitations).
  Each customer is one room; the tenant is the business phone number.
- **Replies** in WhatsApp formatting (`*bold*`, `_italic_`, `~strike~`, `` ```code``` ``), at most 4096
  characters per message, longer answers are split. The first message quotes the question. The message
  is marked read and a "typing…" indicator shows while Salid works (it lasts up to 25 s).
  A reply that hits Meta's rate limit (error 130429 or 131056) is retried twice (after 4 s and 16 s).
  If a part still cannot be sent, the log says how many parts were not sent; nothing is dropped silently.
- **Files**: images and documents (PDF, ...) sent to Salid, up to 15 MB each. Downloaded only when
  Salid answers. Receipts are recorded in the ledger of this number. **Voice notes and other audio** are
  passed to the core as audio files (`audio/ogg`, ...), but the core only reads images, PDFs and text, so
  it skips them as an unsupported file type (no transcription yet). Video, stickers, locations and
  reactions are ignored.
- **Ledger questions**. Executives (`SALID_WHATSAPP_EXEC_NUMBERS`) see everyone's receipts, everyone
  else only their own.
- **Team memory**: each chat sees only its own history, never another customer's.
- **Asking for a person**: a short message such as "human", "คน" or "ติดต่อเจ้าหน้าที่" gets a fixed
  answer (in English and Thai) saying to contact the workspace administrators directly. The model is
  not called. Meta's policy requires a clear way to reach a human; the text is a constant in the code.

Salid **only replies to inbound messages**, so it is always inside the 24-hour customer service
window. It never sends a proactive message (a briefing, a reminder, a broadcast): outside the window
WhatsApp only allows pre-approved **message templates**, which are not implemented.

Not in phase 1 (Discord-only for now): scheduling and meetings, PDF reports and email, deliver-file,
executive-assistant tools, broadcasts, polls, voice, deep tasks, video, image descriptions in memory.

### Usernames and BSUIDs (since April 2026)

A WhatsApp user who adopts a username may be sent to us **without a phone number**: `from` and `wa_id`
are missing, but `from_user_id` (the business-scoped user id, `US.1349...`) is always there.

- **User**: the phone number (`from`) when present, otherwise the BSUID.
- **Room** (the chat and its memory): the BSUID whenever the message has one, otherwise the phone number.
  The BSUID never changes, while a phone number can appear (we messaged them in the last 30 days) or
  disappear, so keying on it keeps one customer in one room for good.
- **Reply**: `to` for a phone number, `recipient` for a BSUID.
- **Executives** are matched on the phone number. A user we only know by BSUID is **not** an executive.
  (The adapter can accept a set of executive BSUIDs, but `SALID_WHATSAPP_EXEC_NUMBERS` keeps digits
  only, so they cannot be listed there yet.)
- A person seen both ways has two user identities (so their own-receipts view can differ), one room.

## 1. Meta developer app

1. Go to <https://developers.facebook.com/apps> → **Create app** → add a name and email → use case
   **Connect with customers through WhatsApp** → choose (or create) a **business portfolio** → Create app.
2. Open **Customize use case → Connect on WhatsApp → API Setup**. Connect a WhatsApp Business account
   (pick one, or **Create WhatsApp Business account**). A **test WABA and test number** are created for
   you.
3. **API Setup** shows the test number's **Phone number ID** (`SALID_WHATSAPP_PHONE_NUMBER_ID`; not the
   phone number itself) and your **WhatsApp Business Account ID** (WABA ID, needed in step 4 below).
   Add your own phone under "To" and send the first message; replying to it opens the 24-hour window.

## 2. Access token

- **Temporary token** (API Setup page): expires quickly, only for the first test.
- **Permanent token**: Business Settings → Users → **System users** → add a system user (Admin, or an
  Employee that you give access) → **Add assets**: your app with **Manage app** (full control) and your
  WhatsApp account with **Manage WhatsApp Business Account** → **Generate token**: pick the app, no
  expiry, and the permissions `whatsapp_business_messaging` and `whatsapp_business_management`
  (`business_management` too if you also use the business API yourself).

This is `SALID_WHATSAPP_ACCESS_TOKEN`. Treat it like a password.

## 3. App secret

App settings → **Basic** → **App secret** → Show. This is `SALID_WHATSAPP_APP_SECRET`: Meta signs
every webhook with it (`X-Hub-Signature-256`) and Salid rejects anything unsigned or wrongly signed
before reading it.

## 4. Webhook

1. Choose any long random string as the **verify token** (`SALID_WHATSAPP_VERIFY_TOKEN`).
2. Start Salid and make `http://127.0.0.1:8092/whatsapp/webhook` reachable over HTTPS with a real
   certificate (self-signed ones are rejected; for a quick test: `cloudflared tunnel --url
   http://127.0.0.1:8092`). In a container set `SALID_WHATSAPP_HOST=0.0.0.0`.
3. **Use cases → Customize → Configuration** (older apps: WhatsApp → Configuration) → Webhook → **Edit**:
   callback URL `https://<your host>/whatsapp/webhook`, the verify token → **Verify and save**. Meta calls
   `GET` with `hub.challenge`; Salid echoes it only for the right token.
4. Subscribe to the **`messages`** field.
5. Subscribe the app to your WhatsApp account (needed for a real WABA):
   `curl -X POST "https://graph.facebook.com/v26.0/<WABA_ID>/subscribed_apps" -H "Authorization: Bearer $SALID_WHATSAPP_ACCESS_TOKEN"`.
6. Switch the app to **Live** mode (top bar of the dashboard). In Development mode some webhooks are not
   sent.

## 5. A real phone number (production)

The test number only reaches the recipients you added. For customers:

1. The number must be yours, able to receive an SMS or voice call, and **not** already used in the
   WhatsApp or WhatsApp Business app (delete that account first).
2. Add it under API Setup (or WhatsApp Manager) and verify it with the code.
3. **Register it for the Cloud API** with a two-step verification PIN you choose (6 digits):
   `POST /<PHONE_NUMBER_ID>/register` with `{"messaging_product":"whatsapp","pin":"<PIN>"}`. Verifying
   alone is not enough.
4. Get the **display name** approved (Meta reviews it; re-register within 14 days after an approval).
5. Use the new number's Phone number ID in `SALID_WHATSAPP_PHONE_NUMBER_ID`.

Business verification is not needed for replies inside the 24-hour window (it raises the limits for
business-initiated messages, and is a prerequisite for the Official Business Account, see Groups).

## 6. Environment (`.env`)

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
  parsing**, caps the body at 3 MB (Meta's documented maximum payload) and answers 200 at once; Meta
  retries are de-duplicated by message id.
- Tokens, the verify token, the app secret, media URLs, message text and phone numbers are never
  logged (a `wa_id` is a phone number; only the internal room id appears in logs). A failed delivery
  status is logged by error code only.
- Put TLS in front. Meta only calls HTTPS URLs.

## Cost

Since 2025-07-01 Meta charges per delivered **template** message only. Non-template replies inside the
24-hour window (everything Salid sends) are **free**. Gemini usage is separate. Check Meta's pricing page
for template rates, which Salid does not use.

## Policy risks (read before onboarding customers)

- **AI providers.** Meta's Business Solution Terms (updated 2026-01-15) only allow "AI Providers" to
  offer **general-purpose AI assistants** on the platform where Meta is legally required to permit it.
  Thailand is not one of those markets. A bot for one company's receipts and ledger is a business
  using AI for a specific job; a multi-tenant general assistant (open chat, web search) can be treated
  as an AI Provider and the account can be restricted. Before onboarding customers outside your own
  company, **ask Meta (Direct Support) in writing**, and keep the WhatsApp scope task-specific.
- **Human contact.** Automation must offer a clear way to reach a person; Salid answers "human" /
  "ติดต่อเจ้าหน้าที่" with a fixed text (it names the workspace administrators generically).
- **Sensitive identifiers.** The policy forbids asking people to send full financial account numbers
  or personal ID card numbers. Receipts and transfer slips can contain them: tell users to cover them.
- **No sharing between customers.** Chats are isolated; do not add cross-chat features on this channel.
- **Edits and deletes are not delivered** on a normal Cloud API number (an edited message arrives as an
  unsupported message, a deletion arrives as nothing; both events exist only for numbers also used in
  the WhatsApp Business app). Salid cannot mirror them in its memory: say so in your privacy notice.

## Limitations

- **No groups.** A Groups API exists, but it needs an **Official Business Account** (business
  verification, at least 30 days on the platform, two-step verification and an approved display name).
  It can only **create new groups** (at most 8 participants) that people join through an invite link;
  a business number **cannot join an existing group**. It delivers every participant message, and has no
  edit/delete. It does not fit "Salid in our team's group", so it is not supported now.
- **24-hour window**: free-form replies only within 24 hours of the customer's last message; no
  proactive messages without templates.
- **Test number**: sandbox only, it can reach only the numbers you add. For production you need your own
  number (section 5).
- Quoted replies to Salid's own messages are recognised only until the process restarts.
