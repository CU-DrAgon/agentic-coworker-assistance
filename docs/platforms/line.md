# Salid on LINE

Salid answers in LINE 1:1 chats and groups with the same brain as in Discord: general chat, links
and web search, questions about the receipt ledger, images and PDFs you send, and team memory of
the room. LINE delivers messages to a **webhook**, so Salid needs a public HTTPS URL.

Run it: `uv run python -m salid.platforms` (it starts every platform that has its secrets set).

## What it does

- **Answers** every message in a 1:1 chat. In a group or multi-person chat only when you @mention
  Salid, say its name ("สลิด", "Salid"), or follow up within ~3 minutes of its answer to you (a cheap
  model checks the follow-up is meant for Salid). Everything else in a group is only remembered.
  An @All is not a call for Salid.
- **Replies** with the reply token (free; it is valid for about a minute and Salid uses it for up to
  50 seconds), quoting the message it answers in groups. The whole answer goes into one reply: chunks
  are merged up to 5000 characters (UTF-16 units) per message and an answer is capped at LINE's 5
  message objects per call. A push (paid, see Pricing) is used only when the token is gone or refused.
  A push that fails with 429 (monthly quota used up) is logged as an error and the answer is lost, so
  watch the quota. In 1:1 chats the typing animation starts as soon as a message arrives. Markdown is
  turned into plain text (LINE renders no markup).
- **Files**: images and PDFs sent to Salid, up to 15 MB each. In a 1:1 chat just send them. In a group
  LINE cannot put an @mention on an image, so **quote** the image (or file) and tag Salid in the text
  ("@Salid is this receipt ok?"); Salid remembers recent images and files for 10 minutes. Receipts
  are recorded in the ledger of your LINE channel.
- **Ledger questions**. Executives (`SALID_LINE_EXEC_USER_IDS`) see everyone's receipts, everyone
  else only their own. A group member without a user id (LINE for PC) is treated as an anonymous
  non-executive.
- **Team memory** of each room. Memory is **never shared between rooms**: a group's memory is used
  only inside that group, a 1:1 chat uses only its own history. This is on purpose (fail closed).
  Edited messages (group `messageEdited` events, newest edit wins) update memory; an unsent message
  is deleted from memory. When Salid joins a group or someone adds it as a friend it posts a short
  Thai + English notice: what is remembered, for how many days, how to make it forget (unsend) and how
  to stop it (remove Salid from the group). A tenant that still has the global memory retention (keep
  forever) and was never saved in the hub gets **7 days** on the first join/follow after a start; an
  admin can change it in the hub, and a hub choice (keep forever included) is never overridden.

Not in phase 1 (Discord-only for now): scheduling and meetings, PDF reports and email, deliver-file,
executive-assistant tools, broadcasts, polls, voice, deep tasks, audio and video, image descriptions
in memory, receipt forwarding, profiles. There is also no chat history import: LINE's API cannot read
past messages, so memory starts when Salid joins. A receipt recorded from a message that is later
unsent is **not** voided automatically yet.

## 1. Create the account and the channel

Since 2024-09-04 a Messaging API channel can no longer be created in the Developers Console. Start
from a LINE Official Account.

1. Register a Business ID and create a **LINE Official Account** (<https://manager.line.biz/>). Pick
   Thailand as the country to get the Thai plans. A free, **unverified** account is enough to run Salid.
2. In LINE Official Account Manager: **Settings → Messaging API → Use Messaging API** (needs the Admin
   role, web version). Choose or create a **Provider**. **A provider cannot be changed afterwards**, and
   LINE user ids are issued per provider: executive ids only work under the provider of this account.
3. In the LINE Developers Console, open the channel:
   - **Basic settings**: copy the **Channel secret**; **Your user ID** (at the bottom) is your own id for
     `SALID_LINE_EXEC_USER_IDS`.
   - **Messaging API** tab: issue a **Channel access token (long-lived)**; turn **Allow bot to join
     group chats** on.
4. In OA Manager → **Response settings**: **Auto-response messages: off**, **Greeting message: off**
   (Salid greets by itself), **Response mode: Bot** (a pure bot; Chat mode is for humans answering in
   OA Manager). Allow users to send photos, videos and files if you want receipts by image.
5. Scan the QR code on the **Messaging API** tab to add Salid as a friend, and invite it to a group.
   Only one Official Account can be in a given group.

Group member lists need a **verified** account (888 THB application fee in Thailand, 2-3 business
days review); Salid does not need them.

## 2. Configure

```bash
SALID_LINE_CHANNEL_SECRET=...
SALID_LINE_CHANNEL_ACCESS_TOKEN=...
SALID_LINE_EXEC_USER_IDS=Uxxxxxxxx,Uyyyyyyyy   # optional: LINE user ids of executives
# SALID_LINE_HOST=127.0.0.1                    # 0.0.0.0 in a container
# SALID_LINE_PORT=8091
```

A LINE user id (`U…`, 33 characters) is issued per provider and is not visible in the app. Your own
is shown in the Developers Console: channel → **Basic settings** → **Your user ID** (your Business ID
must be linked to your LINE account). The Console has no webhook payload log, so for someone else's id
read `source.userId` from the webhook payload yourself.

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
press **Verify** (must say Success), turn **Use webhook** on, and turn **Webhook redelivery** on:
LINE then re-sends events it could not deliver (LINE had several outages in 2026). Salid skips an
event it already handled (same `webhookEventId`), so redelivery never answers twice. The handled-event
list is in memory (5000 events): after a restart a redelivered event can be handled again.
Cloud Run needs CPU always allocated (instance-based billing) and `--min-instances 1`, because the
answer is computed after the 200 is returned and LINE times out after 2 seconds.

## Linking accounts to your organization

Salid treats a LINE account as a **guest of its chat** until the person is verified. Verifying
ties the account to a member of the organization (the company in the command center), which later
lets Salid use what that member may read on the organization's other platforms in a private chat.

1. In the command center, sign in with Google, open your organization and press **Link LINE**
   (API: `POST /api/center/orgs/{org}/identities/line/code`). You get a code like `1234-5678`
   that works **once, for 15 minutes**.
2. Send it to Salid in a **private chat**: `link 1234-5678` (Thai: `เชื่อมบัญชี 1234-5678`).
   Salid answers with a fixed line and never shows the message to the model or stores it.
3. Unlink from the web (`DELETE /api/center/orgs/{org}/identities/line`); removing the
   LINE workspace from the organization also forgets these links.

Rules: a code typed in a **group is ignored** (Salid asks you to use a private chat); five wrong
attempts per person in 10 minutes pauses that person; one LINE account belongs to at most one
member of an organization; every failure says only "invalid or expired".

**Connecting the bot itself to the organization.** An owner or admin presses **Attach LINE
bot** (`POST /api/center/orgs/{org}/workspaces/line/attach-code`) and sends the code to the
bot in a private chat. A bot that already belongs to another organization is refused. A bot that
is attached to no organization keeps working on its own as before.

LINE user ids are per provider; link from the LINE Official Account that belongs to this bot channel.


## Pricing in Thailand (LINE for Business, excluding 7% VAT)

- **Free**: 300 messages per month. **Basic**: 1,280 THB per month, 15,000 messages (0.1 THB per extra
  message). **Pro**: 1,780 THB per month, 35,000 messages (the pricing page says 0.06 THB per extra
  message, the help center says 0.6: check OA Manager).
- **Replies are free** (not counted). **Pushes are counted per recipient**: a push to a group of 40
  people uses 40 messages, whatever the number of message objects (up to 5 per request). Ten slow
  answers in a 30-person group can use the whole Free plan. Salid therefore replies with the token
  whenever it can and fits the answer into one call.
- When the monthly limit is used up, a push returns 429 ("You have reached your monthly limit") and
  the answer is lost. Read the quota with `GET /v2/bot/message/quota` and
  `GET /v2/bot/message/quota/consumption`.

## Policy questions to settle before real use

- **24 hours.** LINE's User Data Policy (3.2.3) says LINE user information stored longer than 24 hours
  needs a notice to the users (Salid posts one when it joins) and that "Friend and Group information
  must not be stored for more than 24 hours" whether or not users were told. It is **not settled**
  whether the text of group messages counts as "Group information". Salid keeps group memory (default
  7 days on LINE) on the assumption that it does not; get LINE's or a lawyer's answer, and if the
  answer is yes, set the hub retention to 1 day or turn memory off for LINE. User ids and display names
  are kept only in memory records and caches.
- **External links.** The LINE Official Account API Terms (5.3) say messages must not contain links to
  web pages you do not control. Answers that cite web search sources can break that: switch the `web`
  feature off for the LINE tenant in the hub until LINE confirms.
- **Deletion.** Users can ask for their data to be deleted (User Data Policy 3.5): unsend deletes a
  message from memory; remove Salid from the group to stop recording.

## Limits worth knowing

- The tenant of LINE is the channel (the bot), so everyone talking to the same bot shares one ledger
  and one settings document (`salid_guilds/{tenant id}`).
- Salid sees all messages in groups it was added to (that is how LINE works). The join notice tells
  the members; remove Salid from the group to stop recording.
- A LINE group sees the bot's answers quoting the asker. Members who only use LINE for PC have no
  user id in the webhook; their messages are remembered anonymously.
- Edit events (`messageEdited`) exist only in group chats, not 1:1 chats with an Official Account.
