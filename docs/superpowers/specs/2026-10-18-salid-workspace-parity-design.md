# Salid on every workspace: one brain, thin platform adapters

Status: agreed direction (2026-10-18). Step 1 (ids) and step 2a (the `Workspace` interface in `salid/workspace.py`, `DiscordWorkspace` in `workspace_discord.py`, `SlackWorkspace` in `platforms/workspace_slack.py`, plus a `FakeWorkspace` for tests) are done; 2b is next: `_agent_kwargs` takes a `Workspace`, one tool group at a time. Read this before adding a Salid feature.

## Goal

Everything Salid does in Discord (team memory, tasks and follow-ups, projects, meetings and
reminders, briefs and reports, documents, approvals, votes, wiki, media) works the same way in
Slack, and as far as each platform allows in LINE, Telegram and WhatsApp. A feature is written
once and every platform gets it. The team adds features in Discord first; they must reach Slack
without a second implementation.

## The rule (applies to every new feature)

1. **Feature logic never imports `discord`.** Logic lives in a pure module (`tasks.py`,
   `projects.py`, `reminders.py`, ...) that takes ids, text and a `Workspace`. Today about 30 of
   these modules already have zero Discord references; keep it that way.
2. **Every platform action goes through `Workspace`** (below): finding a person, posting, a DM,
   an attachment, a button, a schedule. Never call `discord.*`, `slack_sdk` or a platform API
   from a tool or a feature module.
3. **Tools are built once**, in one function that takes a `Workspace` (today
   `SecretaryBot._agent_kwargs`, which takes Discord objects). Chat, voice and every platform
   call it, so they cannot drift.
4. **A capability a platform lacks is declared, not faked.** `Workspace.supports(...)` returns
   False and the tool is left out (or answers "not available on Slack yet"). Example: voice
   meetings exist only on Discord.
5. **Tests**: a feature's tests run against a fake `Workspace`, not a fake Discord client. A
   parity test lists every tool and the platforms that offer it, so a missing adapter is visible.

## `Workspace`: the one interface

One object per (platform, tenant), built per request with the asker and the room.

```python
class Workspace(Protocol):
    platform: str            # "discord" | "slack" | "line" | "telegram" | "whatsapp"
    tenant_id: int           # the guild_id every store uses (see "Ids" below)

    # people
    async def person(self, ref: str) -> Person | None      # name, @mention, email, platform id
    async def people(self) -> list[Person]                 # members Salid may tag
    async def is_exec(self, user: int) -> bool
    async def rank(self, user: int) -> int                 # role hierarchy (COO rules)

    # rooms and memory scope
    async def rooms(self) -> list[Room]                    # readable channels
    async def room_view(self, room: int) -> RoomView       # members, or None = unknown (fail closed)

    # output
    async def post(self, room: int, text: str, *, thread=None, files=(), buttons=()) -> MsgRef
    async def dm(self, user: int, text: str, *, files=(), buttons=()) -> MsgRef | None
    async def edit(self, ref: MsgRef, text: str, *, buttons=()) -> None
    def mention(self, user: int) -> str                    # "<@id>" / "<@U..>" / plain name

    # capabilities
    def supports(self, feature: str) -> bool               # "buttons", "voice", "events", ...
    upload_limit: int
```

- **Buttons** are described neutrally (`Button(id, label, style)`) and a persistent callback id
  (`salid:<feature>:<action>:<tenant>:<object>`, the format Discord `DynamicItem`s already use).
  Discord renders `discord.ui`, Slack renders Block Kit `actions` and routes `block_actions` back
  to the same handler. LINE/Telegram can use quick replies / inline keyboards later; until then
  `supports("buttons")` is False and the feature falls back to typed answers (most already have
  typed forms: "task #id done", "ยืนยัน").
- **Schedules and ticks** (reminders, briefs, follow-ups, votes) stay in the bot process tick and
  call `Workspace.dm/post`; the tick iterates tenants of every platform, not only Discord guilds.
- **`room_view` is the security boundary** for memory (`memory_access.py`): unknown audience means
  the room's own memory only. Slack shared channels, LINE, Telegram and WhatsApp already return
  unknown; Discord returns the real reader set.

Implementations:

- `DiscordWorkspace` (new): wraps `discord.Guild`/`Member`; behaviour must not change.
- `SlackWorkspace`: grows out of `salid/platforms/slack.py` (`SlackPort`).
- LINE / Telegram / WhatsApp: grow out of their `Port`s; most write capabilities stay `False`.

## Calendar and meetings: Google Calendar

Discord uses Scheduled Events; Slack has no equivalent and bots cannot start a Huddle. Decision:
**Salid's own meeting store + reminders on every platform, and Google Calendar as the real
calendar** (the web login is moving to Google OAuth, so people will already have linked Google
accounts).

- `scheduling.CalendarBackend` stays the seam. Add `GoogleCalendar` (Calendar API v3, OAuth per
  user, scope `https://www.googleapis.com/auth/calendar.events`, `conferenceData` to create a
  Meet link). `DiscordCalendar` stays for Discord-only voice events.
- One meeting record (`salid_meetings`) can have several calendar copies (Google event id,
  Discord event id). Reminders, no-show nudges and minutes work from the record, so they work on
  Slack even without Google.
- Who owns the event: the organiser's linked Google account. Attendees are invited by email
  (people need an email in their profile / org membership). Without a linked account Salid still
  schedules in its own store and sends reminders with the meeting link, and says the calendar
  invite needs a Google link.
- Free/busy (`freetime.py`) can read Google free/busy later with the same token.
- Tokens: refresh tokens per user in Secret Manager or encrypted in the DocStore, never logged,
  revocable from the hub. Confirm-before-sending stays (`secretary.py` second-call rule).

## Ids: one scheme for a workspace (must fix first)

Two Slack integrations exist and they disagree:

- `salid/connectors/slack.py` (command center `/center`: OAuth install, HTTP Events, public
  channels only) uses `orgs.workspace_memory_id("slack", team_id)`.
- `salid/platforms/slack.py` (the Salid bot, Socket Mode) uses `platforms.ids.tenant_id`.

For team `T0CA03H1676` they give different ids, so the center and the bot would see two different
ledgers, memories and settings for one workspace, and licensing would not find the bot's tenant.
Before building on Slack: pick one function (proposal: `orgs.workspace_memory_id` for tenants,
since licensing and orgs already depend on it, and the platform adapters call it), migrate the
test data, and make one Slack app the bot (Socket Mode internal app per workspace now; HTTP
Events + OAuth when it goes to the Marketplace, see `docs/platforms/slack.md`).

**Done (step 1):** `platforms.ids` now delegates to `orgs` (`workspace_memory_id` for tenants,
`namespaced_id` for rooms, users and the bot author, `message_namespaced_id` for messages, which
hashes workspace + channel + `ts` because the memory key has no channel), and the Slack connector
calls the same functions, so the tenant/room/user/message helpers are shared.

## Central identity and cross-platform memory

Agreed 2026-10-18. The company (`orgs.Org`) is the centre: its members, its workspaces on every
platform, and one shared store. A question asked on any platform can use what the asker may see
from all of the org's workspaces, under the same fail-closed audience rule.

### One person, many platform accounts

- `orgs.Member` (keyed by the Google login email) gets `identities: dict[str, str]`
  (`platform -> platform user id`, e.g. `{"slack": "U0..", "line": "U4..", "telegram": "8629..",
  "whatsapp": "<BSUID or phone>"}`); `discord_user_id` stays and is read as `identities["discord"]`.
- An identity counts only when **verified**; unverified accounts are guests of their room (today's
  behaviour: own receipts, that room's memory).
  - Discord: the existing `/ตั้งค่า` link + Discord login (done).
  - Slack: automatic when the Slack profile email (`users.info` `profile.email`, scope
    `users:read.email`) equals a member email of the org that owns the workspace, and the user is
    a member of that Slack team (`team_id` check as in `SlackPort.is_admin_user`).
  - LINE, Telegram, WhatsApp (no reliable email): a one-time link code. The member signs in on
    the web with Google, presses "Link LINE / Telegram / WhatsApp", gets a 15-minute single-use
    code, and sends it to Salid in a 1:1 chat ("link 1234-5678" / "เชื่อมบัญชี 1234-5678"). The code
    is redeemed in a transaction (same pattern as `salid_link_codes`), binding the sender's platform
    id to that member. Never accept a code in a group chat.
  - Unlink from the web, and on uninstall / revocation of that platform.
- One platform account belongs to at most one member per org; a second claim is refused.

### Workspaces of an org

- Every platform tenant is an `orgs.Workspace` (`platform` in discord, slack, line, telegram,
  whatsapp, clickup, web) whose `memory_id` = `workspace_memory_id(platform, external_id)` (the one
  id scheme, done in step 1). Slack attaches through the OAuth connector; LINE / Telegram /
  WhatsApp bots attach once with an admin link code (like Discord's `/ตั้งค่า`).
- A platform tenant that belongs to no org keeps working alone (today's behaviour).

### Verified identities: API and flow (step 3, done)

Code: `salid/identity_links.py` (codes, redeem, `identity_of`, `auto_link`), `orgs.Member.identities`
+ `OrgStore.member_by_identity / link_identity / unlink_identity / drop_platform_identities`,
`platforms/core.py` (`Assistant.identity_of(msg) -> (org_id, member_email) | None`, link command,
Slack auto verification), `platforms/slack.py` (`SlackPort.profile_email`).

Routes (all under `/api/center`, Bearer Google ID token like the rest):

- `POST /orgs/{org}/identities/{platform}/code` (any member; platform `slack|line|telegram|whatsapp`)
  -> `{code: "1234-5678", platform, kind: "identity", expires_at, command: "link 1234-5678"}`.
  The button shows the code, the expiry and the command to send to Salid in a PRIVATE chat.
  403 not a member (a super admin who is not a member too), 400 unknown platform / discord, 429 when
  5 unused codes are live (wait up to 15 minutes).
- `GET /orgs/{org}/identities` -> `[{platform, account_hint}]`, the caller's own linked accounts
  (`account_hint` = last 4 characters; Discord is listed read-only). Poll it after showing a code
  (or on focus) to flip the button to "Linked".
- `DELETE /orgs/{org}/identities/{platform}` -> the updated list; 404 when nothing is linked.
- `POST /orgs/{org}/workspaces/{platform}/attach-code` (owner / admin / super) -> same shape with
  `kind: "attach"`. The admin sends `link 1234-5678` to the BOT in a private chat; the bot tenant
  (LINE channel, Telegram bot, WhatsApp number, Slack team) becomes an `orgs.Workspace`
  (`id = <platform>-<tenant>`, `external_id` = tenant key) and then shows up in
  `GET /orgs/{org}/workspaces`. Refused (generic "invalid or expired" in the chat) when that bot
  belongs to another org.

Chat side, same for both kinds: only in a 1:1 (a group gets "please send the code in a private
chat" and the code is not used); fixed Thai + English reply; the message never reaches the model,
the memory log or the room buffer; 5 attempts per person and 30 per bot in 10 minutes. A member
code from a bot attached to ANOTHER org is refused; from a bot attached to no org it links the
account (usable once the admin attaches the bot). Slack needs no code when `users:read.email` is
granted (see `docs/platforms/slack.md`).

Decisions: one account id per platform per member (a second Slack workspace in the same org uses
the same slot, so the first link wins and is not replaced automatically); `identity_of` requires
the sender's tenant to belong to the org that holds the link, so an id linked in org A never counts
in org B's bot; guests (Slack restricted accounts) are never auto-verified; removing the last
workspace of a platform from an org, or uninstalling Slack, forgets that platform's links.
The OAuth connector's `SCOPES` (`salid/connectors/slack.py`) still lacks `users:read.email`; add it
if the OAuth-installed app (not only the Socket Mode app) must auto-verify.

### Scope across platforms (the security rule, unchanged in spirit)

The answer may use a source only if **the asker may read it and so can everyone who reads the
answer**. Across platforms this means:

- **1:1 with a verified member** (Discord DM / private channel, Slack DM, LINE / Telegram /
  WhatsApp 1:1): audience = that person, so Salid may use everything that member may read in every
  workspace of the org (per-platform readability computed with that platform's `Workspace` for the
  member's identity on it). This is where cross-platform questions are answered.
- **A room with an exact audience** (Discord channel with complete member cache, Slack private
  non-shared channel): sources from other workspaces only if every reader of the room is a verified
  member who may read them.
- **A room with an unknown audience** (LINE / Telegram groups, Slack public or shared channels,
  any room whose readers cannot be proven): that room's own memory only, never another platform's.
  Salid may offer "ask me in a private chat" instead.
- Ledger, tasks, projects, people, wiki: stored per workspace today; org-wide views query the
  org's `memory_id`s and filter by the same per-member permissions (non-executives see their own
  receipts across all their linked accounts; executives = org owner/admin role or the platform's
  executive rule).
- Platform data policies still apply per platform (LINE: friend/group information must not be kept
  more than 24 h; see `docs/platforms/line.md`). Retention "forever" therefore does not apply to
  LINE rooms until that is settled.

### Google services (Calendar, Meet)

- Google login (Firebase Auth) only proves who the member is; it gives no Calendar access.
  Calendar needs a separate "Connect Google Calendar" consent on the web (OAuth web server flow,
  `access_type=offline`, scope `https://www.googleapis.com/auth/calendar.events`), storing the
  member's refresh token encrypted, revocable from the web, never logged.
- `scheduling.GoogleCalendar` (a `CalendarBackend`): create / move / cancel events in the
  organiser's calendar, invite attendees by member email, `conferenceData.createRequest` for a
  Meet link; Salid's meeting record keeps the Google event id. Works from any platform; the
  existing confirm-before-acting rule stays.
- Free/busy (`freetime.py`) can use the same token (`freeBusy.query`).

## Build order

1. Unify the workspace id scheme. **Done** (#32).
2. `Workspace` protocol + `DiscordWorkspace` + `SlackWorkspace`. **Done** (#33, step 2a).
3. Verified identities: `Member.identities`, Slack email match, link codes for LINE / Telegram /
   WhatsApp (web button + 1:1 redeem), platform workspaces attached to the org. **Done** (API and
   bot side; the web buttons are for the web teammate, see "Verified identities: API" below).
4. Org-wide scope: in a 1:1 with a verified member, memory / ledger / tasks / projects / people
   across all the org's workspaces; rooms keep the fail-closed rule above.
5. `_agent_kwargs` takes a `Workspace` (step 2b, no behaviour change on Discord), then the first
   tool group on Slack and the other platforms: tasks + follow-ups + task cards, projects and
   progress pictures, people / handover, briefs and reports, reminders, documents (PDF / decks /
   quotations), deliver-file, profiles, catch-up, wiki.
6. Google Calendar + Meet: "Connect Google Calendar" consent, `GoogleCalendar` backend, meetings
   and reminders from any platform.
7. Buttons on Slack (Block Kit): task cards, approvals, votes, polls, video-ad cancel.
8. Parity test and a capability table in this file, updated with every feature.

## Not on other platforms (by design)

- Voice meetings, live captions, voice actions: Discord only (no bot audio API on Slack Huddles,
  LINE, Telegram groups, WhatsApp).
- Role hierarchy (COO disclosure by rank): Discord roles; Slack has admins/owners only, so rank
  there is admin > member until org roles come from the command center.
- History import: Slack yes (rate limits), LINE / Telegram / WhatsApp no (platform has no history).

## Open questions

- Objectives / goals on tasks: a field on `Task` or project-level OKRs (several tasks under one
  objective)? Projects already have milestones and progress; prefer extending projects.
- Which org roles from the command center map to Slack ranks.
