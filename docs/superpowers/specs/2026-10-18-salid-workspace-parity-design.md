# Salid on every workspace: one brain, thin platform adapters

Status: agreed direction, not built yet (2026-10-18). Read this before adding a Salid feature.

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

## Build order

1. Unify the workspace id scheme (above).
2. `Workspace` protocol + `DiscordWorkspace`; move `_agent_kwargs` to take a `Workspace`.
   No behaviour change; the existing Discord tests are the safety net.
3. `SlackWorkspace` and the first tool group on Slack: tasks + follow-ups + task cards, projects
   and progress pictures, briefs and reports, reminders, documents (PDF / decks / quotations),
   deliver-file, profiles, catch-up, wiki.
4. Meetings on Slack: Salid store + reminders, then `GoogleCalendar` behind the Google OAuth link.
5. Buttons on Slack (Block Kit): task cards, approvals, votes, polls, video-ad cancel.
6. Parity test and a capability table in this file, updated with every feature.

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
