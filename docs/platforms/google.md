# Google Calendar and Meet

Each member can connect their own Google Calendar to Salid. Salid then creates, moves and cancels
events in **that member's primary calendar**, invites attendees by email (Google sends the
invitation) and asks Google for a **Meet link**. Salid never uses a shared service account for
this: the event belongs to whoever asked.

Google login on the web (Firebase) only proves who a member is. Calendar needs its own consent,
given once per member with the **Connect Google Calendar** button in the command center.

Code: `salid/google_oauth.py` (consent flow, encrypted token store), `scheduling.GoogleCalendar`
(Calendar API v3 backend, free/busy), `salid/google_tools.py` (agent tools, not wired into a bot
yet), routes at the end of `meeting_scribe/center.py`.

## Google Cloud console (once)

1. **Enable the API**: APIs & Services > Library > *Google Calendar API* > Enable.
2. **OAuth consent screen** (APIs & Services > OAuth consent screen):
   - User type *External* (or *Internal* if everyone is in one Workspace domain).
   - Scopes to add: `openid`, `email` (non-sensitive) and
     `https://www.googleapis.com/auth/calendar.events` (**sensitive**: Google requires app
     verification before the general public can consent; see below).
   - While the app is in *Testing* only the listed **test users** (up to 100) can connect, and their
     refresh tokens expire after 7 days. Add every pilot member's Google account under
     *Test users*. Publish to *In production* and finish verification for real customers.
3. **Credentials** > Create credentials > OAuth client ID > type **Web application**:
   - Authorized redirect URI: `https://salid.frontier-tech.app/api/center/connectors/google/callback`
     (add `http://localhost:8080/api/center/connectors/google/callback` for local work).
   - Copy the client id and secret.
4. Generate the token key (once, keep it in Secret Manager): 
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`

## Environment

All four are needed; if any is missing the feature is **off** (the routes answer 503, the tools are
not offered).

| Variable | Meaning |
| --- | --- |
| `GOOGLE_OAUTH_CLIENT_ID` | Web client id from step 3 |
| `GOOGLE_OAUTH_CLIENT_SECRET` | its secret (Secret Manager; never logged) |
| `GOOGLE_OAUTH_REDIRECT_URI` | exactly the redirect URI registered above |
| `SALID_TOKEN_KEY` | Fernet key that encrypts the stored refresh tokens. **Losing or rotating it disconnects everybody** (they press Connect again). |

## What is stored, and what is not

- Only the **refresh token**, Fernet-encrypted, in the DocStore kind `salid_google_tokens`, key
  `{org_id}:{email}`. The org and email are also inside the ciphertext, so a copied record cannot be
  replayed for another member. Access tokens live in process memory until shortly before they
  expire. Nothing secret is logged or put into an error message (errors carry the HTTP status only).
- `invalid_grant` (the member revoked Salid in their Google account, or the Testing-mode 7 days
  passed) deletes the record: the member has to connect again, and the tools say so.
- The consent `state` is random, stored server-side with (org, member email, PKCE verifier, expiry
  10 minutes) and single use. PKCE (S256) is used as well as the client secret.
- After consent Salid asks Google who consented (userinfo) and refuses, and revokes the grant, if it
  is not the member's own email, or if the calendar scope was not granted.
- Members can disconnect from the web (`DELETE`), which revokes at Google and deletes the record.

## Behaviour

- Event: summary, description, start/end in `SALID_TZ`, attendees by email,
  `conferenceData.createRequest` (`hangoutsMeet`, fresh `requestId`) with `conferenceDataVersion=1`,
  `sendUpdates=all` on create, move and cancel. Salid marks its events and only cancels its own.
- Free/busy (`GoogleCalendar.free_busy`) uses the organiser's token; calendars that token cannot read
  come back as `None` ("unknown"), never as free.
- Agent tools (`google_tools.calendar_tools`): `schedule_google_meeting` and
  `cancel_google_meeting` use the secretary rule: the first call returns a draft, and only the same
  call repeated with `confirm=true` in a **later** message acts. Attendees must be members of the
  asker's organization (email, or the unique start of an email's name). An outside email is accepted
  only from an executive (owner, admin, super admin). Without a connection the tool answers
  `not_connected` and says to press Connect.

## API for the web (all under `/api/center`, Bearer Google ID token)

- `GET  /orgs/{org}/google/calendar` -> `{connected, email, connected_at}` (the caller's own).
- `POST /orgs/{org}/google/calendar/connect` -> `{auth_url}`; navigate the browser there.
- Google redirects the browser to `GET /connectors/google/callback`, which answers a 303 to
  `/center?calendar=connected&org=<org>` or `/center?calendar=error&reason=<code>`; reasons:
  `denied`, `bad_state`, `expired`, `email_mismatch`, `scope_missing`, `no_refresh_token`,
  `exchange_failed`, `forbidden`, `not_available`, `http_<status>`, `network`.
- `DELETE /orgs/{org}/google/calendar` -> `{connected: false, ...}` (revokes and deletes).
- 403 for anyone who is not a **member** of that org (a super admin who is not a member too);
  503 when the feature is not configured. Show the card as "not configured" on 503.
