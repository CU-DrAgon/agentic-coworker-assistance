"""Link a Discord server the Salid bot is already in to an organization, without the bot running.

This does by hand what the `/ตั้งค่า` link flow does once the bot is online: writes the guild record
(name, icon, text channels; the bot corrects the visibility flags when it starts), makes the owner a
hub admin of that server and attaches the server as a Discord workspace of the organization.

    SALID_STORE=firestore uv run python scripts/salid_link_guild.py --guild 123 --org org_abc --email you@x.com

The bot token is read from `.env` (SALID_DISCORD_TOKEN) in-process and never printed. Writes Firestore:
run only on purpose. `--dry-run` prints what would be written.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

API = "https://discord.com/api/v10"


def _token() -> str:
    tok = os.environ.get("SALID_DISCORD_TOKEN", "")
    env = Path(".env")
    if not tok and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("SALID_DISCORD_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        raise SystemExit("SALID_DISCORD_TOKEN not found (env or .env)")
    return tok


def _get(tok: str, path: str):
    req = urllib.request.Request(
        API + path, headers={"Authorization": "Bot " + tok, "User-Agent": "salid-link/1.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--guild", type=int, required=True, help="Discord server (guild) id")
    ap.add_argument("--org", required=True, help="organization id, e.g. org_aef43a633f")
    ap.add_argument("--email", required=True, help="owner's Google email (becomes hub admin)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    email = a.email.lower()

    tok = _token()
    g = _get(tok, f"/guilds/{a.guild}")
    chans = [c for c in _get(tok, f"/guilds/{a.guild}/channels") if c.get("type") == 0]
    icon = f"https://cdn.discordapp.com/icons/{a.guild}/{g['icon']}.png" if g.get("icon") else ""
    print(f"server: {g['name']} ({a.guild}) | text channels: {[c['name'] for c in chans]}")

    from meeting_scribe import center, hub
    from salid.orgs import Workspace, stamp

    guilds, orgs = hub.get_guild_store(), center.get_orgs()
    existing = orgs.org_of_guild(a.guild)
    if existing is not None and existing.id != a.org:
        raise SystemExit(f"server already belongs to organization {existing.id} ({existing.name})")
    ws = Workspace(
        id=f"discord-{a.guild}",
        platform="discord",
        external_id=str(a.guild),
        name=g["name"],
        icon_url=icon,
        status="connected",
        connected_at=stamp(),
        connected_by=email,
    )
    if a.dry_run:
        print(
            "dry run: would write guild record, add admin", email, "and attach", ws.id, "to", a.org
        )
        return

    guilds.upsert_guild(
        a.guild,
        name=g["name"],
        icon_url=icon,
        installed=True,
        active=True,
        channels=[
            {"id": c["id"], "name": c["name"], "visible": True, "readable_history": True}
            for c in chans
        ][:60],
    )
    ref = guilds.client.collection(guilds.collection).document(str(a.guild))
    admins = list((ref.get().to_dict() or {}).get("admin_emails", []))
    if email not in admins:
        admins.append(email)
    ref.set({"admin_emails": admins}, merge=True)
    org = orgs.attach_workspace(a.org, ws)
    print(f"linked: {org.name} <-> {g['name']}")
    print("workspaces:", [(w.platform, w.name, w.status) for w in org.workspaces])
    print("hub admins of the server:", guilds.get(a.guild).admin_emails)


if __name__ == "__main__":
    main()
