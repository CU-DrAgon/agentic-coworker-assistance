"""ClickUp: OAuth + read-only task import. Enabled only with `CLICKUP_CLIENT_ID` / `CLICKUP_CLIENT_SECRET`.

Nothing is written back to ClickUp. A ClickUp "Workspace" (API: team) becomes one Salid workspace;
its tasks are fetched live for the command center's tasks view.
"""

from __future__ import annotations

from urllib.parse import urlencode

import httpx

from .base import Connection, ConnectorConfig, ConnectorError, NotConfigured, http_client

AUTHORIZE = "https://app.clickup.com/api"
API = "https://api.clickup.com/api/v2"


class ClickUpConnector:
    platform = "clickup"

    def __init__(self, cfg: ConnectorConfig, client=None):
        self.cfg = cfg
        self._client = client

    @property
    def http(self) -> httpx.Client:
        if self._client is None:
            self._client = http_client()
        return self._client

    def enabled(self) -> bool:
        return bool(self.cfg.clickup_client_id and self.cfg.clickup_client_secret)

    def _need(self) -> None:
        if not self.enabled():
            raise NotConfigured("ClickUp is not configured")

    def oauth_start(self, org_id: str, state: str = "") -> str:
        self._need()
        q = {
            "client_id": self.cfg.clickup_client_id,
            "redirect_uri": self.cfg.redirect_uri("clickup"),
            "state": state or org_id,
        }
        return f"{AUTHORIZE}?{urlencode(q)}"

    def oauth_callback(self, params: dict[str, str]) -> list[Connection]:
        self._need()
        if params.get("error") or not params.get("code"):
            raise ConnectorError("ClickUp authorization was cancelled")
        try:
            r = self.http.post(
                f"{API}/oauth/token",
                json={
                    "client_id": self.cfg.clickup_client_id,
                    "client_secret": self.cfg.clickup_client_secret,
                    "code": params["code"],
                },
            )
            token = str(r.json().get("access_token") or "")
            if not token:
                raise ConnectorError("ClickUp refused the authorization")
            teams = self.http.get(f"{API}/team", headers={"Authorization": token}).json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ConnectorError("Could not reach ClickUp") from exc
        out = [
            Connection("clickup", str(t["id"]), str(t.get("name") or t["id"]), token=token)
            for t in teams.get("teams", [])
        ]
        if not out:
            raise ConnectorError("This ClickUp account has no workspace")
        return out

    def list_channels(self, token: str, external_id: str) -> list[dict]:
        """Spaces of the team (ClickUp's top-level containers)."""
        r = self.http.get(f"{API}/team/{external_id}/space", headers={"Authorization": token})
        return [{"id": s["id"], "name": s.get("name", "")} for s in r.json().get("spaces", [])]

    def tasks(self, token: str, external_id: str, limit: int = 100) -> list[dict]:
        """Open tasks of the team, read-only."""
        r = self.http.get(
            f"{API}/team/{external_id}/task",
            params={"page": 0, "include_closed": "false", "subtasks": "true"},
            headers={"Authorization": token},
        )
        if r.status_code >= 400:
            raise ConnectorError("ClickUp refused the task list")
        out = []
        for t in r.json().get("tasks", [])[:limit]:
            due = t.get("due_date")
            out.append(
                {
                    "id": str(t.get("id", "")),
                    "title": t.get("name", ""),
                    "status": (t.get("status") or {}).get("status", ""),
                    "assignees": [
                        a.get("username") or a.get("email", "") for a in t.get("assignees", [])
                    ],
                    "due": _ms_date(due),
                    "url": t.get("url", ""),
                }
            )
        return out

    def ingest(self, event: dict) -> int:
        return 0  # read-only: tasks are fetched live, nothing is stored

    def send(self, token: str, channel: str, text: str) -> None:
        raise ConnectorError("ClickUp is read-only")


def _ms_date(ms) -> str:
    from datetime import UTC, datetime

    try:
        return datetime.fromtimestamp(int(ms) / 1000, UTC).date().isoformat()
    except (TypeError, ValueError):
        return ""
