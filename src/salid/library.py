"""Documents library of the command center: one list over everything Salid has stored for an
organization, by category, plus web uploads.

Categories: `receipts` (ledger), `contracts_docs` (files Salid read in chat: PDF / Office / image
notes in the memory log), `minutes` (voice meeting minutes), `reports` (PDF reports Salid made),
`wiki` (knowledge base pages), `ideas` (idea / project documents, salid.ideas), `contests` (salid.contests) and `uploads`
(files uploaded on the web).

Who sees what (decided here in code):

* org owner / admin (and super admins): every item of every connected workspace, except private
  DM notes, which never leave Discord, and ideas marked private by their owner;
* plain member: web uploads and items whose source channel is public in its workspace
  (`public_channel_ids`, see salid.channel_access); never the ledger; ideas only when the owner
  shared them with the team.

Web uploads are stored in GCS (`SALID_RECEIPT_BUCKET`, prefix `salid-docs/<org>/`) when the bucket
is set, else under `SALID_DOCS_LOCAL_DIR`; read, categorised and summarised by Flash-Lite, kept in
the DocStore (`salid_org_docs`, never touched by chat-memory retention) and indexed as a file note
in the memory store under the org's web workspace (channel `WEB_DOCS_CHANNEL`) so Salid finds it.
A ShareLink "documents" from the web workspace to a Discord workspace copies the note into that
guild's memory under `SHARED_DOCS_CHANNEL` (the bot adds that channel to its readable set).
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from . import contests as contests_mod
from . import docs as docs_mod
from .coo_store import MINUTES
from .i18n import resolve_lang
from .ideas import IdeaRepo, library_visible, render_markdown
from .memory_store import MemoryMessage
from .orgs import Org, Workspace
from .sharing import (
    SHARED_DOCS_CHANNEL,
    WEB_DOCS_CHANNEL,
    page_is_public,
    sources_in_org,
)
from .wiki import WikiRepo

logger = logging.getLogger(__name__)

ORG_DOCS = "salid_org_docs"  # guild = web workspace memory id, key = doc id
REPORTS = "salid_reports"  # guild = workspace, key = id; {title, at, uri, by, channel_id}
CATEGORIES = (
    "receipts",
    "contracts_docs",
    "minutes",
    "reports",
    "wiki",
    "ideas",
    "contests",
    "uploads",
)
# ShareLink category -> library categories it releases
SHARE_MAP = {
    "documents": ("contracts_docs", "reports", "uploads"),
    "wiki": ("wiki",),
    "minutes": ("minutes",),
    "ledger": ("receipts",),
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PREVIEW = 300
STORED_TEXT = 20000
LOOKBACK_DAYS = 365
NOTE_RE = re.compile(r"^\[(PDF|DOCX|XLSX|PPTX|DOC|XLS|ข้อความ|รูป)\s*:?\s*([^\]]*)\]\s*(.*)$")
IMAGE_EXT = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


# ------------------------------------------------------------------ model
class DocItem(BaseModel):
    id: str
    title: str
    category: str
    workspace: str  # workspace id inside the org
    workspace_name: str = ""
    platform: str = ""
    date: str = ""
    source_url: str = ""
    who: str = ""
    size: int | None = None
    preview: str = ""
    tags: list[str] = []
    shared_from: str | None = None  # workspace name when it came in through a ShareLink


class DocDetail(DocItem):
    text: str = ""
    extra: dict[str, Any] = {}


@dataclass
class Viewer:
    email: str
    exec: bool  # org owner / admin / super admin


@dataclass
class Page:
    items: list[DocItem]
    next_cursor: str | None
    total: int


@dataclass
class Gathered:
    items: dict[str, DocItem] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)  # id -> full text (detail view)


# ------------------------------------------------------------------ blobs
class LocalBlobs:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, org_id: str, doc_id: str, filename: str, data: bytes, mime: str) -> str:
        ext = _ext(filename)
        path = self.root / org_id / f"{doc_id}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def read(self, uri: str) -> bytes:
        path = Path(uri).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("not an upload path")
        return path.read_bytes()


class GcsBlobs:
    """`gs://{bucket}/{prefix}{org}/{doc id}.{ext}` in the receipt archive bucket."""

    def __init__(self, bucket: str, prefix: str = "salid-docs/", client: Any = None, project=None):
        self.bucket_name = bucket
        self.prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
        self._client, self._project = client, project

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self._project)
        return self._client

    def save(self, org_id: str, doc_id: str, filename: str, data: bytes, mime: str) -> str:
        name = f"{self.prefix}{org_id}/{doc_id}.{_ext(filename)}"
        self.client.bucket(self.bucket_name).blob(name).upload_from_string(data, content_type=mime)
        return f"gs://{self.bucket_name}/{name}"

    def read(self, uri: str) -> bytes:
        head = f"gs://{self.bucket_name}/{self.prefix}"
        if not uri.startswith(head):
            raise ValueError("uri is not in this library's bucket prefix")
        return (
            self.client.bucket(self.bucket_name)
            .blob(uri[len(f"gs://{self.bucket_name}/") :])
            .download_as_bytes()
        )


def make_blobs(settings):
    if settings.receipt_bucket:
        return GcsBlobs(settings.receipt_bucket, settings.docs_prefix, project=settings.gcp_project)
    return LocalBlobs(settings.docs_local_dir)


def _ext(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    return re.sub(r"[^a-z0-9]", "", ext)[:8] or "bin"


# ------------------------------------------------------------------ analysis (Flash-Lite)
def make_analyzer(client, model: str) -> Callable[[bytes, str, str], dict]:
    """(data, filename, mime) -> {tag, summary, text}. Raises when the file cannot be read."""

    def analyze(data: bytes, filename: str, mime: str) -> dict:
        from .image_memory import describe_image

        ext = _ext(filename)
        if mime.startswith("image/") or ext in IMAGE_EXT:
            note = describe_image(
                client, model, data, mime if mime.startswith("image/") else IMAGE_EXT[ext]
            )
            return {
                "tag": (note.type or "image")[:30],
                "summary": note.description.strip(),
                "text": note.text.strip(),
            }
        kind = docs_mod.doc_kind(filename, mime)
        if kind is None:
            raise docs_mod.DocUnreadable("unsupported document type")
        dt = docs_mod.extract_text(data, kind)
        note = None
        try:
            note = docs_mod.analyze(client, model, data, kind, dt)
        except Exception:
            logger.warning("upload analysis failed", exc_info=True)
        text = dt.text or (note.text if note else "")
        return {
            "tag": (note.doc_type if note else "other")[:30] or "other",
            "summary": " ".join(note.summary.split()) if note else " ".join(text.split())[:PREVIEW],
            "text": text,
        }

    return analyze


# ------------------------------------------------------------------ the library
class Library:
    def __init__(
        self,
        *,
        docs,  # DocStore
        ledger,  # salid Store
        memory,  # MemoryStore
        guilds,  # GuildStore (public channels, channel lists)
        settings,
        blobs=None,
        analyzer: Callable[[bytes, str, str], dict] | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.docs, self.ledger, self.memory, self.guilds = docs, ledger, memory, guilds
        self.s, self.blobs, self.analyzer = settings, blobs, analyzer
        self.wiki = WikiRepo(docs)
        self.ideas = IdeaRepo(docs)
        self.contests = contests_mod.ContestRepo(docs)
        self._now = now or (lambda: datetime.now(UTC))

    # ---- channels
    def _channel_sets(self, ws: Workspace) -> tuple[set[int], set[int]]:
        """(public channel ids, every known channel id) of a Discord workspace."""
        if ws.platform != "discord":
            return set(), set()
        g = self.guilds.find(ws.memory_id)
        if g is None:
            return set(), set()
        public = {int(x) for x in g.public_channel_ids}
        return public, public | {int(c.id) for c in g.channels}

    def _lang(self, gid) -> str:
        """The guild's reply language for the markdown rendered into the library."""
        try:
            g = self.guilds.find(gid)
            return resolve_lang(getattr(g, "language_setting", None))
        except Exception:  # noqa: BLE001 - a lookup failure must not break the listing
            return "en"

    # ---- gathering one workspace
    def _gather(self, ws: Workspace, viewer: Viewer, out: Gathered) -> None:
        public, known = self._channel_sets(ws)
        base = {"workspace": ws.id, "workspace_name": ws.name or ws.id, "platform": ws.platform}
        if ws.platform == "web":
            self._gather_uploads(ws, base, out)
        if ws.platform != "discord":
            return
        gid = ws.memory_id
        if viewer.exec:
            self._gather_receipts(gid, base, out)
        self._gather_notes(gid, base, viewer, public, known, out)
        self._gather_minutes(gid, base, viewer, public, out)
        self._gather_wiki(gid, base, viewer, public, out)
        self._gather_reports(gid, base, viewer, public, out)
        self._gather_ideas(gid, base, viewer, out)
        self._gather_contests(gid, base, viewer, out)

    def _gather_uploads(self, ws: Workspace, base: dict, out: Gathered) -> None:
        for d in self.docs.find(ORG_DOCS, ws.memory_id):
            item = DocItem(
                id=f"{ws.id}:upload:{d['id']}",
                title=d.get("title") or d.get("filename", ""),
                category="uploads",
                date=d.get("created_at", ""),
                who=d.get("by", ""),
                size=d.get("size"),
                preview=(d.get("summary") or "")[:PREVIEW],
                tags=[d["tag"]] if d.get("tag") else [],
                **base,
            )
            out.items[item.id] = item
            out.text[item.id] = d.get("text", "")

    def _gather_receipts(self, gid: int, base: dict, out: Gathered) -> None:
        for r in self.ledger.recent(gid, 300):
            link = (
                f"https://discord.com/channels/{gid}/{r.channel_id}/{r.message_id}"
                if r.channel_id and r.message_id
                else ""
            )
            total = "" if r.total is None else f"{r.total:,.2f} {r.currency}"
            item = DocItem(
                id=f"discord-{gid}:receipt:{r.id}",
                title=r.vendor or f"#{r.id}",
                category="receipts",
                date=r.effective_date,
                source_url=link,
                who=r.submitter_name,
                preview=" · ".join(x for x in (total, r.category, r.invoice_no or "") if x),
                **base,
            )
            out.items[item.id] = item
            out.text[item.id] = "\n".join(
                f"{it.name} x{it.qty or 1} = {it.amount}" for it in (r.items or [])
            )

    def _gather_notes(self, gid, base, viewer, public, known, out: Gathered) -> None:
        allowed = known if viewer.exec else public
        allowed = {c for c in allowed if c > 0}
        if not allowed:
            return
        today = self._now().astimezone(self.s.zone).date()
        rows = self.memory.messages(
            gid,
            allowed,
            (today - timedelta(days=LOOKBACK_DAYS)).isoformat(),
            today.isoformat(),
            limit=5000,
        )
        for m in rows:
            if m.channel_id <= 0 or m.channel_id in (WEB_DOCS_CHANNEL, SHARED_DOCS_CHANNEL):
                continue
            for i, line in enumerate(m.text.splitlines()):
                hit = NOTE_RE.match(line.strip())
                if not hit:
                    continue
                label, head, rest = hit.groups()
                title = (
                    head.split(",")[0].strip() if label != "รูป" else f"รูป: {head.strip()}"
                ) or label
                item = DocItem(
                    id=f"discord-{gid}:note:{m.message_id}-{i}",
                    title=title,
                    category="contracts_docs",
                    date=m.created_at.isoformat(timespec="seconds"),
                    source_url=m.jump_url,
                    who=m.author_name,
                    preview=" ".join(rest.split())[:PREVIEW],
                    tags=[label],
                    **base,
                )
                out.items[item.id] = item
                out.text[item.id] = line.strip()

    def _gather_minutes(self, gid, base, viewer, public, out: Gathered) -> None:
        for d in self.docs.find(MINUTES, gid):
            if not viewer.exec and int(d.get("voice_channel_id", 0)) not in public:
                continue
            ended = d.get("ended_at", "")
            item = DocItem(
                id=f"discord-{gid}:minutes:{ended}-{d.get('voice_channel_id', '')}",
                title=f"{d.get('voice_channel_name') or 'meeting'} {ended[:10]}".strip(),
                category="minutes",
                date=ended,
                who=f"{len(d.get('attendee_ids', []))}",
                size=len(d.get("summary", "")),
                preview=" ".join(d.get("summary", "").split())[:PREVIEW],
                **base,
            )
            out.items[item.id] = item
            out.text[item.id] = d.get("summary", "")

    def _gather_wiki(self, gid, base, viewer, public, out: Gathered) -> None:
        for p in self.wiki.list(gid):
            if any(c < 0 for c in p.channel_ids):  # built from a DM: private
                continue
            if not viewer.exec and not page_is_public(p.channel_ids, public):
                continue
            item = DocItem(
                id=f"discord-{gid}:wiki:{p.slug}",
                title=p.title,
                category="wiki",
                date=p.updated_at,
                who=p.author_name,
                size=len(p.body),
                preview=" ".join(p.body.split())[:PREVIEW],
                **base,
            )
            out.items[item.id] = item
            out.text[item.id] = p.body

    def _gather_ideas(self, gid, base, viewer, out: Gathered) -> None:
        for d in self.ideas.all(gid):
            if not library_visible(d, viewer.exec):
                continue
            md = render_markdown(d, self._lang(gid), owner_line=False)
            item = DocItem(
                id=f"discord-{gid}:idea:{d.key}",
                title=d.title,
                category="ideas",
                date=d.updated_at,
                who=d.owner_name,
                size=len(md),
                preview=" ".join(next(iter(s.body for _k, s in d.ordered()), "").split())[:PREVIEW],
                tags=[s.title for _k, s in d.ordered()][:6],
                **base,
            )
            out.items[item.id] = item
            out.text[item.id] = md

    def _gather_contests(self, gid, base, viewer, out: Gathered) -> None:
        for c in self.contests.all(gid):
            if not contests_mod.library_visible(c, viewer.exec):
                continue
            md = contests_mod.render_markdown(c, self._lang(gid))
            item = DocItem(
                id=f"discord-{gid}:contest:{c.id}",
                title=c.name,
                category="contests",
                date=c.updated_at,
                who=c.created_by_name,
                size=len(md),
                preview=" ".join(" · ".join([c.organizer, c.track, c.kind]).split())[:PREVIEW],
                tags=[d.label for d in c.deadlines][:6],
                **base,
            )
            out.items[item.id] = item
            out.text[item.id] = md

    def _gather_reports(self, gid, base, viewer, public, out: Gathered) -> None:
        for d in self.docs.find(REPORTS, gid):
            cid = int(d.get("channel_id", 0))
            if not viewer.exec and cid not in public:
                continue
            item = DocItem(
                id=f"discord-{gid}:report:{d.get('id', '')}",
                title=d.get("title", "report"),
                category="reports",
                date=d.get("at", ""),
                who=d.get("by", ""),
                source_url=d.get("url", ""),
                **base,
            )
            out.items[item.id] = item

    # ---- org-wide listing
    def gather(self, org: Org, viewer: Viewer, workspace: str | None = None) -> Gathered:
        out = Gathered()
        wanted = [w for w in org.workspaces if w.status == "connected"]
        if workspace:
            ws = org.workspace(workspace)
            if ws is None:
                return out
            self._gather(ws, viewer, out)
            self._gather_shared_into(org, ws, viewer, out)
            return out
        for ws in wanted:
            self._gather(ws, Viewer(viewer.email, viewer.exec), out)
        return out

    def _gather_shared_into(self, org: Org, dest: Workspace, viewer: Viewer, out: Gathered) -> None:
        """Items other workspaces release to `dest` (ShareLinks): public ones only; the ledger
        only for execs."""
        for share_cat, lib_cats in SHARE_MAP.items():
            if share_cat == "ledger" and not viewer.exec:
                continue
            for src in sources_in_org(org, dest.id, share_cat):
                ws = org.workspace(src.workspace_id)
                if ws is None:
                    continue
                tmp = Gathered()
                # shared items must have been visible to everyone in the source (public only);
                # the ledger has no audience, so it is shared to execs and is gathered as exec
                self._gather(ws, Viewer(viewer.email, share_cat == "ledger"), tmp)
                for iid, item in tmp.items.items():
                    if item.category in lib_cats and iid not in out.items:
                        out.items[iid] = item.model_copy(update={"shared_from": ws.name or ws.id})
                        out.text[iid] = tmp.text.get(iid, "")

    def list(
        self,
        org: Org,
        viewer: Viewer,
        *,
        category: str | None = None,
        workspace: str | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 30,
    ) -> Page:
        got = self.gather(org, viewer, workspace)
        items = sorted(got.items.values(), key=lambda i: (i.date, i.id), reverse=True)
        if category:
            items = [i for i in items if i.category == category]
        needle = (q or "").casefold().strip()
        if needle:
            items = [
                i
                for i in items
                if needle in f"{i.title} {i.preview} {i.who} {' '.join(i.tags)}".casefold()
                or needle in got.text.get(i.id, "").casefold()
            ]
        start = int(cursor) if cursor and cursor.isdigit() else 0
        limit = max(1, min(limit, 100))
        page = items[start : start + limit]
        nxt = str(start + limit) if start + limit < len(items) else None
        return Page(page, nxt, len(items))

    def counts(self, org: Org, viewer: Viewer) -> dict[str, int]:
        got = self.gather(org, viewer)
        counts = dict.fromkeys(CATEGORIES, 0)
        for i in got.items.values():
            counts[i.category] = counts.get(i.category, 0) + 1
        return counts

    def get(
        self, org: Org, viewer: Viewer, doc_id: str, workspace: str | None = None
    ) -> DocDetail | None:
        """One item the viewer may see (searched in the viewer's own list)."""
        ws_id = doc_id.split(":", 1)[0]
        if org.workspace(ws_id) is None and not doc_id.startswith("discord"):
            return None
        got = self.gather(org, viewer, workspace)
        if doc_id not in got.items:  # may be shared into another workspace
            for ws in org.workspaces:
                self._gather_shared_into(org, ws, viewer, got)
        item = got.items.get(doc_id)
        if item is None:
            return None
        return DocDetail(**item.model_dump(), text=got.text.get(doc_id, ""))

    # ---- web uploads
    def upload(self, org: Org, email: str, filename: str, data: bytes, mime: str) -> DocItem:
        web = org.web_workspace()
        if web is None:
            raise ValueError("organization has no web workspace")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("file too large")
        if not data:
            raise ValueError("empty file")
        if self.analyzer is None:
            raise RuntimeError("no analyzer configured")
        info = self.analyzer(data, filename, mime)  # raises DocUnreadable for unsupported files
        doc_id = uuid.uuid4().hex[:16]
        uri = self.blobs.save(org.id, doc_id, filename, data, mime)
        now = self._now()
        rec = {
            "id": doc_id,
            "filename": filename,
            "title": filename,
            "mime": mime,
            "size": len(data),
            "uri": uri,
            "tag": info.get("tag", ""),
            "summary": info.get("summary", ""),
            "text": (info.get("text") or "")[:STORED_TEXT],
            "created_at": now.isoformat(timespec="seconds"),
            "by": email,
        }
        self.docs.put(ORG_DOCS, web.memory_id, doc_id, rec)
        self._index(org, web, rec)
        return DocItem(
            id=f"{web.id}:upload:{doc_id}",
            title=filename,
            category="uploads",
            workspace=web.id,
            workspace_name=web.name,
            platform="web",
            date=rec["created_at"],
            who=email,
            size=len(data),
            preview=rec["summary"][:PREVIEW],
            tags=[rec["tag"]] if rec["tag"] else [],
        )

    def read_upload(self, org: Org, doc_id: str) -> tuple[dict, bytes] | None:
        web = org.web_workspace()
        key = doc_id.rsplit(":", 1)[-1]
        d = self.docs.get(ORG_DOCS, web.memory_id, key) if web else None
        if not d:
            return None
        return d, self.blobs.read(d["uri"])

    # ---- searchable by Salid
    @staticmethod
    def note_text(rec: dict) -> str:
        head = f"[ไฟล์: {rec['filename']}" + (f", {rec['tag']}" if rec.get("tag") else "") + "]"
        out = f"{head} {' '.join((rec.get('summary') or '').split())}"
        text = " ".join((rec.get("text") or "").split())[:1500]
        return (out + (f" | ข้อความสำคัญ: {text}" if text else ""))[:2600]

    def _message(self, guild_id: int, channel: int, rec: dict) -> MemoryMessage:
        at = datetime.fromisoformat(rec["created_at"])
        return MemoryMessage(
            guild_id=guild_id,
            channel_id=channel,
            message_id=int(rec["id"], 16) % 2**53,
            author_id=0,
            author_name=rec.get("by", ""),
            created_at=at.astimezone(UTC),
            local_date=at.astimezone(self.s.zone).date().isoformat(),
            text=self.note_text(rec),
            jump_url="",
            kind="chat",
        )

    def _index(self, org: Org, web: Workspace, rec: dict) -> None:
        self.memory.add_message(self._message(web.memory_id, WEB_DOCS_CHANNEL, rec))
        for ws in self.document_targets(org):
            self.memory.add_message(self._message(ws.memory_id, SHARED_DOCS_CHANNEL, rec))

    @staticmethod
    def document_targets(org: Org) -> list[Workspace]:
        """Discord workspaces that receive the web uploads (ShareLink "documents" from web)."""
        web = org.web_workspace()
        if web is None:
            return []
        return [
            w
            for w in org.workspaces
            if w.platform == "discord"
            and w.status == "connected"
            and any(s.workspace_id == web.id for s in sources_in_org(org, w.id, "documents"))
        ]

    def reconcile_shared_docs(self, org: Org) -> int:
        """Make each Discord workspace's copies match the ShareLinks (add where shared, remove
        where not). Also rebuilds the web workspace's own search notes. Returns notes written."""
        web = org.web_workspace()
        if web is None:
            return 0
        recs = self.docs.find(ORG_DOCS, web.memory_id)
        targets = {w.id for w in self.document_targets(org)}
        n = 0
        for rec in recs:
            self.memory.add_message(self._message(web.memory_id, WEB_DOCS_CHANNEL, rec))
            n += 1
        ids = [int(r["id"], 16) % 2**53 for r in recs]
        for ws in org.workspaces:
            if ws.platform != "discord":
                continue
            if ws.id in targets:
                for rec in recs:
                    self.memory.add_message(self._message(ws.memory_id, SHARED_DOCS_CHANNEL, rec))
                    n += 1
            elif ids:
                self.memory.delete_messages(ws.memory_id, [i for i in ids])
        return n
