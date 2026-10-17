"""Agent tools of the data studio (logic in `data_studio.py`) and their Discord glue.

`data_tools(ctx)` returns the plain functions `_agent_kwargs` appends to `extra_tools` through
`DataMixin._data_tools`, so chat and voice share them. Guild, asker, channel, the attached files and
who may read which saved dataset are bound here, never model input. Files (charts, exports) leave
through the `outbox` attribute (see `qa.tool_files`). Nothing the user's data holds is logged.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from . import data_studio as ds
from .data_studio import DataError, Dataset, DatasetRepo

logger = logging.getLogger(__name__)
MODEL_ROWS = 100  # rows of a result handed back to the model (the Discord table shows 25)


@dataclass
class Loaded:
    dataset: Dataset
    prof: dict
    name: str
    rec: dict | None = None  # the saved record, None while only in this request


@dataclass
class DataCtx:
    guild_id: int
    asker_id: int
    asker_name: str
    is_exec: bool
    channel_id: int  # where a dataset saved now belongs (0 = private: DM / private chat)
    lang: str
    private: bool  # a DM / private chat: nobody else reads the answer
    files: list  # FileInput of the message (raw bytes for xlsx / xls / csv / tsv / json)
    repo: DatasetRepo
    readable: Callable[[int], bool] = lambda cid: False  # may the asker + audience read channel cid
    font: str = ""
    now: Callable[[], datetime] = datetime.now
    outbox: list[tuple[str, bytes]] = field(default_factory=list)
    loaded: dict[str, Loaded] = field(default_factory=dict)  # per request
    current: str = ""

    # ---- helpers
    def run(self, fn: Callable[[], dict]) -> dict:
        try:
            return fn()
        except DataError as e:
            return {"status": "error", "error": str(e)}
        except Exception:
            logger.exception("data tool failed")
            return {"status": "error", "error": "the data tool failed; try again"}

    def _visible(self) -> list[dict]:
        return [
            r
            for r in self.repo.all(self.guild_id)
            if ds.can_read(r, self.asker_id, self.readable, self.private)
        ]

    def _new_id(self) -> str:
        while True:
            did = "ds-" + secrets.token_hex(3)
            if did not in self.loaded and self.repo.get(self.guild_id, did) is None:
                return did

    def _adopt(self, did: str, dset: Dataset, name: str, rec: dict | None = None) -> Loaded:
        item = Loaded(dset, ds.profile(dset, self.lang), name, rec)
        self.loaded[did] = item
        self.current = did
        return item

    def _from_saved(self, rec: dict) -> str:
        if rec["id"] not in self.loaded:
            self._adopt(rec["id"], self.repo.load(self.guild_id, rec), rec["name"], rec)
        self.current = rec["id"]
        return rec["id"]

    def pick(self, text: str = "") -> tuple[str, Loaded]:
        """The dataset a tool works on: the one named (id or name fragment; this request's first,
        then saved ones the asker may read), else the one loaded last in this request."""
        text = (text or "").strip()
        if not text:
            if self.current in self.loaded:
                return self.current, self.loaded[self.current]
            raise DataError("no dataset is loaded yet; call data_load first")
        low = text.casefold()
        hits = [
            k
            for k, v in self.loaded.items()
            if k == low or low in v.name.casefold() or low in v.dataset.filename.casefold()
        ]
        if len(hits) == 1:
            self.current = hits[0]
            return hits[0], self.loaded[hits[0]]
        saved = [r for r in self._visible() if r["id"] == low or low in r["name"].casefold() or low in r.get("filename", "").casefold()]  # fmt: skip
        if len(hits) == 0 and len(saved) == 1:
            did = self._from_saved(saved[0])
            return did, self.loaded[did]
        names = [self.loaded[k].name for k in hits] + [r["name"] for r in saved]
        if not names:
            raise DataError(f"no dataset matches '{text}'; data_list shows the saved ones")
        raise DataError("several datasets match: " + ", ".join(names[:8]) + "; use the id")

    # ---- tools
    def data_load(self, file_index: int = -1, dataset: str = "") -> dict:
        if (dataset or "").strip():
            did, item = self.pick(dataset)
        else:
            cands = [f for f in self.files if getattr(f, "raw", b"") and not f.skip]
            if file_index != -1:
                cands = [f for f in cands if f.index == file_index]
            if not cands:
                raise DataError(
                    "no spreadsheet / CSV / JSON file is attached to this message"
                    if file_index == -1
                    else f"attachment #{file_index} is not a readable data file"
                    "; use data_list for saved datasets"
                )
            f = cands[-1]
            dset = ds.load_tables(f.raw, f.filename)
            did = self._new_id()
            item = self._adopt(did, dset, f.filename)
        tables = [
            {
                "name": t.name,
                "label": t.label,
                "rows": t.rows,
                "columns": [
                    {"name": c.name, "label": c.label, "type": c.type + (" %" if c.unit else "")}
                    for c in t.columns
                ],
            }
            for t in item.dataset.tables
        ]
        sample = {tp["name"]: tp["sample"][:3] for tp in item.prof["tables"][:3]}
        return {
            "status": "ok",
            "dataset_id": did,
            "name": item.name,
            "saved": item.rec is not None,
            "tables": tables,
            "notes": item.prof["notes"],
            "issues": {tp["name"]: tp["issues"] for tp in item.prof["tables"] if tp["issues"]},
            "sample": sample,
            "profile_md": ds.profile_md(item.dataset, self.lang, item.prof),
        }

    def _query(self, query: str, dataset: str, limit: int = ds.SQL_LIMIT) -> tuple[str, ds.Result]:
        did, item = self.pick(dataset)
        return did, ds.run_sql(item.dataset, query, limit, lang=self.lang)

    def data_sql(self, query: str, dataset: str = "") -> dict:
        _did, res = self._query(query, dataset)
        return {
            "status": "ok",
            "columns": res.columns,
            "row_count": len(res.rows),
            "rows": res.json_rows()[:MODEL_ROWS],
            "truncated": res.truncated or len(res.rows) > MODEL_ROWS,
            "result_md": ds.result_md(res, self.lang),
        }

    def data_chart(
        self,
        query: str,
        kind: str = "bar",
        x: str = "",
        y: list[str] | None = None,
        title: str = "",
        dataset: str = "",
    ) -> dict:
        _did, res = self._query(query, dataset)
        png = ds.chart_png(res, kind, x or None, y, title, self.lang, configured_font=self.font)
        name = f"chart-{_ascii(title or kind)}.png"
        self.outbox.append((name, png))
        return {
            "status": "ok",
            "attached": True,
            "note": f"{kind} chart of {len(res.rows)} rows attached as {name}",
            "rows": len(res.rows),
        }

    def data_export(self, query: str, file_format: str = "xlsx", name: str = "") -> dict:
        _did, res = self._query(query, "", ds.EXPORT_LIMIT)
        fmt = (file_format or "xlsx").lower().lstrip(".")
        base = _ascii(name or "result")
        if fmt == "csv":
            fname, data = f"{base}.csv", ds.result_csv(res)
        else:
            fname, data = f"{base}.xlsx", ds.result_xlsx(res)
        self.outbox.append((fname, data))
        return {
            "status": "ok",
            "attached": True,
            "note": f"{len(res.rows)} rows attached as {fname}",
            "truncated": res.truncated,
        }

    def data_save(self, name: str = "", dataset: str = "") -> dict:
        did, item = self.pick(dataset)
        if item.rec is not None:
            return {"status": "ok", "dataset_id": did, "note": "already saved"}
        rec = self.repo.save(
            self.guild_id,
            item.dataset,
            name=name or item.name,
            owner_id=self.asker_id,
            owner_name=self.asker_name,
            channel_id=self.channel_id,
            prof=item.prof,
            did=did,
        )
        item.rec, item.name = rec, rec["name"]
        return {
            "status": "ok",
            "dataset_id": rec["id"],
            "name": rec["name"],
            "visible_to": "owner only"
            if not self.channel_id
            else "people who can read this channel",  # fmt: skip
        }

    def data_list(self) -> dict:
        rows = [
            {
                "dataset_id": r["id"],
                "name": r["name"],
                "file": r.get("filename", ""),
                "owner": r.get("owner_name", ""),
                "created": r.get("created_at", "")[:10],
                "tables": [f"{m['name']} ({m['rows']} rows)" for m in r["tables"]],
            }
            for r in self._visible()
        ]
        return {"status": "ok", "datasets": rows}

    def data_forget(self, dataset: str) -> dict:
        low = (dataset or "").strip().casefold()
        recs = [r for r in self._visible() if low and (r["id"] == low or low in r["name"].casefold())]  # fmt: skip
        if len(recs) != 1:
            raise DataError("name one saved dataset (see data_list)" if not recs else "several datasets match; use the id")  # fmt: skip
        rec = recs[0]
        if rec["owner_id"] != self.asker_id and not self.is_exec:
            raise DataError("only the owner or an executive can delete this dataset")
        self.repo.delete(self.guild_id, rec)
        self.loaded.pop(rec["id"], None)
        if self.current == rec["id"]:
            self.current = ""
        return {"status": "ok", "deleted": rec["name"]}


def _ascii(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", ds.slug(text, "data").replace("_", "-")).strip("-")[:40] or "data"  # fmt: skip


def data_tools(ctx: DataCtx) -> list[Callable]:
    """The functions `_agent_kwargs` appends to `extra_tools` (docstrings = tool descriptions)."""

    def data_load(file_index: int = -1, dataset: str = "") -> dict:
        """Open a spreadsheet / CSV / JSON for analysis and get its profile: tables (one per sheet),
        columns with types, empty %, ranges, top values, sample rows and likely problems (total
        rows, mixed types). ALWAYS call this first and read the profile before saying anything
        about the data. Thai dates in Buddhist years are converted, "1,234" and "฿1,200" and "12%"
        become numbers. Use the returned table and column NAMES in data_sql.

        Args:
            file_index: the attachment number (#N shown on the file); -1 = the last data file of
                this message.
            dataset: instead of a file: id or name fragment of a saved dataset (see data_list).
        """
        return ctx.run(lambda: ctx.data_load(file_index, dataset))

    def data_sql(query: str, dataset: str = "") -> dict:
        """Run ONE read-only SQL SELECT (DuckDB dialect; WITH allowed) over the loaded tables and
        get the rows. Use it for every number you report: sums, counts, averages, group by, top N,
        growth, joins between sheets. Table and column names come from data_load. Text columns are
        VARCHAR; dates are DATE / TIMESTAMP. At most 200 rows; aggregate instead of listing. An
        error message lists the real column names: fix the query and retry.

        Args:
            query: the SELECT statement, no semicolon, no comments needed.
            dataset: id or name fragment; empty = the dataset loaded last.
        """
        return ctx.run(lambda: ctx.data_sql(query, dataset))

    def data_chart(
        query: str,
        kind: str = "bar",
        x: str = "",
        y: list[str] | None = None,
        title: str = "",
        dataset: str = "",
    ) -> dict:
        """Run a SELECT and attach the result as a PNG chart (the image is attached to your reply
        automatically; say what it shows in one or two lines). Use for trends over time (line),
        comparisons between categories (bar), shares of a whole (pie, at most about 8 slices),
        relationships (scatter) and distributions (hist). Never for a single number. The query
        should already be aggregated and sorted.

        Args:
            query: the SELECT that produces the data to plot.
            kind: bar, line, pie, scatter or hist.
            x: result column for the x axis / categories (empty = the first column).
            y: result column names with the values (empty = the numeric columns).
            title: a short chart title in the user's language.
            dataset: id or name fragment; empty = the dataset loaded last.
        """
        return ctx.run(lambda: ctx.data_chart(query, kind, x, y, title, dataset))

    def data_export(query: str, file_format: str = "xlsx", name: str = "") -> dict:
        """Run a SELECT over the loaded dataset and attach the full result (up to 50,000 rows) as an
        Excel or CSV file. Use when the user wants the table, the cleaned data or the numbers as a
        file.

        Args:
            query: the SELECT statement.
            file_format: "xlsx" or "csv".
            name: file name without extension (English letters), empty = result.
        """
        return ctx.run(lambda: ctx.data_export(query, file_format, name))

    def data_save(name: str = "", dataset: str = "") -> dict:
        """Keep the loaded dataset so the team can ask more questions about it later (even without
        the file). Offer it when the person is likely to come back; save only when they agree. A
        dataset posted in a channel is then readable by people who can read that channel; from a DM
        or private chat only by the owner. Over 3 MB compressed it cannot be kept.

        Args:
            name: a short name to find it by, e.g. "Q3 sales"; empty = the file name.
            dataset: id of the loaded dataset; empty = the one loaded last.
        """
        return ctx.run(lambda: ctx.data_save(name, dataset))

    def data_list() -> dict:
        """List the saved datasets the asker may open (id, name, file, owner, date, tables)."""
        return ctx.run(ctx.data_list)

    def data_forget(dataset: str) -> dict:
        """Delete a saved dataset. Only its owner or an executive can; ask the person to confirm
        first.

        Args:
            dataset: id or name fragment of the saved dataset.
        """
        return ctx.run(lambda: ctx.data_forget(dataset))

    tools: list[Callable] = [
        data_load, data_sql, data_chart, data_export, data_save, data_list, data_forget
    ]  # fmt: skip
    for fn in tools:
        fn.outbox = ctx.outbox  # type: ignore[attr-defined]
    return tools


class DataMixin:
    def _data_tools(self, message, guild, asker, files, lang: str, is_exec: bool = False) -> list:
        """The data tools for one request (chat or voice). [] when the flag is off; never raises."""
        try:
            if not self.team.enabled(guild.id, "data"):
                return []
            private = self._data_private_here(message, guild, asker)
            ctx = DataCtx(
                guild_id=guild.id,
                asker_id=asker.id,
                asker_name=getattr(asker, "display_name", str(asker.id)),
                is_exec=is_exec,
                channel_id=0 if private else getattr(message.channel, "id", 0),
                lang=lang,
                private=private,
                files=list(files or []),
                repo=DatasetRepo(self.docs, clock=lambda: self.s.now()),
                readable=lambda cid: self._data_channel_readable(message, guild, asker, cid),
                font=getattr(self.s, "pdf_font", "") or "",
                now=self.s.now,
            )
            return data_tools(ctx)
        except Exception:
            logger.warning("data tools unavailable", exc_info=True)
            return []

    def _data_private_here(self, message, guild, asker) -> bool:
        """A DM or the asker's private Salid channel: nobody else reads the answer."""
        if getattr(message, "room", None) is not None:
            return False
        if getattr(message, "guild", None) is None:
            return True
        try:
            return self.private_chats.owner_of(guild.id, message.channel.id) == asker.id
        except Exception:
            return False

    def _data_channel_readable(self, message, guild, asker, cid: int) -> bool:
        """The asker, and everybody who will read this answer, can read channel `cid`."""
        try:
            src = guild.get_channel(cid)
            if src is None:
                return False
            viewers = self._channel_viewers(guild, src)
            if asker.id not in viewers:
                return False
            room = getattr(message, "room", None)
            if room is not None:
                audience = set(room)
            elif self._data_private_here(message, guild, asker):
                audience = {asker.id}
            elif message.channel.id == cid:
                return True
            else:
                audience = set(self._channel_viewers(guild, message.channel, fresh=True))
            return audience <= viewers
        except Exception:
            logger.warning("data channel check failed", exc_info=True)
            return False
