"""Meeting minutes document with a speaker-by-speaker transcript (meeting_doc.py + glue)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from salid import meeting_doc as md
from salid.meeting_doc_bot import MeetingDocMixin
from salid.pdf_report import find_fonts

T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)


def at(sec: int) -> datetime:
    return T0 + timedelta(seconds=sec)


LINES = [
    (at(0), "แอน", "สวัสดีทุกคน"),
    (at(20), "แอน", "เรามาเริ่มกันเลย"),
    (at(40), "บอย", "ผมส่งรายงาน 3 ฉบับวันศุกร์"),
    (at(50), md.SALID_SPEAKER, "รับทราบค่ะ"),
    (at(200), "แอน", "ขอบคุณค่ะ"),
    (at(230), "แอน", "แล้วเจอกัน"),
    (at(400), "แอน", "ช้ากว่า 60 วินาที"),
]


# ------------------------------------------------------------------ turns and stats
def test_consecutive_lines_of_one_speaker_merge_within_60_seconds():
    turns = md.group_turns(LINES)
    assert [(t.speaker, t.text) for t in turns[:2]] == [
        ("แอน", "สวัสดีทุกคน เรามาเริ่มกันเลย"),
        ("บอย", "ผมส่งรายงาน 3 ฉบับวันศุกร์"),
    ]
    assert turns[2].speaker == md.SALID_SPEAKER
    assert turns[3].text == "ขอบคุณค่ะ แล้วเจอกัน"  # 30 s apart
    assert turns[4].text == "ช้ากว่า 60 วินาที"  # 170 s after the previous line: a new turn
    assert turns[0].at == at(0)


def test_lines_are_ordered_and_blank_lines_dropped():
    turns = md.group_turns([(at(10), "บอย", "b"), (at(0), "แอน", "a"), (at(11), "บอย", "  ")])
    assert [(t.speaker, t.text) for t in turns] == [("แอน", "a"), ("บอย", "b")]


def test_speaker_stats_count_turns_and_share_without_salid():
    rows = md.speaker_stats(md.group_turns(LINES))
    names = [r[0] for r in rows]
    assert md.SALID_SPEAKER not in names and set(names) == {"แอน", "บอย"}
    by = {r[0]: r for r in rows}
    assert by["แอน"][1] == 3 and by["บอย"][1] == 1
    assert sum(r[2] for r in rows) in (99, 100, 101)
    assert rows[0][0] == "แอน"  # busiest first


# ------------------------------------------------------------------ the markdown
def info():
    return md.MeetingInfo(
        room="ห้องประชุม",
        started_at=T0,
        ended_at=T0 + timedelta(minutes=25),
        attendees=["แอน", "บอย"],
        summary="**สรุปการประชุม**\nคุยเรื่องรายงาน\n**มติ/ข้อตกลง**\n- ส่งวันศุกร์",
    )


def test_markdown_has_header_summary_sections_transcript_and_note():
    text = md.render_markdown(info(), md.group_turns(LINES), "th")
    assert text.startswith("# รายงานการประชุม")
    assert "ห้องประชุม" in text and "2026-10-15" in text and "09:00 - 09:25 (25 นาที)" in text
    assert "แอน, บอย" in text
    assert "## สรุปการประชุม" in text and "## มติ/ข้อตกลง" in text
    assert "## บทพูดของแต่ละคน" in text and "ถอดความจากเสียงอัตโนมัติ" in text
    assert "**09:00 แอน:** สวัสดีทุกคน เรามาเริ่มกันเลย" in text
    assert "**09:00 สลิด:** รับทราบค่ะ" in text
    assert "| แอน | 3 |" in text


def test_english_document_labels_salid():
    text = md.render_markdown(info(), md.group_turns(LINES), "en")
    assert text.startswith("# Meeting minutes") and "## Transcript" in text
    assert "Auto-transcribed from speech" in text and "**09:00 Salid:**" in text


def test_transcript_can_be_left_out_or_missing():
    short = md.render_markdown(info(), md.group_turns(LINES), "en", include_transcript=False)
    assert "Transcript" not in short and "Decisions" not in short.split("##")[0]
    none = md.render_markdown(info(), [], "en")
    assert "No transcript is available" in none


# ------------------------------------------------------------------ clean mode
class FakeModels:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate_content(self, model, contents, config):
        self.calls.append(contents)
        if isinstance(self.reply, Exception):
            raise self.reply
        text = self.reply(contents) if callable(self.reply) else self.reply
        return SimpleNamespace(text=text)


def llm(reply):
    return SimpleNamespace(models=FakeModels(reply))


def sent_items(contents):
    raw = contents[0]
    return json.loads(raw[raw.index("[{") :])


def test_clean_replaces_only_the_text_and_keeps_the_skeleton():
    turns = md.group_turns(LINES)

    def reply(contents):
        return json.dumps(
            [{"i": r["i"], "t": r["t"].replace("เอ่อ", "").strip()} for r in sent_items(contents)],
            ensure_ascii=False,
        )

    turns2 = [md.Turn(t.at, t.speaker, "เอ่อ " + t.text) for t in turns]
    out = md.clean_turns(llm(reply), "m", turns2, "th")
    assert [(t.at, t.speaker) for t in out] == [(t.at, t.speaker) for t in turns]
    assert [t.text for t in out] == [t.text for t in turns]


def test_clean_never_sends_speakers_or_times_and_ignores_reassignment():
    turns = md.group_turns(LINES)
    seen = {}

    def reply(contents):
        seen["prompt"] = contents[0]
        items = sent_items(contents)
        # the model tries to relabel and merge: extra keys / fewer items
        return json.dumps(
            [{"i": r["i"], "t": r["t"], "speaker": "บอย"} for r in items][:-1], ensure_ascii=False
        )

    out = md.clean_turns(llm(reply), "m", turns, "th")
    assert out == turns
    assert "แอน" not in seen["prompt"] and "09:00" not in seen["prompt"]


def test_clean_rejects_changed_numbers_and_wild_lengths_per_turn():
    turns = [md.Turn(at(0), "แอน", "ส่งรายงาน 3 ฉบับวันศุกร์นะคะ"), md.Turn(at(70), "บอย", "โอเคครับ")]

    def reply(contents):
        return json.dumps(
            [
                {"i": 0, "t": "ส่งรายงาน 5 ฉบับวันศุกร์"},  # number changed: rejected
                {"i": 1, "t": "โอเค"},  # short original: accepted
            ],
            ensure_ascii=False,
        )

    out = md.clean_turns(llm(reply), "m", turns, "th")
    assert out[0] == turns[0] and out[1].text == "โอเค" and out[1].speaker == "บอย"


def test_clean_falls_back_to_verbatim_on_any_failure():
    turns = md.group_turns(LINES)
    assert md.clean_turns(llm(RuntimeError("boom")), "m", turns, "th") == turns
    assert md.clean_turns(llm("not json"), "m", turns, "th") == turns
    assert md.clean_turns(None, "m", turns, "th") == turns


# ------------------------------------------------------------------ files
def test_pdf_and_docx_bytes():
    text = md.render_markdown(info(), md.group_turns(LINES), "th")
    docx = md.build_file("docx", text, org="x", lang="th", generated_at=T0, fonts=lambda: None)
    assert zipfile.is_zipfile(io.BytesIO(docx))
    assert md.build_file("md", text, org="x", lang="th", generated_at=T0, fonts=lambda: None)
    try:
        fonts = find_fonts("")
    except Exception:  # no Thai font on this machine
        return
    pdf = md.build_file("pdf", text, org="x", lang="th", generated_at=T0, fonts=lambda: fonts)
    assert pdf.startswith(b"%PDF")


def test_file_stem():
    assert md.file_stem(T0) == "minutes-2026-10-15-0900"


# ------------------------------------------------------------------ the tool: access rules
def record(**kw):
    base = {
        "voice_channel_id": 77,
        "voice_channel_name": "ห้องประชุม",
        "attendee_ids": [10, 11],
        "attendee_names": ["แอน", "บอย"],
        "started_at": T0.isoformat(),
        "ended_at": (T0 + timedelta(minutes=25)).isoformat(),
        "summary": "**สรุปการประชุม**\nคุยเรื่องรายงาน",
    }
    base.update(kw)
    return base


def make_ctx(asker=10, can_view=False, audience_ok=True, private_ok=True, lines=None, **kw):
    private = []

    def send_private(name, data, caption):
        private.append((name, data, caption))
        return private_ok

    ctx = md.MeetingDocCtx(
        asker_id=asker,
        lang="th",
        org="ทดสอบ",
        records=lambda: [record()],
        can_view_voice=lambda cid: can_view,
        lines=lambda rec: LINES if lines is None else lines,
        audience_ok=lambda rec: audience_ok,
        send_private=send_private,
        fonts=lambda: None,
        now=lambda: T0,
        **kw,
    )
    return ctx, private


def test_attendee_gets_the_file_in_the_reply():
    ctx, private = make_ctx(asker=10)
    out = ctx.meeting_document("last", "docx")
    assert out["status"] == "ok" and out["file"] == "minutes-2026-10-15-0900.docx"
    assert out["delivery"] == "attached to the reply" and private == []
    name, data = ctx.outbox[0]
    assert name.endswith(".docx") and zipfile.is_zipfile(io.BytesIO(data))


def test_someone_who_can_see_the_room_but_did_not_attend_is_allowed():
    ctx, _ = make_ctx(asker=99, can_view=True)
    assert ctx.meeting_document("last", "md")["status"] == "ok"


def test_outsider_is_refused():
    ctx, private = make_ctx(asker=99, can_view=False)
    out = ctx.meeting_document("last", "md")
    assert out["status"] == "no_meeting" and ctx.outbox == [] and private == []


def test_shared_room_sends_privately_and_says_so():
    ctx, private = make_ctx(asker=10, audience_ok=False)
    out = ctx.meeting_document("last", "md")
    assert out["delivery"] == "sent_privately" and "privately" in out["note"]
    assert ctx.outbox == [] and private and private[0][0].endswith(".md")
    assert "ห้องประชุม" in private[0][2]


def test_private_delivery_failure_is_reported_and_nothing_is_posted():
    ctx, _ = make_ctx(asker=10, audience_ok=False, private_ok=False)
    out = ctx.meeting_document("last", "md")
    assert out["status"] == "error" and ctx.outbox == []


def test_summary_only_and_missing_transcript():
    ctx, _ = make_ctx(lines=[])
    out = ctx.meeting_document("last", "md")
    assert out["transcript_available"] is False and "summary only" in out["note"]
    ctx2, _ = make_ctx()
    ctx2.meeting_document("last", "md", include_transcript=False)
    assert b"Transcript" not in ctx2.outbox[0][1] and "บทพูด".encode() not in ctx2.outbox[0][1]


def test_clean_mode_through_the_tool_and_bad_format():
    def reply(contents):
        return json.dumps(
            [{"i": r["i"], "t": r["t"] + "!"} for r in sent_items(contents)], ensure_ascii=False
        )

    ctx, _ = make_ctx(client=llm(reply), model="m")
    out = ctx.meeting_document("last", "md", mode="clean")
    assert out["mode"] == "clean" and "สวัสดีทุกคน เรามาเริ่มกันเลย!" in ctx.outbox[0][1].decode()
    assert "ทำความสะอาด" not in ctx.outbox[0][1].decode()
    assert ctx.meeting_document("last", "exe")["status"] == "error"


def test_meeting_selector_by_room_and_date():
    recs = [
        record(),
        record(
            voice_channel_name="ห้องขาย",
            started_at="2026-10-14T09:00:00+00:00",
            ended_at="2026-10-14T10:00:00+00:00",
            voice_channel_id=78,
        ),
    ]
    ok = lambda cid: True  # noqa: E731
    assert md.pick_record(recs, "ขาย", 10, ok)["voice_channel_id"] == 78
    assert md.pick_record(recs, "2026-10-14", 10, ok)["voice_channel_id"] == 78
    assert md.pick_record(recs, "last", 10, ok)["voice_channel_id"] == 77


def test_tool_function_exposes_the_outbox():
    ctx, _ = make_ctx()
    (fn,) = md.meeting_document_tools(ctx)
    assert fn.__name__ == "meeting_document" and fn.outbox is ctx.outbox


# ------------------------------------------------------------------ glue: who may read where
class Chan:
    def __init__(self, id, members):
        self.id, self.members = id, members


def viewer(uid, can):
    return SimpleNamespace(id=uid, bot=False, can=can)


class Voice:
    id = 77

    def permissions_for(self, m):
        return SimpleNamespace(view_channel=m.can)


def audience(message, owner=None):
    me = SimpleNamespace(
        private_chats=SimpleNamespace(owner_of=lambda g, c: owner),
    )
    guild = SimpleNamespace(id=1, get_channel=lambda cid: Voice())
    asker = viewer(10, True)
    return MeetingDocMixin._meeting_audience_ok(me, message, guild, asker, {"voice_channel_id": 77})


def test_audience_rules():
    ch_ok = Chan(5, [viewer(10, True), viewer(11, True)])
    ch_mixed = Chan(6, [viewer(10, True), viewer(12, False)])
    assert audience(SimpleNamespace(guild=None, channel=ch_ok, room=None))  # DM
    assert audience(SimpleNamespace(guild=object(), channel=ch_ok, room=None))
    assert not audience(SimpleNamespace(guild=object(), channel=ch_mixed, room=None))
    assert audience(
        SimpleNamespace(guild=object(), channel=ch_mixed, room=None), owner=10
    )  # own channel
    assert not audience(
        SimpleNamespace(guild=object(), channel=ch_ok, room=frozenset({10}))
    )  # voice
    assert not audience(SimpleNamespace(guild=object(), channel=Chan(7, []), room=None))
