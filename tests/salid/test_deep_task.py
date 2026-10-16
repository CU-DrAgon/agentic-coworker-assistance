"""Deep tasks: scoped export, isolated env/settings, output parsing, cleanup (fake subprocess)."""

import json
import os
from pathlib import Path

import pytest
from conftest import ALICE, EXEC_ID, GUILD

from salid.cli import main as cli_main
from salid.config import Settings
from salid.deep_task import (
    DeepTaskError,
    build_cli_settings,
    build_command,
    parse_cli_output,
    run_deep_task,
)
from salid.i18n import t

GOOD = {
    "session_id": "s1",
    "response": "เสร็จแล้ว",
    "stats": {
        "models": {
            "gemini-3-flash-preview": {
                "api": {"totalRequests": 5},
                "tokens": {"total": 1234},
            }
        }
    },
}


class FakeRun:
    """Stands in for the subprocess: records what it was given, writes report.md like the CLI."""

    def __init__(self, stdout=None, report="# รายงาน\nเนื้อหา", rc=0, boom=None):
        self.stdout = json.dumps(GOOD) if stdout is None else stdout
        self.report, self.rc, self.boom = report, rc, boom
        self.seen = {}

    def __call__(self, cmd, env, cwd, timeout):
        if self.boom:
            raise self.boom
        self.seen = {
            "cmd": cmd,
            "env": env,
            "cwd": Path(cwd),
            "timeout": timeout,
            "files": sorted(p.name for p in Path(cwd).iterdir()),
            "csv": (Path(cwd) / "receipts.csv").read_text(encoding="utf-8"),
            "task": (Path(cwd) / "TASK.md").read_text(encoding="utf-8"),
            "agg": json.loads((Path(cwd) / "aggregates.json").read_text(encoding="utf-8")),
            "settings": json.loads(
                (Path(env["GEMINI_CLI_HOME"]) / ".gemini/settings.json").read_text()
            ),
        }
        if self.report is not None:
            (Path(cwd) / "report.md").write_text(self.report, encoding="utf-8")
        return self.rc, self.stdout, ""


@pytest.fixture
def ds(settings):
    return Settings(
        exec_user_ids={EXEC_ID}, gemini_cli_path="/opt/gemini", deep_task_timeout=77
    ).model_copy(update={"tz": settings.tz})


def test_disabled_without_binary_path(seeded, settings):
    with pytest.raises(DeepTaskError, match="SALID_GEMINI_CLI_PATH") as e:
        run_deep_task(settings, seeded, "x", guild_id=GUILD)
    assert str(e.value) == t("deep_disabled", "th")


def test_run_returns_report_stats_and_cleans_up(seeded, ds):
    fake = FakeRun()
    r = run_deep_task(ds, seeded, "วิเคราะห์ค่าใช้จ่ายเดือนนี้", guild_id=GUILD, runner=fake)
    assert r.report.startswith("# รายงาน") and r.text == r.report
    assert r.tokens == 1234 and r.models == {
        "gemini-3-flash-preview": {"requests": 5, "tokens": 1234}
    }
    assert r.pinned_ok and r.session_id == "s1"
    assert fake.seen["timeout"] == 77
    assert fake.seen["files"] == ["TASK.md", "aggregates.json", "receipts.csv"]  # nothing else
    assert "วิเคราะห์ค่าใช้จ่ายเดือนนี้" in fake.seen["task"]
    assert not fake.seen["cwd"].exists() and not fake.seen["cwd"].parent.exists()  # temp dir gone


def test_exec_export_has_everything_and_user_export_is_scoped(seeded, ds):
    f1, f2 = FakeRun(), FakeRun()
    run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=f1)
    assert "Bob" in f1.seen["csv"] and f1.seen["agg"]["overall"]["total"] == 13370.5
    run_deep_task(ds, seeded, "t", guild_id=GUILD, submitter_id=ALICE, runner=f2)
    assert "Bob" not in f2.seen["csv"] and "Makro" in f2.seen["csv"]
    assert f2.seen["agg"]["overall"] == {
        "total": 1250.5,
        "totals_by_currency": {"THB": 1250.5},
        "count": 2,
    }
    assert "Bob" not in json.dumps(f2.seen["agg"], ensure_ascii=False)


def test_export_excludes_internal_identifiers(seeded, ds):
    f = FakeRun()
    run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=f)
    header = f.seen["csv"].splitlines()[0]
    assert "sha256" not in header and "message" not in header and "channel" not in header
    assert "original_uri" not in header


def test_env_is_minimal_and_isolated(seeded, ds, monkeypatch):
    for k in ("SALID_DISCORD_TOKEN", "GH_TOKEN", "DISCORD_WEBHOOK_URL", "SALID_DB_PATH"):
        monkeypatch.setenv(k, "SECRET-VALUE")
    f = FakeRun()
    run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=f)
    env = f.seen["env"]
    assert "SECRET-VALUE" not in json.dumps(env)
    assert not any(k.startswith("SALID_") or k in {"GH_TOKEN"} for k in env)
    assert env["HOME"] == env["GEMINI_CLI_HOME"] != os.path.expanduser("~")
    assert (
        env["GOOGLE_GENAI_USE_VERTEXAI"] == "true" and env["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    )
    assert (
        env["GOOGLE_CLOUD_PROJECT"] == ds.gcp_project and env["GOOGLE_CLOUD_LOCATION"] == "global"
    )


def test_command_pins_model_and_blocks_shell(ds):
    cmd = build_command(ds)
    assert cmd[0] == "/opt/gemini" and cmd[cmd.index("-m") + 1] == "gemini-3-flash-preview"
    assert cmd[cmd.index("--approval-mode") + 1] == "auto_edit" and "yolo" not in " ".join(cmd)
    assert cmd[cmd.index("--output-format") + 1] == "json"


def test_settings_pin_every_alias_and_remove_shell_tools(seeded, ds):
    f = FakeRun()
    run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=f)
    cfg = f.seen["settings"]
    assert cfg == build_cli_settings("gemini-3-flash-preview")
    assert cfg["model"]["name"] == "gemini-3-flash-preview" and cfg["model"]["disableLoopDetection"]
    res = cfg["modelConfigs"]["modelIdResolutions"]
    assert all(v == {"default": "gemini-3-flash-preview"} for v in res.values())
    assert {"gemini-3.5-flash", "gemini-3.8-flash", "gemini-3-flash-preview", "flash"} <= set(res)
    assert (
        "run_shell_command" not in cfg["tools"]["core"] and "web_fetch" not in cfg["tools"]["core"]
    )
    assert cfg["experimental"]["dynamicModelConfiguration"]


def test_unpinned_model_is_flagged(seeded, ds):
    out = {**GOOD, "stats": {"models": {"gemini-3.8-flash": {"api": {}, "tokens": {"total": 9}}}}}
    r = run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=FakeRun(stdout=json.dumps(out)))
    assert not r.pinned_ok and r.tokens == 9


def test_falls_back_to_response_text_without_report(seeded, ds):
    r = run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=FakeRun(report=None))
    assert r.report is None and r.text == "เสร็จแล้ว"


def test_symlinked_report_is_ignored(seeded, ds):
    class Evil(FakeRun):
        def __call__(self, cmd, env, cwd, timeout):
            out = super().__call__(cmd, env, cwd, timeout)
            (Path(cwd) / "report.md").unlink()
            (Path(cwd) / "report.md").symlink_to("/etc/hostname")
            return out

    r = run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=Evil())
    assert r.report is None and r.text == "เสร็จแล้ว"


def test_output_with_noise_and_errors(seeded, ds):
    assert parse_cli_output("warn: x\n" + json.dumps(GOOD) + "\n")["session_id"] == "s1"
    with pytest.raises(DeepTaskError):
        run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=FakeRun(stdout="not json", rc=1))
    with pytest.raises(DeepTaskError):
        run_deep_task(
            ds,
            seeded,
            "t",
            guild_id=GUILD,
            runner=FakeRun(stdout=json.dumps({"error": {"m": "x"}})),
        )


def test_timeout_propagates_and_still_cleans_up(seeded, ds):
    f = FakeRun(boom=DeepTaskError("งานใหญ่ใช้เวลานานเกิน 77 วินาที จึงยกเลิก"))
    with pytest.raises(DeepTaskError, match="นานเกิน"):
        run_deep_task(ds, seeded, "t", guild_id=GUILD, runner=f)


def test_cli_deep_refuses_non_exec(capsys):
    cli_main(["deep", "ทำรายงาน", "--as-user", "5"])
    assert "เฉพาะผู้บริหาร" in capsys.readouterr().out


def test_real_runner_kills_on_timeout(tmp_path):
    from salid.deep_task import _run

    rc, out, _ = _run(["python3", "-c", "print('hi')"], {"PATH": os.environ["PATH"]}, tmp_path, 20)
    assert rc == 0 and out.strip() == "hi"
    with pytest.raises(DeepTaskError, match="นานเกิน"):
        _run(["sleep", "30"], {"PATH": os.environ["PATH"]}, tmp_path, 0.5)


# --------------------------------------------------------------------- Discord side


def test_trigger_parsing():
    from salid.deep_task import parse_deep_trigger

    assert parse_deep_trigger("งานใหญ่: วิเคราะห์เดือนนี้") == "วิเคราะห์เดือนนี้"
    assert parse_deep_trigger("  /deep  ทำรายงาน ") == "ทำรายงาน"
    assert parse_deep_trigger("/DEEP") == "" and parse_deep_trigger("งานใหญ่：x") == "x"
    assert parse_deep_trigger("เดือนนี้ใช้จ่ายเท่าไร") is None


def test_reply_parts_attaches_long_reports():
    from salid.deep_task import INLINE_LIMIT, DeepResult, reply_parts

    short = DeepResult(text="สั้น", tokens=5, models={"m": {"requests": 1, "tokens": 5}})
    text, att = reply_parts(short)
    assert att is None and text.startswith("สั้น") and "tokens 5" in text
    long = DeepResult(text="บรรทัด\n" * 1000, seconds=3.0)
    text, att = reply_parts(long)
    assert att == long.text.encode() and len(text) < INLINE_LIMIT + 300 and "report.md" in text


class FakeMessage:
    def __init__(self):
        self.replies, self.sent = [], []
        outer = self

        class Chan:
            async def send(self, text, **kw):
                outer.sent.append((text, kw))

        self.channel = Chan()

    async def reply(self, text, **kw):
        self.replies.append((text, kw))


@pytest.fixture
def bot(seeded, ds):
    from salid.bot import SecretaryBot

    return SecretaryBot(ds, seeded, None)


async def test_bot_refuses_non_exec_and_runs_no_task(bot, monkeypatch):
    called = []
    monkeypatch.setattr("salid.bot.run_deep_task", lambda *a, **k: called.append(1))
    m = FakeMessage()
    await bot._deep(m, GUILD, "ทำรายงาน", is_exec=False)
    assert "เฉพาะผู้บริหาร" in m.replies[0][0] and not called and not bot._deep_tasks


async def test_bot_disabled_and_usage(bot):
    m = FakeMessage()
    await bot._deep(m, GUILD, "", is_exec=True)
    assert "บอกโจทย์" in m.replies[0][0]
    bot.s = bot.s.model_copy(update={"gemini_cli_path": ""})
    await bot._deep(m, GUILD, "x", is_exec=True)
    assert "SALID_GEMINI_CLI_PATH" in m.replies[1][0]


async def test_bot_acks_then_posts_result_in_background(bot, monkeypatch):
    import asyncio

    from salid.deep_task import DeepResult

    def fake_run(settings, store, task, **kw):
        assert task == "ทำรายงาน" and "submitter_id" not in kw  # exec: unscoped
        return DeepResult(text="# ผล", report="# ผล", seconds=2, tokens=10)

    monkeypatch.setattr("salid.bot.run_deep_task", fake_run)
    m = FakeMessage()
    await bot._deep(m, GUILD, "ทำรายงาน", is_exec=True)
    assert "เดี๋ยวสลิดไล่ดูให้นะคะ" in m.replies[0][0] and not m.replies[1:]  # ack first
    await asyncio.gather(*bot._deep_tasks)
    assert "# ผล" in m.replies[1][0] and "tokens 10" in m.replies[1][0]


async def test_bot_rejects_second_concurrent_task_and_reports_failure(bot, monkeypatch):
    import asyncio
    import threading

    release = threading.Event()

    def slow(settings, store, task, **kw):
        release.wait(5)
        raise DeepTaskError("พัง")

    monkeypatch.setattr("salid.bot.run_deep_task", slow)
    m1, m2 = FakeMessage(), FakeMessage()
    await bot._deep(m1, GUILD, "a", is_exec=True)
    await bot._deep(m2, GUILD, "b", is_exec=True)
    assert "กำลังทำงานใหญ่" in m2.replies[0][0]
    release.set()
    await asyncio.gather(*bot._deep_tasks)
    assert m1.sent[0][0] == "❌ พัง"


# ------------------------------------------------------------------ language
def test_task_file_report_language(seeded, ds, tmp_path):
    from salid.deep_task import prepare_workspace

    def task_md(lang, task="analyze"):
        ws = tmp_path / f"ws-{lang}"
        ws.mkdir(exist_ok=True)
        prepare_workspace(ws, seeded, ds, task, GUILD, lang=lang)
        return (ws / "TASK.md").read_text(encoding="utf-8")

    assert "ฉบับสุดท้ายเป็นภาษาไทย" in task_md("th")
    assert "เป็นภาษาอังกฤษ (English)" in task_md("en")
    assert "ด้วยภาษาเดียวกับที่เขียนโจทย์" in task_md("auto")


def test_deep_messages_follow_the_language(seeded, settings):
    from salid.deep_task import DeepResult, parse_deep_trigger, reply_parts

    with pytest.raises(DeepTaskError) as e:
        run_deep_task(settings, seeded, "x", guild_id=GUILD, lang="en")
    assert str(e.value) == t("deep_disabled", "en") and "not enabled" in str(e.value)
    with pytest.raises(DeepTaskError) as e:  # auto follows the task text
        run_deep_task(settings, seeded, "analyze spending", guild_id=GUILD, lang="auto")
    assert "not enabled" in str(e.value)
    long = DeepResult(text="line\n" * 1000, seconds=3.0)
    text, _ = reply_parts(long, "en")
    assert "attached as report.md" in text and "model:" in text
    assert parse_deep_trigger("deep task: analyze") == "analyze"
    assert parse_deep_trigger("Big task: analyze") == "analyze"
