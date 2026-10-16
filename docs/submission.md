# Salid — submission notes (Google Cloud AI Builder Cup 2026)

Salid (สลิด) is an AI teammate that lives in a company's Discord. It reads what the team
shares, keeps the books, joins meetings, schedules, follows up and reports, and it keeps
company secrets according to the Discord role hierarchy. People talk to it like a colleague,
in Thai or English. No commands to learn.

## Problem

Small Thai companies run on chat. Receipts, decisions, files and meeting talk are scattered
across channels; nobody has time to keep the ledger, chase action items or write the daily
report. Off-the-shelf bots need commands and setup that non-technical owners never finish.

## What Salid does

| Area | What the team sees |
| --- | --- |
| Receipts and money | Post a receipt, slip, invoice or PDF anywhere: Salid records it, flags duplicates (asks before deducting), forwards misplaced receipts to the receipt channel, answers "how much did we spend on X" with per-currency totals, and makes PDF expense reports. |
| Team memory | Remembers every channel it may see (text, images, PDFs, Word/Excel/PowerPoint, audio/video transcripts) for a retention period the admin sets; daily summaries per channel; answers "what did we decide yesterday" with links to the source messages. |
| Files and links | Send it any file, voice message, web link or YouTube video: it says what is inside and decides what to do next (record, summarise, suggest next steps). |
| Meetings | Schedules Discord or Google Meet meetings, reminds people with one-click links, nudges no-shows, joins Discord voice to take notes, answers when called by name, posts minutes. |
| Coordinator (COO) | Turns chat and meetings into tasks, follows up each morning by DM, posts an evening report (spending, who did what, meetings, risks) in a private leadership channel. |
| Confidentiality | Higher roles can see lower roles' work, never the reverse; hidden information is offered via a consent request to its owner, and only that grant is unlocked. |
| Feels like a person | Per-channel modes (off / listen / call / join / chat), instant "give me a moment" acks, short natural replies, DMs, @mentions, 👀 when it has read a file quietly. |

## Built on Google Cloud

- **Gemini on Vertex AI** (`gemini-3.5-flash` for answers, `gemini-3.5-flash-lite` for cheap
  gates, summaries, image and document reading) through the **Agent Development Kit (ADK)**:
  one agent with code-bound tools (ledger, memory, files, web, scheduling, PDFs, email,
  settings, coordinator).
- **Gemini URL context + Google Search** for links and current information, **Gemini audio**
  for meeting and media transcription, **Gemini CLI** for long multi-step analysis.
- **Cloud Run** (web hub service + bot worker pool), **Firestore** (ledger, memory,
  settings, tasks, grants), **Cloud Storage** (receipt archive), **Secret Manager**,
  **Cloud Build / Artifact Registry**.

## Safety by design

- Every data tool is scoped in code to the guild, the asker and the answering channel's
  audience; the model never chooses scope.
- Outward actions (meetings, voiding receipts, email) need an explicit confirmation in a later
  message; settings changes need admin or executive rights.
- No audio or image files are stored; only text notes. Retention is set per server; deleting
  a Discord message deletes it from memory and re-summarises that day.
- Salid asks for the minimum Discord permissions (no Administrator, no Manage Messages).

## Demo video (3 minutes) — script

1. **0:00 Hook (15 s).** Owner's voice: "Our company runs on Discord. Salid is the teammate
   who never forgets." Show the server with Salid online.
2. **0:15 Receipts (30 s).** Drop a Thai receipt photo and an English invoice PDF in a random
   channel. Salid: "ขอดูแป๊บนึงนะครับ" → records both, forwards to #tax-invoice. Ask
   "เดือนนี้จ่ายไปเท่าไหร่" → ฿ and $ shown separately. "มีบิลซ้ำไหม" → finds it, asks to confirm.
3. **0:45 Memory and files (35 s).** Ask in DM "ลูกค้า ABC ตกลงราคาเท่าไหร่" → answer with a
   link to last week's message. Send a Word contract → summary + "set a reminder for the
   first payment?". Send a voice message → Salid answers what was asked.
4. **1:20 Meetings (40 s).** "นัดประชุมทีมพรุ่งนี้ 10 โมงกับ @A @B ในห้อง Voice" → confirm →
   event created, DMs sent. Show reminder + no-show nudge DM. Salid joins voice, someone says
   "สลิด เมื่อกี้เราตกลงอะไร" → spoken answer, minutes posted after.
5. **2:00 Coordinator and secrets (40 s).** Evening report in #รายงานจากสลิด (who did what,
   spending, risks). A junior asks about a manager-only topic → Salid offers to ask; the
   manager gets Approve/Decline in DM; after Approve the junior gets the answer.
6. **2:40 Close (20 s).** "No commands. Just talk to Salid." Architecture slide: Discord →
   Cloud Run → ADK + Gemini on Vertex → Firestore. GitHub link.

## Submission checklist

- [ ] Working deployed link (Cloud Run hub + bot online in a demo server)
- [ ] Demo video ≤ 3 minutes (script above)
- [ ] Deck (problem, solution, demo screenshots, architecture, safety, roadmap)
- [ ] Public GitHub repository (history scanned for secrets: no tokens found in 69 commits)
- [ ] All materials in English
