# TODO

Project status for the Google Cloud AI Builder Cup. **Submission deadline: 2026-10-18.**
Move items between sections as work progresses.

## To do

### Before submission (by 2026-10-17)
- [ ] Re-authorize Salid on the team server with the new permissions (1166416)
- [ ] Create the Google OAuth Client ID for hub sign-in and list the allowed admin emails
- [ ] Test the bot against real Firestore (`SALID_STORE=firestore`)
- [ ] Deploy the web service (`meeting-scribe`, adds `/hub`) to Cloud Run
- [ ] Deploy the bot as the Cloud Run worker pool `salid-bot` (`scripts/deploy_salid.sh --apply`)
- [ ] Scan the full git history for secrets, then make the repository public (the rules require a public repo)
- [ ] Update the architecture diagram (multi-tenant, voice, channel modes)

### Submission materials (English, due 2026-10-18)
- [ ] Working deployed link
- [ ] Demo video, under 3 minutes
- [ ] Deck explaining the solution
- [ ] Link to the public GitHub repository

### Later / ideas
- [ ] Channel modes for voice meetings (today voice answers only when called by name)
- [ ] Keep snoozes and the join-mode budget across restarts
- [ ] Google Search grounding for up-to-date answers
- [ ] Data deletion (per receipt, and when Salid leaves a server), privacy policy page, retention period
- [ ] Follow-up and commitment tracking
- [ ] LINE integration

## In progress
- [ ] Per-channel modes: off / listen / call / join / chat ([#14](../../pull/14)) — live Discord test, then merge
- [ ] PolyForm Noncommercial license ([#13](../../pull/13)) — waiting for merge
- [ ] Live test of voice meetings in a real Discord voice channel

## Done
- [x] Receipt photos read into an expense ledger ([#6](../../pull/6))
- [x] Spending Q&A with ADK, Firestore storage, Gemini CLI deep tasks ([#7](../../pull/7))
- [x] `/hub` web command center with Google sign-in ([#8](../../pull/8))
- [x] Deploy scripts for the bot: Dockerfile, Cloud Build, worker pool ([#9](../../pull/9))
- [x] Multi-tenant per Discord server, zero-config setup, general chat, `/ตั้งค่า` hub linking ([#10](../../pull/10))
- [x] Voice meetings: join, transcribe, answer on wake word, summarize ([#11](../../pull/11))
- [x] All of the above merged into `main` ([#12](../../pull/12))
- [x] Checked the competition rules: no specific license required; the repository must be public
