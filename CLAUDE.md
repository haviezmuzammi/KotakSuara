# CLAUDE.md — rules for agents working on KotakSuara

KotakSuara (`yassalam-tts-id`) is an Indonesian TTS + voice-clone service for Yassalam's Islamic study
(kajian/dakwah) narration. Production runs on a Windows GPU PC on Yassalam's LAN; you (a cloud agent)
work on the code through GitHub and cannot reach the running service directly. Read `README.md` first.

## Non-negotiable rules
1. **Never synthesize Quran verses.** `[AYAT]...[/AYAT]` and Arabic-script spans typed into a script must become
   silence with timestamps (`app/text/segmenter.py`). Do not weaken this, add a bypass flag, or read
   transliterations of verses aloud. The only Arabic that is spoken comes from the curated `arabic_terms` list in
   `config/lexicon.yaml` (names of Allah, honorifics, common dhikr), spoken with the engine's Arabic model.
   Dhikr that are also Quranic phrases (basmalah, hamdalah, istirja') may be spoken — Yassalam decision
   2026-10-11. Additions to `arabic_terms` must be reviewed by Yassalam's asatidzah; never add whole verses.
2. **Voice cloning requires recorded consent.** Every profile needs `consent_by`, `consent_date`,
   `consent_evidence`. Never add a path that clones a voice without them. Never commit voice recordings,
   datasets, or reference audio of real people (they live in `data/` on the PC, which is git-ignored).
3. **Licenses are explicit.** Each engine declares `license`, `commercial_ok`, `unofficial`. Only
   `commercial_ok: true` output may be used in monetized YouTube videos. Check a model's license before
   adding it; default to `commercial_ok=False` when unsure.
4. **No secrets in git.** Configuration comes from `.env` (see `.env.example`). Do not hardcode keys, IPs of
   private hosts, or tokens. The repository is public.
5. **Do not expose the service publicly.** Access is limited by the Windows Firewall + `ALLOWED_CLIENTS`.
   Cloud agents reach it only through `docs/ROADMAP.md`: C1 GitHub job queue (private repo
   mmymedia-lab/KotakSuara-jobs), C2 gateway `https://suara.yassalam.id` (bearer token from the environment
   secret `KOTAKSUARA_TOKEN`) or C3 MCP server `https://suara.yassalam.id/mcp` (`.mcp.json`, secret
   `KOTAKSUARA_MCP_TOKEN`). Never print, log or commit those tokens; never add port forwarding.
6. Religious content: Ahlus Sunnah wal Jama'ah (manhaj salaf) framing; do not author fatwas. Narration
   scripts come from Yassalam's team.

## Working on the code
- Python 3.11, FastAPI, SQLite. Code, comments and identifiers in English; user-facing reports in Indonesian.
- Tests run without a GPU: `pytest --cov=app` (CI does the same). Keep `app/text` coverage >= 80%.
  Use the fake engines in `tests/conftest.py`; never require model downloads in unit tests.
- New engine: follow "Adding an engine" in `README.md` (one file in `app/engines/`, register it, add tests).
- Indonesian text rules live in `app/text/normalizer_id.py` and `config/lexicon.yaml`. Prefer lexicon
  entries over code for pronunciation fixes.
- GPU-only checks (`scripts/live_run.ps1`) run on the PC; ask the maintainer to run them and report.
- Keep changes small and reviewed via pull requests; describe what you verified and what you could not.
