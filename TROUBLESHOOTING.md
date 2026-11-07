# Troubleshooting

Start with the free check, which names the fix for most problems:

```bash
scripts/smoke.sh
```

Then find the symptom below. Commands run in the WSL terminal from the project folder. Installation is in [INSTALL.md](INSTALL.md), and how the parts fit together in [ARCHITECTURE.md](ARCHITECTURE.md).

## Keys

**`.env` is lost or damaged.** Its only copy is in `~/backups/study-assistant/env/`, under `/home`, never in the archives on D:. Copy it back with `cp ~/backups/study-assistant/env/<newest> .env && chmod 600 .env`. If that copy is gone too, **create new keys in the Anthropic and OpenRouter consoles** and write them into `.env` as in INSTALL.md, step 6. Then revoke the old keys in both consoles.

**"STUDY_ASSISTANT_ANTHROPIC_KEY is not set in .env"** (or the OpenRouter one). That option has no key. Add the line to `.env`; the name must be exactly as shown. The app never reads keys from the environment, so `export` doesn't help.

**Deep fails at once with "HTTP 401, authentication_error"** (or 403), and Sonnet isn't tried: the key was revoked or mistyped, and Sonnet would use the same key. Make a new key in the Anthropic console. Qwen shows the same kind of error for a bad OpenRouter key.

**Deep fails with a usage-limit error.** The Anthropic workspace reached the monthly limit set in its console. The app's own figure (`hosted_monthly_limits` in `config.yaml`) is only shown beside the month's spend; the console's limit is what stops calls. Raise it there, or use Qwen or a local option until the month turns.

## Starting

**"Can't start: Ollama isn't answering"**: `sudo systemctl start ollama`. If it keeps stopping, see `journalctl -u ollama -e`.

**"Can't start: port 8000 is already in use"**: the app is probably running already in another terminal; open http://localhost:8000. Otherwise `ss -ltnp 'sport = :8000'` shows what holds the port.

**"No sandbox image"** at start, or the code, PDF, Word, PowerPoint and SymPy tools fail with "The sandbox isn't available": run `scripts/build_sandbox.sh`. **"No office image"**, or decks and Word/PowerPoint previews fail: run `scripts/build_office.sh`. Both need Docker running (`sudo systemctl start docker`) and a network, once.

**"Docker couldn't start the sandbox: permission denied"**: your user isn't in the `docker` group yet. Run `sudo usermod -aG docker "$USER"`, then open a new terminal.

**The page is blank, or the frontend is old**: `start.sh` rebuilds it when its source changes. To force a rebuild: `rm -rf frontend/dist && scripts/start.sh`.

**"metadata.yaml can't be read"** in the library pane. A typo in `library/metadata.yaml`, often a quote left open or a tab. The message gives the line. Until it's fixed, nothing that depends on the file goes ahead:
- ingestion waits;
- Explain and Suggest refuse hosted options, since the app can't tell which pages are kept out;
- notes anchored to a page leave search.

Fix the line; the app picks the file up again without a restart. `exclude_pages` must be page numbers, such as `[1]` or `[1, 3]`.

## Ingestion

**"Not added: …"** under **Add PDF or deck…**: the reason is in the message. A file with that name is already in `library/` (rename it, or remove the old one first), the same document is there under another name, it is over 200 MB (`upload_max_bytes` in `config.yaml`), or it isn't the PDF or `.pptx` deck its name says. Copying a file into `library/` yourself still works as before.

**A document shows "failed"**: hover its row in the library pane for the stage and error. A stage is tried three times before the document is marked failed, and every finished stage is kept. Once the cause is fixed, stop the app and queue the library again from that stage. This queues every document, and each one is redone from that stage:

```bash
cd backend && ../.venv/bin/python -m app.ingest.worker --once --reindex-from parse
```

Use `--reindex-from formulas` to keep the parse. A file that changes is queued again by itself.

**"LocalEntryNotFoundError" or "offline" in a parse error**, or the smoke test says the parser's models aren't in the cache: the Hugging Face models haven't been downloaded yet. Start the app with the network on and add any PDF; the first parse downloads them (about 4 GB, into `~/.cache/huggingface`).

**A deck fails at parse**: the office image is missing (`scripts/build_office.sh`), or LibreOffice gave up on the file. Open it in PowerPoint and save it again as `.pptx`. If its fonts look wrong in the viewer, check `office_fonts_dir` in `config.yaml`.

**Summaries fail with "model not found"**: the summary tree's model isn't pulled. Run `ollama pull qwen3.8:27b`, or leave it: the document is searchable without summaries.

**Figures say "waiting for your OK"**: descriptions use a hosted model, so they wait until you accept the estimated cost in the library pane. Until then the figures are searched by their captions only.

**A changed PDF lost highlights or notes**: when a file changes, each highlight and note looks for its passage again in the new file. One it can't find for sure is marked lost, never moved by a guess. Lost highlights are listed in Notes, and a lost note can be attached to a new passage.

## Answers

**An answer is very slow, or the GPU looks full**: only one local chat model fits in 16 GB. Switching between Fast and Balanced reloads it (20–60 s), and the selector says when. Something else may be using the GPU too: `nvidia-smi` lists the processes. Never run two evaluations at once when one uses a local model, or both spill to the CPU.

**Fast or Balanced stops with "stopped at its token limit"**: the model thought for too long. Ask again in Deep (the button under the answer), or split the question.

**Qwen stops without an answer**: its thinking reached the 16,000-token limit (3 of the 35 gold questions). The app offers Deep for the same question. It never switches by itself.

**An answer has no citations**: gpt-oss leaves them out about once in ten answers, and the answer says so. Ask again, or use another option.

**Web search: "is SearXNG running? scripts/searxng.sh start"**: run that. If it says there's no image, run `docker pull searxng/searxng:2026.10.19-81371861f` once. Web search is offered only in Deep and Qwen, and only for a question with Web on.

**Explain or Suggest says a page is "kept away from hosted models"**: the page is in `exclude_pages` for that document in `library/metadata.yaml`. Choose a local option, or remove the page from the list. A page taken off the list only comes back into search when the document is chunked again, with the reindex command under Ingestion.

## Backups

**"Backups are off: set backup_dir in config.yaml"**: set it as in INSTALL.md, step 6. **"Can't create …: Is the drive there?"**, or backups fail otherwise: the backup drive isn't mounted (`ls /mnt/d`), or `backup_dir` in `config.yaml` points elsewhere. `scripts/backup.sh verify` checks the newest archive end to end, and the log is in `~/backups/study-assistant/backup.log`.

**Restoring**: see [INSTALL.md, step 9](INSTALL.md#9-backups-optional). A restore checks the whole archive before writing anything, and never overwrites a file that differs. Ollama models, the Docker images and SearXNG aren't in backups: install them again as in INSTALL.md, steps 3 and 5.
