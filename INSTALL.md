# Installing the study assistant

From an empty WSL distribution to a first answer. Every step is needed unless it says optional. Lines starting with `PS>` run in Windows PowerShell; everything else runs in the WSL terminal, from the project folder once it exists. How the parts fit together is in [ARCHITECTURE.md](ARCHITECTURE.md); when something fails, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## What you need

- Windows 11 with WSL2, and an NVIDIA GPU with 16 GB, using the **Windows** NVIDIA driver. Don't install a Linux driver inside WSL: WSL passes the Windows one through. Tested on an RTX 4090 Laptop GPU with driver 621.14, in Ubuntu 26.04.
- About 35 GB of free disk inside WSL:

  | What | Size |
  |---|---|
  | Required local models (Fast, embeddings, the helper) | 16 GB |
  | Parser and reranker models | 4 GB |
  | Python packages, PyTorch and its CUDA libraries included | several GB |
  | Three Docker images | 2 GB |

  The optional models need more: Balanced 15 GB, the summary tree's model 17 GB.
- Keep the project and its data under `/home`, never under `/mnt/c` or `/mnt/d`: Windows drives are slow from WSL and can't keep Linux file permissions.
- For Deep and Qwen only: an Anthropic and an OpenRouter API key. Everything else is free and local.

## 1. WSL and Ubuntu

```
PS> wsl --install -d Ubuntu-26.04
```

Create your user when it asks. Then, in Ubuntu, check that the GPU is visible and that systemd is on:

```bash
nvidia-smi                      # names your GPU; if not, update the Windows NVIDIA driver
cat /etc/wsl.conf               # should contain [boot] systemd=true
```

If `systemd=true` is missing, add these two lines to `/etc/wsl.conf` with `sudo nano /etc/wsl.conf`, then restart WSL with `PS> wsl --shutdown`:

```
[boot]
systemd=true
```

## 2. System packages and Docker

```bash
sudo apt update
sudo apt install -y git curl build-essential python3.14 python3.14-venv docker.io docker-buildx
sudo usermod -aG docker "$USER"            # use Docker without sudo
sudo systemctl enable --now docker
```

Close the terminal and open a new one, so the `docker` group applies. Then:

```bash
docker ps                                   # an empty table, not "permission denied"
```

Docker runs inside WSL here. Docker Desktop isn't needed.

## 3. Ollama and the local models

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull gpt-oss:20b                     # Fast, the default answer option (13 GB)
ollama pull bge-m3                          # embeddings for search (1.2 GB)
ollama pull qwen3:1.7b                      # rewrites follow-up questions for search (1.4 GB)
```

Optional:

```bash
ollama pull gemma4:26b-a4b-it-qat           # Balanced (15 GB)
ollama pull qwen3.8:27b                     # writes the summary tree for overview questions (17 GB)
```

Without the summary model, ingestion still makes every document searchable; only the summarize stage fails, and the library pane says so.

## 4. The project

```bash
cd ~
git clone https://github.com/wesmail/study-assistant-share.git study-assistant
cd study-assistant
python3.14 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```

The requirements bring PyTorch with its CUDA libraries, and Node.js for the frontend (`nodejs-wheel`), so neither needs installing separately. They are several GB, so allow some minutes. Check the GPU from Python:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"     # True
```

## 5. The Docker images

Three images, each built or pulled once. The app then needs no network for them.

```bash
scripts/build_sandbox.sh                    # runs model-written code, SymPy, TeX, pandoc (about 1 GB)
scripts/build_office.sh                     # LibreOffice, for Word and PowerPoint files and decks (about 0.8 GB)
docker pull searxng/searxng:2026.10.19-81371861f
scripts/searxng.sh start                    # web search, on 127.0.0.1:8888 only
```

The sandbox build downloads TeX packages once, while the image is built, because containers never have a network. If a build fails partway, run it again: Docker keeps the finished layers.

## 6. Settings and keys

`config.yaml` holds the settings, over the defaults in `backend/app/config.py`. Check these for your machine:

- `office_fonts_dir: /mnt/c/Windows/Fonts`: decks render with your Windows fonts, mounted read-only into the office container. Remove the line to use the image's stand-in fonts.
- `backup_dir`: where daily backups go. It is unset, so backups are off until you set it: uncomment the example line and point it at a folder on another drive if you can, such as `/mnt/d/backups/study-assistant`.
- `hosted_monthly_limits`: your own Anthropic workspace limit, shown beside the month's spend.
- `backup_local_dir: ~/backups/study-assistant`: where the copy of `.env` goes, under `/home`.

For Deep and Qwen only, put your keys in `.env` in the project folder, readable only by you:

```bash
cat > .env <<'KEYS'
STUDY_ASSISTANT_ANTHROPIC_KEY=sk-ant-...
STUDY_ASSISTANT_OPENROUTER_KEY=sk-or-...
KEYS
chmod 600 .env
```

The app reads the keys only from this file, never from the environment, and never prints or logs them. An option without its key fails with a message naming the setting. Set a monthly spending limit in each provider's console: the app shows its spend, but the console's limit is the one that stops calls.

## 7. First run

Put PDFs or PowerPoint decks in `library/` (or add them later with the library pane's **Add PDF or deck…** button), then:

```bash
scripts/start.sh
```

Open http://localhost:8000 in a Windows browser.

- **The first start builds the frontend.** `start.sh` rebuilds it whenever its source changes.
- **The first document downloads the parser's models** from Hugging Face, about 4 GB. The first search downloads the reranker.
- **Ingestion.** The library pane shows each document's stage; a document is searchable once it's indexed.
- **Figure descriptions** use a hosted model, so they wait until you agree to the estimated cost in the library pane.
- **Ctrl+C** stops the server and the worker. A job that was running resumes at the next start.

Titles come from the file's metadata or its first page, never from its name. Correct one in the library pane, which writes `library/metadata.yaml`. To keep pages of a document out of search and away from hosted models (a CV slide, say), add `exclude_pages: [1]` under its file name in that file.

## 8. Check the install

```bash
scripts/smoke.sh
```

A free check, using no keys or hosted model and leaving your library and data alone. In a scratch folder it:
- makes a small PDF and a deck;
- ingests them with the real worker;
- asks Fast one question that must be answered from page 2 and cite it;
- runs Python and SymPy in the sandbox;
- checks SearXNG.

Each failure names its fix. Run it after the first start, since it needs the parser's models in the Hugging Face cache, and run it while nothing else is using the GPU.

## 9. Backups (optional)

```bash
scripts/backup.sh create                    # one now
scripts/backup.sh install-timer             # daily at 12:30, or at the next WSL start if missed
```

A backup holds what git can't recreate: the database (with your notes, highlights and suggestions), the parsed pages, the library files, `outputs/` and the gitignored eval grades. Archives go to `backup_dir` in `config.yaml`; the last 7 days are kept, plus one a week for 4 weeks. The timer runs at 12:30, or at the next WSL start if that was missed, and `scripts/start.sh` starts one when the newest is more than 20 hours old. It is safe while the app is running: the database is copied with SQLite's backup API.

`.env` is never in an archive. Its copy is in `~/backups/study-assistant/env/` (`backup_local_dir`), readable only by you. If it is lost, create new keys in the Anthropic and OpenRouter consoles.

```bash
scripts/backup.sh list                      # what there is
scripts/backup.sh verify                    # check the newest end to end
scripts/backup.sh copy /mnt/e/backups       # the newest to an external drive, checked there
git clone https://github.com/wesmail/study-assistant-share.git fresh && cd fresh && python3.14 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
scripts/backup.sh restore <backup_dir>/<archive>.tar.gz --to "$PWD" --env
```

A restore checks the whole archive before writing anything and never overwrites a file that differs. Ollama models, the Docker images and SearXNG aren't backed up: install them again as in steps 3 and 5.
