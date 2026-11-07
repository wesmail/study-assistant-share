# Contributing

Thank you for helping. This is a research prototype maintained by one person, so small, well-tested changes are the easiest to take in.

## The workflow

1. **Work on a branch, and open a pull request.** Never push to `main`, even if GitHub lets you.

   ```bash
   git switch -c fix-short-description
   # change, commit, then:
   git push -u origin fix-short-description
   ```

   Then open a pull request on GitHub against `main`.
2. **Run the checks before you open it.** From the project folder:

   ```bash
   export PATH="$PWD/.venv/bin:$PATH"           # node and npm come with the Python packages
   (cd backend && python -m pytest -q)          # backend; the Docker tests need the images (INSTALL.md, step 5)
   npm --prefix frontend ci                     # once, if frontend/node_modules is missing
   npm --prefix frontend test                   # frontend
   scripts/smoke.sh                             # the free end-to-end check, with nothing else using the GPU
   ```

   All three must pass. If a test fails for a reason unrelated to your change, say so in the pull request.
3. **Describe the change.** In the pull request, write:
   - what changed, and why;
   - how you tested it, beyond the checks above: a new test, a run in the app, a screenshot;
   - **measurements** when a change could move answer quality, speed or cost (a prompt, a model, search, chunking). Give before and after numbers on the same questions, and say how many questions and how they were graded.
4. **Keep it focused.** One change per pull request. Update the docs (README.md, INSTALL.md, ARCHITECTURE.md, TROUBLESHOOTING.md) when what a user sees or does changes.

## Never commit

- **Books or other copyrighted text**: no PDFs or decks, and no passages, page text or figure captions copied from them, not even in tests, fixtures or eval results. `library/` and `data/` are git-ignored for this reason. Write your own example text for tests.
- **Keys**: `.env` stays out of git. Never paste a key into code, a test, an issue or a pull request. If one slips out, revoke it in its console at once.
- **Personal paths and data**: no `/home/<you>/…` or `C:\Users\<you>\…` in code or tests (compute paths from the project folder). No one else's names, e-mail addresses, CVs or private documents.
- **Your own `library/metadata.yaml` or `config.yaml` settings**: the repository keeps the template and the defaults.

## Licence

The project is under the GNU Affero General Public License v3.0 ([LICENSE](LICENSE)). By opening a pull request, you agree that your contribution is released under the same licence.

## Reporting bugs

Open a GitHub issue with the **Bug report** template. It asks for:
- the steps;
- what you expected, and what happened;
- the answer mode;
- your setup;
- what `scripts/smoke.sh` printed.

Leave out keys, and text from books you can't share.
