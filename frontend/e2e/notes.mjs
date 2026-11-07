// Notes in the running app (roadmap item 4, step 2): `npm run e2e:notes`.
// Start a server on a copy of the data (never the real one) and a browser with
// --remote-debugging-port=9333 first; the copy should have no notes yet.
//
//   PHASE=write    a note from a selection (Markdown, LaTeX, tags, previewed), its marker on the
//                  page opening it, a note from a highlight, a free note, an edit, the filter,
//                  and the Obsidian export saved to $OUT/study-notes.zip.
//   PHASE=reload   after a reload: the three notes are still there, and the markers drawn.
//
// Check the export with: .venv/bin/python eval/annotations/check_export.py $OUT/study-notes.zip
import { mkdirSync, writeFileSync } from "node:fs";
import puppeteer from "puppeteer-core";

const BROWSER = process.env.BROWSER_URL ?? "http://127.0.0.1:9333";
const APP = process.env.APP_URL ?? "http://localhost:8001";
const PHASE = process.env.PHASE ?? "write";
const OUT = process.env.OUT ?? "e2e-notes";
const DOC = "Understanding Deep Learning";
const PAGE = 60;

mkdirSync(OUT, { recursive: true });
const problems = [];
const check = (label, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label} ${detail}`);
  if (!ok) problems.push(label);
};

const browser = await puppeteer.connect({ browserURL: BROWSER, defaultViewport: { width: 1600, height: 1000 } });
const page = await browser.newPage();
const consoleErrors = [];
page.on("pageerror", (error) => consoleErrors.push(String(error)));
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("dialog", (dialog) => void dialog.accept());
await page.goto(APP, { waitUntil: "networkidle2" });
await page.waitForSelector('[data-testid="mode-fast"]', { timeout: 30_000 });

async function openPage() {
  await page.evaluate(() => document.querySelector('[data-testid="text-layer"] .endOfContent')?.remove());
  await page.click(`button[title="${DOC}"]`);
  await page.waitForFunction(() => document.querySelector('[data-testid="page-input"]')
    && document.querySelector('[data-testid="text-layer"] .endOfContent'), { timeout: 30_000 });
  await page.evaluate(() => document.querySelector('[data-testid="text-layer"] .endOfContent')?.remove());
  await page.click('[data-testid="page-input"]', { clickCount: 3 });
  await page.keyboard.type(String(PAGE));
  await page.keyboard.press("Enter");
  await page.waitForFunction((wanted) => document.querySelector('[data-testid="page-input"]').value === String(wanted)
    && document.querySelector('[data-testid="text-layer"] .endOfContent'), { timeout: 30_000 }, PAGE);
}

/** Drag across the n-th long text run on the page; dragged again 1 px over if Chromium only set a caret. */
async function selectRun(n) {
  const box = await page.evaluate((index) => {
    const spans = [...document.querySelectorAll(".textLayer span")].filter((s) => s.textContent.length > 40);
    const r = spans[index].getBoundingClientRect();
    return { x0: r.left + 2, x1: r.right - 2, y: (r.top + r.bottom) / 2 };
  }, n);
  for (const nudge of [0, 1]) {
    await page.evaluate(() => window.getSelection().removeAllRanges());
    await page.mouse.move(box.x0 - nudge, box.y);
    await page.mouse.down();
    await page.mouse.move(box.x1, box.y, { steps: 10 });
    await page.mouse.up();
    if (await page.waitForSelector('[data-testid="selection-toolbar"]', { timeout: 3_000 }).catch(() => null)) return box;
  }
  throw new Error(`no selection on run ${n}`);
}

async function writeNote(text, tags) {
  await page.waitForSelector('[data-testid="note-text"]');
  await page.type('[data-testid="note-text"]', text);
  await page.type('[data-testid="note-tags"]', tags);
}

async function saved() {
  return page.waitForResponse((r) => r.url().endsWith("/api/notes") && r.request().method() === "POST", { timeout: 10_000 })
    .then((r) => r.json());
}

const countNotes = () => page.$$eval('[data-testid="note"]', (nodes) => nodes.length);

if (PHASE === "write") {
  await openPage();

  // 1. A note for a selected passage.
  await selectRun(6);
  await page.click('[data-testid="tool-note"]');
  await page.waitForSelector('[data-testid="note-draft"]');
  const draftQuote = await page.$eval('[data-testid="note-draft"] blockquote', (n) => n.textContent);
  check("Add note opens the notes with the passage", draftQuote.length > 20, `"${draftQuote.slice(0, 60)}"`);
  await writeNote("The **toy model** has eight parameters; the gradient of one term is\n\n"
    + "$$\\frac{\\partial \\ell_i}{\\partial \\beta_0} = -2\\left(f[x_i, \\phi] - y_i\\right)\\omega_3 \\cos[\\cdot]$$\n\n"
    + "and inline $\\phi = \\{\\beta_0, \\omega_0, \\dots\\}$.", "backprop toy-model");
  check("the preview renders the maths", await page.$$eval('[data-testid="note-preview"] .katex', (n) => n.length) >= 2);
  const first = saved();
  await page.keyboard.down("Control");
  await page.keyboard.press("Enter");
  await page.keyboard.up("Control");
  const note1 = await first;
  check("saved, snapped to the page's words", note1.snapped === true, `quote "${note1.quote.slice(0, 60)}"`);
  await page.waitForSelector(`[data-testid="note"][data-note="${note1.id}"] .katex`);
  check("the note shows its maths and tags", await page.$eval(`[data-testid="note"][data-note="${note1.id}"]`,
    (n) => n.textContent.includes("#backprop") && n.textContent.includes("#toy-model")));
  await page.waitForSelector(`[data-testid="note-marker"][data-note="${note1.id}"]`);
  check("its marker and underline are on the page", await page.$$eval(`[data-testid="note-underline"][data-note="${note1.id}"]`,
    (n) => n.length) === note1.rects.length);
  await page.screenshot({ path: `${OUT}/notes-1-saved.png` });

  // 2. The marker opens it.
  await page.click('[data-testid="notes-back"]');
  await page.click(`[data-testid="note-marker"][data-note="${note1.id}"]`);
  await page.waitForSelector(`[data-testid="note"][data-note="${note1.id}"].border-teal-700`);
  check("the marker opens the note in the panel", true);

  // 3. A note for a highlight.
  await selectRun(9);
  const highlighted = page.waitForResponse((r) => r.url().endsWith("/api/annotations") && r.request().method() === "POST");
  await page.click('[data-testid="tool-highlight"]');
  const annotation = await (await highlighted).json();
  await page.waitForSelector(`[data-annotation="${annotation.id}"]`);
  const spot = await page.evaluate((id) => {
    const r = document.querySelector(`[data-annotation="${id}"]`).getBoundingClientRect();
    return { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
  }, annotation.id);
  await page.mouse.click(spot.x, spot.y);
  await page.waitForSelector('[data-testid="highlight-note"]');
  await page.click('[data-testid="highlight-note"]');
  await writeNote("Why the squared loss: maximum likelihood under Gaussian noise.", "loss");
  const second = saved();
  await page.click('[data-testid="note-save"]');
  const note2 = await second;
  check("a highlight's note gets the highlight's passage", note2.quote === annotation.quote && note2.page === PAGE,
    `"${note2.quote.slice(0, 50)}"`);

  // 4. A free note.
  await page.click('[data-testid="new-note"]');
  await writeNote("Exam: revise chapters 7 and 8; derive $\\nabla_\\phi L$ by hand once.", "exam");
  const third = saved();
  await page.click('[data-testid="note-save"]');
  const note3 = await third;
  check("a note without a passage", note3.doc_id === null && note3.page === null);
  await page.waitForFunction(() => document.querySelectorAll('[data-testid="note"]').length === 3);

  // 5. An edit.
  await page.click(`[data-testid="note"][data-note="${note2.id}"] [data-testid="note-edit"]`);
  await page.click(`[data-testid="note"][data-note="${note2.id}"] [data-testid="note-text"]`);
  await page.keyboard.press("End");
  await page.keyboard.type(" See eq. (7.6).");
  const changed = page.waitForResponse((r) => r.url().endsWith(`/api/notes/${note2.id}`) && r.request().method() === "PATCH");
  await page.click(`[data-testid="note"][data-note="${note2.id}"] [data-testid="note-save"]`);
  check("edited", (await (await changed).json()).body_md.endsWith("See eq. (7.6)."));

  // 6. The filter.
  await page.type('[data-testid="notes-filter"]', "exam");
  check("the filter finds the free note", await countNotes() === 1);
  await page.click('[data-testid="notes-filter"]');
  await page.keyboard.down("Control");
  await page.keyboard.press("KeyA");
  await page.keyboard.up("Control");
  await page.keyboard.press("Backspace");
  await page.click('[data-testid="notes-this-document"]');
  check("This document leaves out the free note", await countNotes() === 2);
  await page.screenshot({ path: `${OUT}/notes-2-panel.png` });

  // 7. The export.
  const href = await page.$eval('[data-testid="export-notes"]', (a) => a.getAttribute("href"));
  const all = await page.evaluate(async () => {
    const response = await fetch("/api/notes/export");
    return { type: response.headers.get("content-type"), bytes: [...new Uint8Array(await response.arrayBuffer())] };
  });
  writeFileSync(`${OUT}/study-notes.zip`, Buffer.from(all.bytes));
  check("the export is a zip", all.type === "application/zip", `${all.bytes.length} bytes; this document's link ${href}`);
  writeFileSync(`${OUT}/notes.json`, JSON.stringify({ ids: [note1.id, note2.id, note3.id], page: PAGE }));
} else {
  const { readFileSync } = await import("node:fs");
  const { ids } = JSON.parse(readFileSync(`${OUT}/notes.json`, "utf-8"));
  await page.click('[data-testid="open-notes"]');
  await page.waitForFunction(() => document.querySelectorAll('[data-testid="note"]').length > 0, { timeout: 10_000 });
  const listed = await page.$$eval('[data-testid="note"]', (nodes) => nodes.map((n) => Number(n.dataset.note)));
  check("all three notes are there after a reload", ids.every((id) => listed.includes(id)), JSON.stringify(listed));
  await openPage();
  const markers = await page.$$eval('[data-testid="note-marker"]', (nodes) => nodes.map((n) => Number(n.dataset.note)));
  check("both passage notes have their markers", ids.slice(0, 2).every((id) => markers.includes(id)), JSON.stringify(markers));
}

check("no errors in the browser console", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await page.close();
await browser.disconnect();
console.log(problems.length ? `${problems.length} problem(s)` : "all ok");
process.exit(problems.length ? 1 : 0);
