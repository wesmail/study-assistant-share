// Highlights in the running app (roadmap item 4, gate item 1): `npm run e2e:annotations`.
// Start a server on a copy of the data (never the real one) and a browser with
// --remote-debugging-port=9333 first.
//
//   PHASE=select   makes each selection in eval/annotations/selections.json with the mouse,
//                  a drag from its first word to its last, clicks Highlight, and saves what the
//                  browser sent and what the server stored to $OUT/selections-result.json.
//   PHASE=reload   after a reload: every stored highlight is still listed and drawn on its page.
//
// Score with: .venv/bin/python eval/annotations/score_selections.py $OUT/selections-result.json
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import puppeteer from "puppeteer-core";

const BROWSER = process.env.BROWSER_URL ?? "http://127.0.0.1:9333";
const APP = process.env.APP_URL ?? "http://localhost:8001";
const PHASE = process.env.PHASE ?? "select";
// SET=selections (the gate set) or SET=selections-heldout, or PLAN=path.
const PLAN = process.env.PLAN ?? new URL(`../../eval/annotations/${process.env.SET ?? "selections"}.json`, import.meta.url).pathname;
const OUT = process.env.OUT ?? "e2e-annotations";
const RESULT = `${OUT}/selections-result.json`;
const NAME = PLAN.split("/").pop().replace(/\.json$/, "");

mkdirSync(OUT, { recursive: true });
const plan = JSON.parse(readFileSync(PLAN, "utf-8")).selections;
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
await page.goto(APP, { waitUntil: "networkidle2" });
await page.waitForSelector('[data-testid="mode-fast"]', { timeout: 30_000 });
const titles = Object.fromEntries((await page.evaluate(async () =>
  (await (await fetch("/api/documents")).json()).documents.map((d) => [d.id, d.title]))));

let openDoc = null;
let openPageNumber = null;

/** Open a document at a page and wait until that page's text layer is drawn. */
async function show(docId, pageNumber) {
  if (openDoc !== docId) {
    await page.evaluate(() => document.querySelector('[data-testid="text-layer"] .endOfContent')?.remove());
    await page.click(`button[title="${titles[docId].replaceAll('"', '\\"')}"]`);
    await page.waitForFunction(() => document.querySelector('[data-testid="page-input"]')
      && document.querySelector('[data-testid="text-layer"] .endOfContent'), { timeout: 30_000 });
    openDoc = docId;
    openPageNumber = 1;
  }
  if (openPageNumber !== pageNumber) {
    await page.evaluate(() => document.querySelector('[data-testid="text-layer"] .endOfContent')?.remove());
    await page.click('[data-testid="page-input"]', { clickCount: 3 });
    await page.keyboard.type(String(pageNumber));
    await page.keyboard.press("Enter");
    await page.waitForFunction((wanted) =>
      document.querySelector('[data-testid="page-input"]').value === String(wanted)
        && document.querySelector('[data-testid="text-layer"] .endOfContent'),
    { timeout: 30_000 }, pageNumber);
    openPageNumber = pageNumber;
  }
}

/** A point on the page in PDF points, scrolled into view, in screen pixels. */
async function onScreen([x, y]) {
  return page.evaluate((px, py) => {
    const layer = document.querySelector('[data-testid="text-layer"]');
    const area = layer.parentElement.parentElement;
    const scale = Number(layer.style.getPropertyValue("--total-scale-factor"));
    let box = layer.getBoundingClientRect();
    const areaBox = area.getBoundingClientRect();
    area.scrollTop += box.top + py * scale - areaBox.top - area.clientHeight / 2;
    box = layer.getBoundingClientRect();
    return { x: box.left + px * scale, y: box.top + py * scale, scale };
  }, x, y);
}

async function select(selection) {
  await show(selection.doc_id, selection.page);
  await page.evaluate(() => window.getSelection().removeAllRanges());
  const start = await onScreen(selection.start);
  const end = await onScreen(selection.end);
  // Scrolling for the end may have moved the start: both are read after the last scroll.
  const startNow = await onScreen(selection.start);
  const endNow = await page.evaluate(([px, py]) => {
    const layer = document.querySelector('[data-testid="text-layer"]');
    const scale = Number(layer.style.getPropertyValue("--total-scale-factor"));
    const box = layer.getBoundingClientRect();
    return { x: box.left + px * scale, y: box.top + py * scale };
  }, selection.end);
  const drag = async (nudge) => {
    await page.mouse.move(startNow.x - nudge, startNow.y);
    await page.mouse.down();
    await page.mouse.move(endNow.x, endNow.y, { steps: 12 });
    await page.mouse.up();
    return page.waitForSelector('[data-testid="selection-toolbar"]', { timeout: 3_000 }).catch(() => null);
  };
  // At some sub-pixel start points Chromium puts a caret in pdf.js's scaled text instead of
  // starting a selection (reproduced with raw CDP events). A reader would just drag again:
  // so does this, once, 1 px to the left, and it's counted.
  let retried = false;
  let toolbar = await drag(0);
  if (!toolbar) {
    retried = true;
    await page.evaluate(() => window.getSelection().removeAllRanges());
    toolbar = await drag(1);
  }
  const selected = await page.evaluate(() => window.getSelection().toString());
  if (!toolbar) {
    await page.screenshot({ path: `${OUT}/${selection.id}-no-toolbar.png` });
    return { id: selection.id, error: "no toolbar", retried, selected, start: startNow, end: endNow };
  }
  const posted = page.waitForResponse((response) =>
    response.url().endsWith("/api/annotations") && response.request().method() === "POST", { timeout: 10_000 });
  await page.click('[data-testid="tool-highlight"]');
  const response = await posted;
  const sent = JSON.parse(response.request().postData());
  const stored = await response.json();
  await page.waitForSelector(`[data-annotation="${stored.id}"]`, { timeout: 5_000 });
  if (Number(selection.id.slice(1)) % 5 === 1) {
    await page.screenshot({ path: `${OUT}/${selection.id}.png` });
  }
  return { id: selection.id, retried, selected, sent, stored, scale: startNow.scale };
}

if (PHASE === "select") {
  const results = [];
  for (const selection of plan) {
    const result = await select(selection);
    results.push(result);
    check(`${selection.id} doc ${selection.doc_id} p. ${selection.page} ${selection.kind}`,
      Boolean(result.stored), result.stored ? `snapped=${result.stored.snapped}${result.retried ? " (dragged twice)" : ""} "${result.stored.quote.slice(0, 60)}"`
        : `${result.error}: selected "${result.selected.slice(0, 80)}" ${JSON.stringify([result.start, result.end])}`);
  }
  writeFileSync(RESULT, JSON.stringify({ app: APP, plan: NAME, results }, null, 1));
  console.log(`saved ${RESULT}`);
} else {
  const { results } = JSON.parse(readFileSync(RESULT, "utf-8"));
  await page.reload({ waitUntil: "networkidle2" });
  await page.waitForSelector('[data-testid="mode-fast"]', { timeout: 30_000 });
  openDoc = null;
  for (const result of results.filter((r) => r.stored)) {
    const { id, doc_id: docId, page: pageNumber } = result.stored;
    const listed = await page.evaluate(async (doc) =>
      (await (await fetch(`/api/annotations?doc_id=${doc}`)).json()).annotations.map((a) => a.id), docId);
    await show(docId, pageNumber);
    const drawn = await page.$$eval(`[data-annotation="${id}"]`, (nodes) => nodes.length);
    check(`${result.id} still saved and drawn after a reload`, listed.includes(id) && drawn === result.stored.rects.length,
      `listed=${listed.includes(id)} drawn=${drawn}/${result.stored.rects.length}`);
  }
}

check("no errors in the browser console", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await page.close();
await browser.disconnect();
console.log(problems.length ? `${problems.length} problem(s)` : "all ok");
process.exit(problems.length ? 1 : 0);
