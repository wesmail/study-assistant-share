// Suggest, the annotated copy and Explain in the running app (roadmap item 4, steps 5-6):
// `npm run e2e:suggest`. Start a server on a copy of the data first, whose config lets Balanced
// suggest (suggest_options: [balanced, deep]) so that nothing is spent, and a browser with
// --remote-debugging-port=9333. Uses the local model: run nothing else on the GPU meanwhile.
import { mkdirSync } from "node:fs";
import puppeteer from "puppeteer-core";

const BROWSER = process.env.BROWSER_URL ?? "http://127.0.0.1:9333";
const APP = process.env.APP_URL ?? "http://localhost:8001";
const OUT = process.env.OUT ?? "e2e-suggest";
const DOC = "Understanding Deep Learning";
const PAGE = 60;
const MODEL_MS = 600_000;
const ONLY = process.env.ONLY ?? "all";      // "explain": only step 4
// UDL p. 60, eq. (7.6), as the parser stored its box (PDF points from the top left).
const EQUATION = [281.52, 388.08, 515.30, 400.75];

async function goTo(number) {
  if (await page.$eval('[data-testid="page-input"]', (n) => n.value) === String(number)) return;
  await page.evaluate(() => document.querySelector(".textLayer .endOfContent")?.remove());
  await page.click('[data-testid="page-input"]', { clickCount: 3 });
  await page.keyboard.type(String(number));
  await page.keyboard.press("Enter");
  await page.waitForFunction((wanted) => document.querySelector('[data-testid="page-input"]').value === String(wanted)
    && document.querySelector(".textLayer .endOfContent"), { timeout: 30_000 }, number);
}

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
await page.goto(APP, { waitUntil: "networkidle2" });
await page.waitForSelector('[data-testid="mode-balanced"]', { timeout: 30_000 });
await page.click('[data-testid="mode-balanced"]');

await page.click(`button[title="${DOC}"]`);
await page.waitForFunction(() => document.querySelector('[data-testid="page-input"]')
  && document.querySelector(".textLayer .endOfContent"), { timeout: 30_000 });
await page.evaluate(() => document.querySelector(".textLayer .endOfContent")?.remove());
await page.click('[data-testid="page-input"]', { clickCount: 3 });
await page.keyboard.type(String(PAGE));
await page.keyboard.press("Enter");
await page.waitForFunction((wanted) => document.querySelector('[data-testid="page-input"]').value === String(wanted)
  && document.querySelector(".textLayer .endOfContent"), { timeout: 30_000 }, PAGE);

if (ONLY !== "explain") {
// 1. Suggest with Balanced, on this page.
await page.click('[data-testid="open-suggest"]');
await page.waitForSelector('[data-testid="suggest-panel"]');
await page.select('[data-testid="suggest-panel"] select', "balanced");
await page.type('[data-testid="suggest-instruction"]', "Highlight the key definitions and results a student should remember.");
const asked = page.waitForResponse((r) => r.url().endsWith("/api/suggestions"), { timeout: MODEL_MS });
await page.click('[data-testid="suggest-run"]');
const result = await (await asked).json();
await page.waitForSelector('[data-testid="suggest-report"]', { timeout: 10_000 });
const made = result.suggestions?.length ?? 0;
check("Balanced suggested highlights, found in the text", made > 0,
  `${made} of ${result.asked} quotes found; report: ${await page.$eval('[data-testid="suggest-report"]', (n) => n.textContent)}`);
const dashed = await page.$$eval('[data-testid="saved-highlight"][data-status="suggested"]', (n) => n.length);
check("they are drawn as suggestions on the page", dashed > 0, `${dashed} boxes`);
await page.screenshot({ path: `${OUT}/1-suggested.png` });

// 2. Accept one from its menu, reject one from the list, accept the rest.
if (made >= 2) {
  const first = result.suggestions.find((s) => s.page === PAGE) ?? result.suggestions[0];
  const spot = await page.evaluate((id) => {
    const r = document.querySelector(`[data-annotation="${id}"]`)?.getBoundingClientRect();
    return r ? { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 } : null;
  }, first.id);
  if (spot) {
    await page.mouse.click(spot.x, spot.y);
    await page.waitForSelector('[data-testid="highlight-accept"]', { timeout: 5_000 });
    check("a suggestion's menu shows its note", Boolean(await page.$('[data-testid="highlight-body"]')));
    await page.click('[data-testid="highlight-accept"]');
    await page.waitForFunction((id) => document.querySelector(`[data-annotation="${id}"]`)?.dataset.status === "accepted",
      { timeout: 5_000 }, first.id);
    check("accepted from its menu", true);
  }
  const before = await page.$$eval('[data-testid="suggestion"]', (n) => n.length);
  await page.click('[data-testid="suggestion-reject"]');
  await page.waitForFunction((n) => document.querySelectorAll('[data-testid="suggestion"]').length === n - 1, { timeout: 5_000 }, before);
  check("rejected from the list", true);
  if (before > 1) {
    await page.click('[data-testid="accept-all"]');
    await page.waitForFunction(() => document.querySelectorAll('[data-testid="suggestion"]').length === 0, { timeout: 5_000 });
    check("the rest accepted at once", true);
  }
}

// 3. The annotated copy.
const exported = page.waitForResponse((r) => r.url().includes("/api/annotations/export/"));
await page.click('[data-testid="make-copy"]');
const copy = await (await exported).json();
await page.waitForSelector('[data-testid="copy-download"]');
const fetched = await page.evaluate(async (url) => {
  const response = await fetch(url);
  return { type: response.headers.get("content-type"), disposition: response.headers.get("content-disposition"),
    size: (await response.arrayBuffer()).byteLength };
}, await page.$eval('[data-testid="copy-download"]', (a) => a.getAttribute("href")));
check("the annotated copy is made and downloads", copy.highlights > 0 && fetched.type === "application/pdf"
  && fetched.disposition.startsWith("attachment"), `${copy.highlights} highlights, ${fetched.size} bytes`);

await page.click('[data-testid="open-suggest"]');
}

// 4. Explain an equation selected on the page.
await goTo(PAGE);
const box = await page.evaluate(([x0, y0, x1, y1]) => {
  const layer = document.querySelector(".textLayer");
  const scale = Number(layer.style.getPropertyValue("--total-scale-factor"));
  const r = layer.getBoundingClientRect();
  return { x0: r.left + x0 * scale + 1, x1: r.left + (x1 - 40) * scale, y: r.top + ((y0 + y1) / 2) * scale };
}, EQUATION);
for (const nudge of [0, 1]) {
  await page.mouse.move(box.x0 - nudge, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x1, box.y, { steps: 10 });
  await page.mouse.up();
  if (await page.waitForSelector('[data-testid="selection-toolbar"]', { timeout: 3_000 }).catch(() => null)) break;
}
const explained = page.waitForResponse((r) => r.url().endsWith("/api/explain"), { timeout: MODEL_MS });
await page.click('[data-testid="tool-explain"]');
await page.waitForSelector('[data-testid="explain-panel"]');
const response = await explained;
const body = await response.json();
if (response.ok()) {
  await page.waitForSelector('[data-testid="explain-rows"]', { timeout: 10_000 });
  const rows = await page.$$eval('[data-testid="explain-span"]', (n) => n.length);
  check("Explain shows the equation in two columns, from the book's LaTeX", rows > 0 && body.from_book === true,
    `${rows} rows; text ${JSON.stringify(body.text).slice(0, 80)}`);
} else {
  check("Explain answered", false, JSON.stringify(body).slice(0, 200));
}
await page.screenshot({ path: `${OUT}/2-explain.png` });
await page.click('[data-testid="explain-close"]');
await page.waitForSelector('[data-testid="explanation-links"] button');
await page.click('[data-testid="explanation-links"] button');
check("the chat's link opens it again", Boolean(await page.waitForSelector('[data-testid="explain-panel"]', { timeout: 5_000 })));

check("no errors in the browser console", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await page.close();
await browser.disconnect();
console.log(problems.length ? `${problems.length} problem(s)` : "all ok");
process.exit(problems.length ? 1 : 0);
