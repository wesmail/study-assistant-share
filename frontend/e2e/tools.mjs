// File cards in a real browser: `npm run e2e:tools -- <out dir>`. Local model only.
// Start the server, with Docker available, and a browser with --remote-debugging-port=9333.
import puppeteer from "puppeteer-core";

const out = process.argv[2] ?? ".";
const BROWSER = process.env.BROWSER_URL ?? "http://127.0.0.1:9333";
const APP = process.env.APP_URL ?? "http://localhost:8000";
const ANSWER_MS = 240_000;

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

async function ask(question) {
  await page.focus("textarea");
  for (const [index, line] of question.split("\n").entries()) {
    if (index > 0) {                    // a plain Enter would send it
      await page.keyboard.down("Shift");
      await page.keyboard.press("Enter");
      await page.keyboard.up("Shift");
    }
    await page.keyboard.type(line);
  }
  await page.keyboard.press("Enter");
  await page.waitForFunction(
    () => [...document.querySelectorAll("footer button")].some((button) => button.textContent === "Send"),
    { timeout: ANSWER_MS },
  );
}

const cardName = () => page.evaluate(() =>
  [...document.querySelectorAll('[data-testid="file-card"] p:first-child')].at(-1)?.textContent ?? null);

async function open() {
  await page.evaluate(() => {
    const card = [...document.querySelectorAll('[data-testid="file-card"]')].at(-1);
    [...card.querySelectorAll("button")].find((button) => button.textContent === "Preview")?.click();
  });
}

async function download() {
  return page.evaluate(async () => {
    const card = [...document.querySelectorAll('[data-testid="file-card"]')].at(-1);
    const response = await fetch(card.querySelector("a").href);
    return {
      status: response.status,
      type: response.headers.get("content-type"),
      disposition: response.headers.get("content-disposition"),
      bytes: (await response.arrayBuffer()).byteLength,
    };
  });
}

await page.goto(APP, { waitUntil: "networkidle2" });

await ask(["Run this code and plot.", "", "```python", "import matplotlib.pyplot as plt",
           "plt.plot([1, 3, 2])", "plt.title('Cards')", "plt.show()", "```"].join("\n"));
await page.waitForSelector('[data-testid="file-card"]', { timeout: 10_000 });
await page.waitForFunction(() => document.querySelector('[data-testid="file-preview"] img')?.complete, { timeout: 10_000 });
const width = await page.$eval('[data-testid="file-preview"] img', (image) => image.naturalWidth);
check("a plot appears by itself", width > 100, `${width} px wide`);
await page.screenshot({ path: `${out}/plot-card.png` });

await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((button) => button.textContent === "New chat").click();
});
await ask("What is the Cholesky decomposition, in two sentences?");
await ask("Export this answer as a PDF.");
check("a PDF gets a card", (await cardName())?.endsWith(".pdf"), (await cardName()) ?? "no card");
await open();
await page.waitForSelector('[data-testid="file-preview"] iframe', { timeout: 5_000 });
const exportNotes = await page.evaluate(() => {
  const answer = [...document.querySelectorAll('[data-testid="answer"]')].at(-1);
  return { warnings: answer.querySelectorAll('[data-testid="unverified"]').length,
           uncited: answer.querySelectorAll('[data-testid="uncited"]').length };
});
check("an export shows no unrelated warnings", exportNotes.warnings === 0 && exportNotes.uncited === 0,
      JSON.stringify(exportNotes));
const pdf = await download();
check("the PDF downloads", pdf.status === 200 && pdf.type === "application/pdf" && pdf.disposition.startsWith("attachment"),
      JSON.stringify(pdf));
await new Promise((resolve) => setTimeout(resolve, 2000));
await page.screenshot({ path: `${out}/pdf-card.png` });

await ask("Export this answer as a Word document.");
check("a Word file gets a card", (await cardName())?.endsWith(".docx"), (await cardName()) ?? "no card");
await open();
// Since roadmap item 7 the preview is LibreOffice's render. That the second export saves the
// answer (not a note about the first) is checked in e2e/office.mjs, on the source text its
// fallback shows.
const rendered = await page.waitForSelector('[data-testid="rendered-preview"]', { timeout: 60_000 })
  .then(() => true, () => false);
check("Word previews as its render", rendered);
const body = await page.evaluate(async () => {
  const card = [...document.querySelectorAll('[data-testid="file-card"]')].at(-1);
  const response = await fetch(`${card.querySelector("a").href.split("?")[0]}/preview`);
  return response.ok ? `${response.headers.get("content-type")}` : `HTTP ${response.status}`;
});
check("the Word preview is a PDF", body === "application/pdf", body);
const word = await download();
check("the Word file downloads", word.status === 200 && word.bytes > 1000, JSON.stringify(word));
await page.screenshot({ path: `${out}/word-card.png` });

check("no errors in the browser console", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await page.close();
await browser.disconnect();
console.log(problems.length ? `FAILED: ${problems.join(", ")}` : "All checks passed");
process.exit(problems.length ? 1 : 0);
