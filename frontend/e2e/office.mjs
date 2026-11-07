// Roadmap item 7 in the running app: `npm run e2e:office -- <out dir>`. Local model only (Fast).
// Start a server on a copy of the data (never the real one) with the decks ingested, Docker and
// the office image available, and a browser with --remote-debugging-port=9333.
//
//   1. Word and PowerPoint cards preview as LibreOffice renders them (a PDF in a frame).
//   2. When rendering fails (the request answered 422 here), the card shows the source text,
//      labelled with the reason and that the layout differs.
//   3. A citation to a deck opens the deck's render at the cited slide.
//   4. A highlight and a note made on deck slides (by eval/office/deck_backup_check.py on the
//      same copy) are drawn on their slides.
import puppeteer from "puppeteer-core";

const out = process.argv[2] ?? ".";
const BROWSER = process.env.BROWSER_URL ?? "http://127.0.0.1:9333";
const APP = process.env.APP_URL ?? "http://localhost:8001";
const ANSWER_MS = 300_000;
const DPG = "Forecasting Ground Motion with Large Sequence Models: Implications for the Einstein Telescope";
const ALBERTA = "Machine Learning in Hadron Physics and Σ⁰ Hyperon Production at HADES";

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
  if (message.type() === "error" && !message.text().includes("422")) consoleErrors.push(message.text());
});

async function ask(question) {
  await page.focus("textarea");
  await page.keyboard.type(question);
  await page.keyboard.press("Enter");
  await page.waitForFunction(
    () => [...document.querySelectorAll("footer button")].some((button) => button.textContent === "Send"),
    { timeout: ANSWER_MS },
  );
}

const lastCard = () => page.evaluate(() =>
  [...document.querySelectorAll('[data-testid="file-card"] p:first-child')].at(-1)?.textContent ?? null);

async function togglePreview() {
  await page.evaluate(() => {
    const card = [...document.querySelectorAll('[data-testid="file-card"]')].at(-1);
    [...card.querySelectorAll("button")].find((b) => ["Preview", "Hide"].includes(b.textContent))?.click();
  });
}

async function renderedPreview(label) {
  await togglePreview();
  const ok = await page.waitForFunction(() => {
    const frame = [...document.querySelectorAll('[data-testid="rendered-preview"]')].at(-1);
    return frame && frame.src.startsWith("blob:");
  }, { timeout: 60_000 }).then(() => true, () => false);
  const served = await page.evaluate(async () => {
    const card = [...document.querySelectorAll('[data-testid="file-card"]')].at(-1);
    const response = await fetch(`${card.querySelector("a").href.split("?")[0]}/preview`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    return { status: response.status, type: response.headers.get("content-type"),
             pdf: String.fromCharCode(...bytes.slice(0, 5)) };
  });
  check(`${label} previews as its render`, ok && served.status === 200 && served.type === "application/pdf"
        && served.pdf === "%PDF-", JSON.stringify(served));
  await new Promise((resolve) => setTimeout(resolve, 2500));
  await page.screenshot({ path: `${out}/${label.replace(/\W+/g, "-")}.png` });
}

await page.goto(APP, { waitUntil: "networkidle2" });
await page.waitForSelector('[data-testid="mode-fast"]', { timeout: 30_000 });

// 1 and 2: previews and the fallback.
await ask("What is the Cholesky decomposition, in two sentences?");
await ask("Export this answer as a Word document.");
check("a Word file gets a card", (await lastCard())?.endsWith(".docx"), (await lastCard()) ?? "no card");
await renderedPreview("the Word file");
await togglePreview();      // hide

await page.setRequestInterception(true);
const refuse = (request) => {
  if (request.url().endsWith("/preview")) {
    void request.respond({ status: 422, contentType: "application/json",
                           body: JSON.stringify({ detail: "test: LibreOffice stopped" }) });
  } else {
    void request.continue();
  }
};
page.on("request", refuse);
await togglePreview();
const fallback = await page.waitForFunction(() => {
  const preview = [...document.querySelectorAll('[data-testid="file-preview"]')].at(-1);
  return preview && preview.textContent.includes("LibreOffice couldn't render it") ? preview.textContent : null;
}, { timeout: 20_000 }).then((handle) => handle.jsonValue(), () => "");
check("a failed render falls back to the labelled source text",
      fallback.includes("(test: LibreOffice stopped)") && fallback.includes("layout in the file itself differs")
      && /triangular|factor/i.test(fallback), fallback.slice(0, 160));
page.off("request", refuse);
await page.setRequestInterception(false);
await togglePreview();

await ask("Make a short PowerPoint deck of two slides from this answer.");
check("a deck gets a card", (await lastCard())?.endsWith(".pptx"), (await lastCard()) ?? "no card");
await renderedPreview("the deck");

// 3: a citation to a deck opens the cited slide.
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((button) => button.textContent === "New chat").click();
});
await ask("In my DPG talk on SeismoGPT, how many parameters do the single-station and the array models have?");
const cited = await page.evaluate(() => {
  const answer = [...document.querySelectorAll('[data-testid="answer"]')].at(-1);
  return { text: answer.textContent.slice(0, 300), citations: answer.querySelectorAll('[data-testid="citation"]').length };
});
console.log("deck answer", JSON.stringify(cited));
// Every citation to the deck: the viewer must open the deck at that source's own slide.
const targets = await page.evaluate(() => {
  const answer = [...document.querySelectorAll('[data-testid="answer"]')].at(-1);
  const sources = Object.fromEntries([...answer.querySelectorAll('[data-testid="source"]')]
    .map((node) => [node.querySelector("span").textContent.trim(), node.textContent]));
  return [...answer.querySelectorAll('[data-testid="citation"]')].map((link, index) => {
    const number = /#cite-(\d+)/.exec(link.getAttribute("href"))[1];
    const label = sources[`S${number}`] ?? "";
    return { index, number, label, page: /pp?\. (\d+)/.exec(label)?.[1] ?? null };
  });
});
console.log("citations", JSON.stringify(targets.map(({ number, page, label }) => ({ number, page, label: label.slice(0, 90) }))));
const onDeck = targets.filter((target) => target.label.includes("Einstein Telescope") && target.page);
check("the answer cites the deck", onDeck.length > 0, `${targets.length} citations`);
for (const target of onDeck.filter((t, i, all) => all.findIndex((o) => o.number === t.number) === i)) {
  await page.evaluate((index) => {
    const answer = [...document.querySelectorAll('[data-testid="answer"]')].at(-1);
    answer.querySelectorAll('[data-testid="citation"]')[index].click();
  }, target.index);
  await page.waitForFunction((wanted) => document.querySelector('[data-testid="page-input"]')?.value === wanted
    && document.querySelector('[data-testid="pdf-canvas"]')?.width > 0, { timeout: 30_000 }, target.page)
    .catch(() => undefined);
  await new Promise((resolve) => setTimeout(resolve, 1500));
  const where = await page.evaluate(() => ({
    page: document.querySelector('[data-testid="page-input"]')?.value,
    regions: document.querySelectorAll('[data-testid="highlight"]').length,
  }));
  check(`citation S${target.number} opens the deck at slide ${target.page}`, where.page === target.page, JSON.stringify(where));
  await page.screenshot({ path: `${out}/deck-citation-S${target.number}.png` });
}

// 4: a highlight and a note on deck slides.
async function openAt(title, number) {
  await page.evaluate(() => document.querySelector(".textLayer .endOfContent")?.remove());
  await page.click(`button[title="${title}"]`);
  await page.waitForFunction(() => document.querySelector('[data-testid="page-input"]')
    && document.querySelector(".textLayer .endOfContent"), { timeout: 30_000 });
  await page.evaluate(() => document.querySelector(".textLayer .endOfContent")?.remove());
  await page.click('[data-testid="page-input"]', { clickCount: 3 });
  await page.keyboard.type(String(number));
  await page.keyboard.press("Enter");
  await page.waitForFunction((wanted) => document.querySelector('[data-testid="page-input"]').value === String(wanted)
    && document.querySelector(".textLayer .endOfContent"), { timeout: 30_000 }, number);
  await new Promise((resolve) => setTimeout(resolve, 1500));
}
await openAt(DPG, 12);
const onTwelve = await page.$$eval('[data-testid="saved-highlight"]', (nodes) => nodes.length);
check("the highlight is drawn on DPG slide 12", onTwelve > 0, `${onTwelve} boxes`);
await page.screenshot({ path: `${out}/deck-highlight.png` });
await openAt(ALBERTA, 19);
const markers = await page.$$eval('[data-testid="note-marker"]', (nodes) => nodes.length);
check("the note's marker is on Alberta slide 19", markers > 0, `${markers} markers`);
await page.screenshot({ path: `${out}/deck-note.png` });

check("no errors in the browser console", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await page.close();
await browser.disconnect();
console.log(problems.length ? `FAILED: ${problems.join(", ")}` : "All checks passed");
process.exit(problems.length ? 1 : 0);
