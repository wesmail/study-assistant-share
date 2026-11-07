// The feedback button in the running app (roadmap item 2): `npm run e2e:feedback`.
// Start the server first, and a browser with --remote-debugging-port=9333.
//
//   PHASE=rate   asks one Fast question and one Deep question with web search on (this
//                sends it to Anthropic and costs about $0.10-0.30), rates the first up and
//                the second down, each with a note, and makes the second a gold candidate.
//   PHASE=reload after a server restart: both ratings are still in the review list.
import puppeteer from "puppeteer-core";

const BROWSER = process.env.BROWSER_URL ?? "http://127.0.0.1:9333";
const APP = process.env.APP_URL ?? "http://localhost:8000";
const PHASE = process.env.PHASE ?? "rate";
const FAST_QUESTION = process.env.FAST_QUESTION ?? "How is proper time defined?";
const DEEP_QUESTION = process.env.DEEP_QUESTION
  ?? "What is the current Particle Data Group world average for the W boson mass, with its uncertainty?";
const ANSWER_MS = 300_000;

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

const lastAnswer = (selector) => page.evaluateHandle((inside) => {
  const answers = document.querySelectorAll('[data-testid="answer"]');
  return answers[answers.length - 1].querySelector(inside);
}, selector);

async function ask(question) {
  await page.type("textarea[aria-label='Your question']", question);
  await page.keyboard.press("Enter");
  await page.waitForFunction(
    () => [...document.querySelectorAll("footer button")].some((button) => button.textContent === "Send"),
    { timeout: ANSWER_MS },
  );
  return page.evaluate(() => {
    const answers = document.querySelectorAll('[data-testid="answer"]');
    const answer = answers[answers.length - 1];
    return {
      error: answer.querySelector('[role="alert"]')?.textContent ?? null,
      sources: answer.querySelectorAll('[data-testid="source"]').length,
      webSources: answer.querySelectorAll('[data-testid="web-sources"] li').length,
      bar: Boolean(answer.querySelector('[data-testid="feedback-bar"]')),
    };
  });
}

async function rate(rating, note) {
  await (await lastAnswer(`[data-testid="rate-${rating}"]`)).click();
  await page.waitForFunction(() => {
    const answers = document.querySelectorAll('[data-testid="answer"]');
    return answers[answers.length - 1].querySelector('[data-testid="feedback-saved"]');
  }, { timeout: 10_000 });
  await (await lastAnswer("textarea[aria-label='Feedback note']")).type(note);
  await (await lastAnswer('[data-testid="save-note"]')).click();
  await page.waitForFunction(() => {
    const answers = document.querySelectorAll('[data-testid="answer"]');
    return answers[answers.length - 1].querySelector('[data-testid="feedback-saved"]');
  }, { timeout: 10_000 });
  const pressed = await page.evaluate((chosen) => {
    const answers = document.querySelectorAll('[data-testid="answer"]');
    return answers[answers.length - 1].querySelector(`[data-testid="rate-${chosen}"]`).getAttribute("aria-pressed");
  }, rating);
  check(`thumbs ${rating} saved with its note`, pressed === "true");
}

async function openList(filterIndex = 3) {
  await page.click('[data-testid="open-feedback"]');
  await page.waitForSelector('[data-testid="feedback-pane"]');
  await page.click(`[data-testid="filter-${filterIndex}"]`);
  await page.waitForFunction(
    (index) => document.querySelector(`[data-testid="filter-${index}"]`)?.getAttribute("aria-pressed") === "true",
    {}, filterIndex,
  );
  await new Promise((resolve) => setTimeout(resolve, 500));
  return page.evaluate(() => [...document.querySelectorAll('[data-testid="feedback-entry"]')].map((entry) => entry.innerText));
}

if (PHASE === "rate") {
  const fast = await ask(FAST_QUESTION);
  check("Fast answered", fast.error === null, fast.error ?? `${fast.sources} sources`);
  check("Fast answer has the feedback bar", fast.bar);
  await rate("up", "Live check: clear definition.");

  await page.click('[data-testid="mode-deep"]');
  await page.click('[data-testid="web-toggle"]');
  const deep = await ask(DEEP_QUESTION);
  check("Deep answered with web pages", deep.error === null && deep.webSources > 0,
    deep.error ?? `${deep.sources} sources, ${deep.webSources} web pages`);
  await rate("down", "Live check: a test thumbs-down, not a real complaint.");

  const entries = await openList();
  check("both in the review list", entries.length >= 2, `${entries.length} entries`);
  check("newest first, with its note", entries[0]?.includes("Live check: a test thumbs-down"));
  await page.click('[data-testid="feedback-entry"] button[aria-expanded]');
  await page.waitForSelector('[data-testid="feedback-detail"]');
  const detail = await page.$eval('[data-testid="feedback-detail"]', (element) => ({
    text: element.innerText,
    webLinks: element.querySelectorAll('[data-testid="feedback-sources"] a').length,
  }));
  check("the entry shows its answer and web pages", detail.text.length > 200 && detail.webLinks > 0,
    `${detail.text.length} chars, ${detail.webLinks} web links`);
  await page.click('[data-testid="toggle-candidate"]');
  await page.waitForSelector('[data-testid="candidate-badge"]', { timeout: 10_000 });
  check("made a gold candidate", true);
  await page.screenshot({ path: process.env.SHOT ?? "feedback.png" });
} else {
  const entries = await openList();
  check("after a restart both are still listed", entries.length >= 2, `${entries.length} entries`);
  check("the candidate is kept", entries[0]?.includes("gold candidate"));
  check("the up rating is kept", entries.some((text) => text.includes("Live check: clear definition.")));
}

check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | "));
await page.close();
await browser.disconnect();
if (problems.length) {
  console.log(`\n${problems.length} problem(s): ${problems.join("; ")}`);
  process.exit(1);
}
console.log("\nall checks passed");
