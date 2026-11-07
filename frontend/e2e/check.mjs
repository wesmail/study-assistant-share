// Drives the running app in a real browser: `npm run e2e -- <out dir>`.
// Start the server first, and a browser with --remote-debugging-port=9333.
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
  if (question) {
    await page.type("textarea", question);
    await page.keyboard.press("Enter");
  }
  await page.waitForFunction(
    () => [...document.querySelectorAll("footer button")].some((button) => button.textContent === "Send"),
    { timeout: ANSWER_MS },
  );
  // The crops are requested when the answer finishes, so give them time to arrive.
  await page.waitForFunction(
    () => [...document.querySelectorAll('[data-testid="unverified"] img')].every((img) => img.complete),
    { timeout: 15_000 },
  );
  return page.evaluate(() => {
    const answers = document.querySelectorAll('[data-testid="answer"]');
    const answer = answers[answers.length - 1];
    return {
      text: answer.innerText.length,
      katex: answer.querySelectorAll(".katex").length,
      katexErrors: answer.querySelectorAll(".katex-error").length,
      citations: answer.querySelectorAll('[data-testid="citation"]').length,
      sources: answer.querySelectorAll('[data-testid="source"]').length,
      crops: answer.querySelectorAll('[data-testid="unverified"] img').length,
      cropsLoaded: [...answer.querySelectorAll('[data-testid="unverified"] img')].every((img) => img.naturalWidth > 0),
      shiki: answer.querySelectorAll('[data-testid="code-block"] pre.shiki span').length,
      error: answer.querySelector('[role="alert"]')?.textContent ?? null,
      badge: answer.querySelector('[data-testid="model-badge"]')?.textContent ?? null,
      fallback: answer.querySelector('[data-testid="fallback-note"]')?.textContent ?? null,
      searched: answer.querySelector('[data-testid="search-query"]')?.textContent ?? null,
      sourceTitles: [...answer.querySelectorAll('[data-testid="source"]')].map((node) => node.textContent),
      body: answer.innerText,
    };
  });
}

// gpt-oss writes no citation in about 1 answer in 10, and the app says so. That isn't
// what these checks are about, so such an answer gets one more try.
async function askCited(question) {
  const first = await ask(question);
  if (first.sources > 0) return first;
  console.log("no citation in this answer, asking once more");
  return ask(question);
}

const painted = () =>
  page.evaluate(() => {
    const canvas = document.querySelector('[data-testid="pdf-canvas"]');
    const { width, height } = canvas;
    if (!width) return 0;
    const pixels = canvas.getContext("2d").getImageData(0, 0, width, height).data;
    let dark = 0;
    for (let i = 0; i < pixels.length; i += 4) if (pixels[i] < 128) dark++;
    return dark;
  });

await page.goto(APP, { waitUntil: "networkidle2" });
const stripWidth = await page.$eval('[data-testid="viewer-strip"]', (node) => node.getBoundingClientRect().width)
  .catch(() => null);
check("with nothing open the viewer is a narrow strip", stripWidth !== null && stripWidth < 60, `${stripWidth} px`);
const titles = await page.$$eval('[data-testid="library"] li', (rows) => rows.map((row) => row.innerText));
check("library lists the documents", titles.length >= 5, `${titles.length} documents`);

const rubbia = await page.$('input[aria-label*="Phenomenology of Particle Physics"]');
await rubbia.click();

const physics = await askCited(
  "What are the two equivalent forms of the amplitude factor W_2 for Z exchange in W+W- production at e+e- colliders?",
);
console.log("physics answer", JSON.stringify(physics));
check("the answer has text", physics.text > 100 && !physics.error, physics.error ?? "");
check("math renders with KaTeX", physics.katex > 0, `${physics.katex} formulas, ${physics.katexErrors} failed`);
check("citations are links", physics.citations > 0, `${physics.citations}`);
check("cited sources are listed", physics.sources > 0, `${physics.sources}`);
check("the answer's badge says it ran locally", physics.badge?.startsWith("local · "), physics.badge ?? "none");
check("unverified formulas show their crop", physics.crops > 0 && physics.cropsLoaded, `${physics.crops} crops`);
await page.screenshot({ path: `${out}/answer.png` });
const chat = await page.$('[data-testid="answer"]');
await chat.screenshot({ path: `${out}/answer-full.png` });

const citation = await page.$('[data-testid="citation"]');
// A live answer can have no readable citation. That fails above, and the later checks still run.
if (citation) {
  await citation.click();
  await page.waitForSelector('[data-testid="highlight"]', { timeout: 30_000 });
  await page.waitForFunction(
    () => document.querySelector('[data-testid="pdf-canvas"]').width > 0, { timeout: 30_000 });
  await new Promise((resolve) => setTimeout(resolve, 1500));
  check("the PDF page is drawn", (await painted()) > 1000, `${await painted()} dark pixels`);
  const highlights = (await page.$$('[data-testid="highlight"]')).length;
  check("the cited region is highlighted", highlights > 0, `${highlights} boxes`);
  console.log("page", await page.$eval('[data-testid="page-input"]', (node) => node.value));
  await page.screenshot({ path: `${out}/viewer.png` });
  const viewerWidth = await page.$eval('[data-testid="viewer"]', (node) => node.getBoundingClientRect().width);
  check("a citation opens the viewer full width", viewerWidth > 300, `${viewerWidth} px`);
  await page.click('[data-testid="close-viewer"]');
  check("the close button folds it back to the strip", Boolean(await page.$('[data-testid="viewer-strip"]')));
}

await page.click('input[aria-label*="Phenomenology of Particle Physics"]');
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((button) => button.textContent === "New chat").click();
});
const code = await ask("Write a short Python function that computes softmax with numpy.");
console.log("code answer", JSON.stringify(code));
// Shiki loads on first use, so it arrives a moment after the answer ends.
const highlighted = await page
  .waitForSelector('[data-testid="code-block"] pre.shiki span', { timeout: 20_000 })
  .then(() => true, () => false);
check("code is highlighted by Shiki", highlighted);
await page.screenshot({ path: `${out}/code.png` });

await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((button) => button.textContent === "New chat").click();
});
await page.click('button[title*="Phenomenology of Particle Physics"]');
check("opening a document opens the viewer", Boolean(await page.waitForSelector('[data-testid="viewer"]', { timeout: 5000 })
  .catch(() => null)));
await page.click('[data-testid="scope-document"]');
const scopeTitle = await page.$eval('[data-testid="scope-note"]', (node) => node.textContent);
check("this document names the open document", scopeTitle.includes("Phenomenology"), scopeTitle);
const scoped = await askCited("What is the mass of the W boson?");
console.log("scoped answer", JSON.stringify({ sources: scoped.sourceTitles }));
check(
  "this document searches only the open document",
  scoped.sourceTitles.length > 0 && scoped.sourceTitles.every((title) => title.includes("Phenomenology")),
  `${scoped.sourceTitles.length} sources`,
);
await page.click('[data-testid="scope-library"]');

// The answer selector: four options in config order, and what each says about the machine.
const labels = await page.$$eval('[aria-label="Answer model"] button', (nodes) => nodes.map((node) => node.firstChild.textContent));
const costs = await page.$$eval('[data-testid^="cost-"]', (nodes) => nodes.map((node) => node.dataset.testid));
check("only the hosted options show an expected cost", costs.join(",") === "cost-qwen,cost-deep", costs.join(","));
const spendText = await page.$eval('[data-testid="month-spend"]', (node) => node.textContent);
check("the selector shows this month's hosted spend", /^Hosted this month: Anthropic \$\d+\.\d\d of \$\d+ · OpenRouter \$\d+\.\d\d$/.test(spendText), spendText);
check("the selector lists the four options in order", labels.join(",") === "Fast,Balanced,Qwen,Deep", labels.join(","));
const fastNote = await page.$eval('[data-testid="mode-line"]', (node) => node.textContent);
check("a local option says it runs on this machine", fastNote.startsWith("Runs on this machine"), fastNote);
const noteLines = await page.$eval('[data-testid="mode-note"]', (node) => node.getBoundingClientRect().height);
check("the mode note is one line", noteLines < 24, `${noteLines} px high`);
const shown = () => page.$eval('[data-testid="note-details"]', (node) => getComputedStyle(node).visibility);
const hiddenFirst = await shown();
await page.focus('[data-testid="note-details-button"]');           // as Tab would
const onFocus = await shown();
await page.keyboard.press("Escape");
check("its details show on keyboard focus and hide on Escape", hiddenFirst === "hidden" && onFocus === "visible"
  && (await shown()) === "hidden", `${hiddenFirst} → ${onFocus}`);
await page.click('[data-testid="mode-qwen"]');
const qwenNote = await page.$eval('[data-testid="mode-note"]', (node) => node.textContent);
check("Qwen names OpenRouter and its one provider", qwenNote.includes("OpenRouter") && qwenNote.includes("DeepInfra"), qwenNote);
await page.click('[data-testid="mode-fast"]');

// A real OpenRouter call, so it costs money and runs only when asked for: E2E_QWEN=1.
if (process.env.E2E_QWEN) {
  await page.click('[data-testid="mode-qwen"]');
  await page.screenshot({ path: `${out}/qwen-selector.png` });
  await page.type("textarea", "In two sentences, why is the log-cosh loss quadratic near zero?");
  await page.keyboard.press("Enter");
  await page.waitForFunction(
    () => document.querySelector('[data-testid="waiting"]')?.textContent?.startsWith("Thinking"),
    { timeout: ANSWER_MS },
  );
  await new Promise((resolve) => setTimeout(resolve, 2500));
  const waiting = await page.$eval('[data-testid="waiting"]', (node) => node.textContent);
  check("Qwen shows a live thinking clock", /^Thinking… \d+ s|^Thinking… \d+:\d\d/.test(waiting), waiting);
  await page.screenshot({ path: `${out}/qwen-thinking.png` });
  const qwen = await ask("");
  console.log("qwen answer", JSON.stringify({ badge: qwen.badge, error: qwen.error, text: qwen.text }));
  check("the Qwen answer has text or offers Deep",
    (qwen.text > 20 && !qwen.error) || Boolean(await page.$('[data-testid="resend-deep"]')), qwen.error ?? "");
  check("the badge names Qwen and DeepInfra", qwen.badge === "hosted · qwen/qwen3.8-27b · DeepInfra", qwen.badge ?? "none");
  const cost = await page.$$eval('[data-testid="answer-cost"]', (nodes) => nodes.at(-1)?.textContent ?? "");
  check("the Qwen answer shows its cost", /^\$0\.\d+ for this answer · [\d,]+ in, [\d,]+ out tokens/.test(cost), cost);
  const back = await page.$eval('[data-testid="mode-fast"]', (node) => node.getAttribute("aria-pressed"));
  check("the selector goes back to the last local choice", back === "true");
  await page.screenshot({ path: `${out}/qwen-answer.png` });
}

// Phase 5.1: a cited figure next to a Deep answer, and a screenshot with a question. One Deep
// call and one local Balanced answer, so only when asked for: E2E_FIGURES=1.
if (process.env.E2E_FIGURES) {
  await page.evaluate(() => {
    [...document.querySelectorAll("button")].find((button) => button.textContent === "New chat").click();
  });
  await page.click('[data-testid="mode-deep"]');
  const deepNote = await page.$eval('[data-testid="mode-note"]', (node) => node.textContent);
  check("Deep's note says figure images go too", deepNote.includes("the images of any figures among them"), deepNote);
  const figureAnswer = await ask("In Understanding Deep Learning, what does the double descent figure show about the test error with label noise?");
  console.log("figure answer", JSON.stringify({ badge: figureAnswer.badge, error: figureAnswer.error, text: figureAnswer.text }));
  await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="cited-figures"] img')]
    .every((image) => image.complete), { timeout: 15_000 });
  const figures = await page.$$eval('[data-testid="cited-figures"] img', (images) =>
    images.map((image) => ({ src: image.getAttribute("src"), alt: image.getAttribute("alt"),
      loaded: image.complete && image.naturalWidth > 0 })));
  check("a cited figure is shown next to the answer", figures.length > 0 && figures.every((f) => f.loaded), JSON.stringify(figures));
  check("the figure asked about is the one shown first", /Figure 8\.10/.test(figures[0]?.alt ?? ""), figures[0]?.alt ?? "none");
  await page.screenshot({ path: `${out}/figure-answer.png` });

  await page.click('[data-testid="mode-fast"]');
  const url = figures[0]?.src;
  await page.evaluate(async (src) => {
    const blob = await (await fetch(src)).blob();
    const input = document.querySelector('[data-testid="image-picker"]');
    const transfer = new DataTransfer();
    transfer.items.add(new File([blob], "screenshot.png", { type: "image/png" }));
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, url);
  await page.waitForSelector('[data-testid="attached-images"] img');
  const notice = await page.$eval('[data-testid="image-notice"]', (node) => node.textContent);
  check("Fast says it can't read images and offers the ones that can", notice.includes("Fast can't read images")
    && notice.includes("Balanced") && notice.includes("Deep"), notice);
  await page.type("textarea", "What does this plot show? Name the axes.");
  const blocked = await page.evaluate(() =>
    [...document.querySelectorAll("footer button")].find((button) => button.textContent === "Send").disabled);
  check("Fast won't send an image", blocked);
  await page.screenshot({ path: `${out}/image-notice.png` });
  await page.click('[data-testid="image-use-balanced"]');
  await page.focus("textarea");
  await page.keyboard.press("Enter");
  const shot = await ask("");
  console.log("screenshot answer", JSON.stringify({ badge: shot.badge, error: shot.error, body: shot.body.slice(0, 400) }));
  const bubble = await page.$$eval('[data-testid="question-images"] img', (images) => images.length);
  check("the image shows in the question's bubble", bubble >= 1, `${bubble}`);
  check("Balanced answered about the image", shot.text > 40 && !shot.error && /local · gemma/.test(shot.badge ?? ""), shot.badge ?? "");
  check("the answer names the axes", /hidden|error|train|test/i.test(shot.body), shot.body.slice(0, 200));
  check("a question with an image is searched as written", !/\(about:/.test(shot.searched ?? ""), shot.searched ?? "not rewritten");
  await page.screenshot({ path: `${out}/screenshot-answer.png` });
}

// A real hosted call, so it costs money and runs only when asked for:
//   E2E_DEEP="hosted · claude-opus-5-5"   the badge the answer should have
//   E2E_FALLBACK=1                        the main model is expected to fail
if (process.env.E2E_DEEP) {
  await page.click('[data-testid="mode-deep"]');
  const note = await page.$eval('[data-testid="mode-note"]', (node) => node.textContent);
  check("deep mode says what leaves the machine", note.includes("Anthropic"), note);
  const deep = await ask("In one sentence, what does the Cholesky decomposition factor a matrix into?");
  console.log("deep answer", JSON.stringify(deep));
  check("the deep answer has text", deep.text > 20 && !deep.error, deep.error ?? "");
  check("the badge names the model that answered", deep.badge === process.env.E2E_DEEP, deep.badge ?? "none");
  check(
    process.env.E2E_FALLBACK ? "the fallback is shown" : "there was no fallback",
    Boolean(deep.fallback) === Boolean(process.env.E2E_FALLBACK),
    deep.fallback ?? "",
  );
  const fast = await page.$eval('[data-testid="mode-fast"]', (node) => node.getAttribute("aria-pressed"));
  check("the toggle goes back to fast after a question", fast === "true");
  await page.click('[data-testid="mode-deep"]');
  await page.screenshot({ path: `${out}/deep.png` });
}

// The follow-up that went wrong in real use. Two hosted calls, so only when asked for.
if (process.env.E2E_FOLLOWUP) {
  await page.evaluate(() => {
    [...document.querySelectorAll("button")].find((button) => button.textContent === "New chat").click();
  });
  await page.click('[data-testid="mode-deep"]');
  const first = await ask("What is SeismoGPT?");
  await page.click('[data-testid="mode-deep"]');
  const followUp = await ask("I don't understand!");
  await page.screenshot({ path: `${out}/followup.png` });
  console.log("follow-up", JSON.stringify({ searched: followUp.searched, sources: followUp.sources }));
  console.log("follow-up text", followUp.body.slice(0, 900).replaceAll("\n", " "));
  check("the first answer is about SeismoGPT", /SeismoGPT/.test(first.body) && first.sources > 0);
  check("the follow-up searched for the topic", /SeismoGPT/.test(followUp.searched ?? ""), followUp.searched ?? "none");
  check("the follow-up has sources", followUp.sources > 0, `${followUp.sources}`);
  check("the follow-up explains SeismoGPT", /SeismoGPT/.test(followUp.body));
  check(
    "it doesn't claim the library lacks SeismoGPT",
    !/(library|your (?:books?|documents?|files?|collection))[^.]{0,60}(doesn['’]t|does not|don['’]t|do not|lacks?|no mention|contains? no)/i.test(followUp.body),
  );
}

check("no errors in the browser console", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await page.close();
await browser.disconnect();
console.log(problems.length ? `FAILED: ${problems.join(", ")}` : "All checks passed");
process.exit(problems.length ? 1 : 0);
