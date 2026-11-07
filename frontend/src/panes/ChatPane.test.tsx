import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStore } from "../store/store";
import type { Message } from "../store/store";
import type { AnswerOption } from "../api";
import { ChatPane, asksForCode, asksForFile, clock, costLine, dollars, duration, readImage, toolHint } from "./ChatPane";

const OPTIONS: AnswerOption[] = [
  { id: "fast", label: "Fast", provider: "ollama", model: "gpt-oss:20b", fallback: null, route: null, hosted: false,
    expected_seconds: 17, expected_note: "35 gold questions, round 2", loaded: true, expected_usd: null, price: null, vision: false },
  { id: "balanced", label: "Balanced", provider: "ollama", model: "gemma4:26b-a4b-it-qat", fallback: null, route: null,
    hosted: false, expected_seconds: 103, expected_note: "10 derivation questions, round 3", loaded: false, expected_usd: null, price: null, vision: true },
  { id: "qwen", label: "Qwen", provider: "openrouter", model: "qwen/qwen3.8-27b", fallback: null, route: "DeepInfra",
    hosted: true, expected_seconds: 188, expected_note: "35 gold questions, model bench", loaded: null, expected_usd: 0.0154,
    price: { input: 0.15, output: 1.875, checked: "2026-11-03" }, vision: false },
  { id: "deep", label: "Deep", provider: "anthropic", model: "claude-opus-5-5", fallback: "claude-sonnet-5-5",
    route: null, hosted: true, expected_seconds: 28, expected_note: "35 gold questions, round 2", loaded: null,
    expected_usd: 0.0903, price: { input: 4, output: 20, checked: "2026-11-03" }, vision: true },
];

const reply = (change: Partial<Message>): Message => ({
  id: 1, role: "assistant", text: "", thinking: "", sources: [], citations: [], unverified: [],
  error: null, truncated: false, streaming: false, finished: true,
  model: "gpt-oss:20b", hosted: false, fallbackFrom: null, fallbackReason: null, searchQuery: null,
  tools: [], files: [], option: "fast", servedBy: null, question: "Q?", startedAt: Date.now(), searched: true,
  cutOff: false, noAnswer: false, spend: null, images: [], web: false, webSources: [], webCitations: [], webProposal: null,
  questionId: null, feedback: null, check: false, checkStatus: null, checks: null, replaced: [], ...change,
});

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  useStore.setState({
    messages: [], mode: "fast", localMode: "fast", sending: false, scope: [], scopeMode: "library", viewer: null,
    options: OPTIONS, switchSeconds: [20, 55],
    monthSpend: [{ provider: "anthropic", usd: 15.84, limit_usd: 30 }, { provider: "openrouter", usd: 0.6812, limit_usd: null }],
    documents: [{ id: 7, path: "a.pdf", title: "Book A", author: null, year: null, kind: "pdf", pages: 1, status: "indexed",
      chunks: 1 }],
  });
});
afterEach(cleanup);

describe("the answer selector", () => {
  const note = () => screen.getByTestId("mode-note").textContent ?? "";

  it("lists the four options in order and starts on fast, which stays on this machine", () => {
    render(<ChatPane />);
    expect(screen.getAllByRole("button").filter((b) => b.dataset.testid?.startsWith("mode-")).map((b) => b.textContent))
      .toEqual(["Fast", "Balanced", "Qwen~$0.015", "Deep~$0.090"]);
    expect(screen.getByTestId("mode-fast").getAttribute("aria-pressed")).toBe("true");
    expect(note()).toContain("Runs on this machine.");
    expect(note()).toContain("Usually 17 s an answer");
  });

  it("says it in one short line, with the detail in a tooltip a keyboard can reach (item 9)", () => {
    render(<ChatPane />);
    const line = () => screen.getByTestId("mode-line").textContent;
    expect(line()).toBe("Runs on this machine · usually 17 s");
    fireEvent.click(screen.getByTestId("mode-balanced"));
    expect(line()).toBe("Runs on this machine · usually about 2 min · first answer +20–55 s to load");
    fireEvent.click(screen.getByTestId("mode-deep"));
    expect(line()).toBe("Sent to Anthropic · usually 28 s, about $0.090");
    fireEvent.click(screen.getByTestId("mode-qwen"));
    expect(line()).toBe("Sent to OpenRouter, DeepInfra only · usually about 3 min, about $0.015");
    const button = screen.getByTestId("note-details-button");
    const tooltip = screen.getByRole("tooltip", { hidden: true });
    expect(button.tagName).toBe("BUTTON");                  // reachable with Tab
    expect(button.getAttribute("aria-describedby")).toBe(tooltip.id);
    expect(tooltip.textContent).toContain("never switches by itself");
    expect(tooltip.className).toContain("group-focus-within:visible");
  });

  it("warns that a local model that isn't loaded makes the first answer wait", () => {
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("mode-balanced"));
    expect(note()).toContain("Runs on this machine.");
    expect(note()).toContain("about 2 min an answer");
    expect(note()).toContain("isn't loaded, so the first answer waits 20–55 s longer");
  });

  it("names OpenRouter and the one provider it may use for Qwen", () => {
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("mode-qwen"));
    expect(note()).toContain("Goes to OpenRouter, which may pass it only to DeepInfra");
    expect(note()).toContain("keeps or trains on prompts");
    expect(note()).toContain("previous question and answer");
    expect(note()).toContain("never switches by itself");
    expect(note()).not.toContain("Runs on this machine");
  });

  it("says what leaves the machine when deep is chosen", () => {
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("mode-deep"));
    expect(screen.getByTestId("mode-deep").getAttribute("aria-pressed")).toBe("true");
    const note = screen.getByTestId("mode-note").textContent;
    expect(note).toContain("Anthropic");
    expect(note).toContain("previous question and answer");
    expect(note).toContain("passages it cited");
  });
});

describe("the scope selector", () => {
  const open = { docId: 7, page: 1, regions: [], opened: 1 };
  const pressed = (name: string) => screen.getByTestId(`scope-${name}`).getAttribute("aria-pressed");
  const disabled = (name: string) => (screen.getByTestId(`scope-${name}`) as HTMLButtonElement).disabled;

  it("starts on the library, with the other two unavailable", () => {
    render(<ChatPane />);
    expect(pressed("library")).toBe("true");
    expect(disabled("document")).toBe(true);
    expect(disabled("ticked")).toBe(true);
  });

  it("offers this document once one is open, and names it", () => {
    useStore.setState({ viewer: open });
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("scope-document"));
    expect(pressed("document")).toBe("true");
    expect(screen.getByTestId("scope-note").textContent).toBe("Book A");
  });

  it("counts the ticked documents", () => {
    useStore.setState({ scope: [7], scopeMode: "ticked" });
    render(<ChatPane />);
    expect(screen.getByTestId("scope-ticked").textContent).toBe("Ticked (1)");
    expect(pressed("ticked")).toBe("true");
  });

  it("shows the library as active when the chosen scope can't apply", () => {
    useStore.setState({ scopeMode: "document", viewer: null });
    render(<ChatPane />);
    expect(pressed("library")).toBe("true");
  });
});

describe("the search line", () => {
  it("shows what a follow-up was rewritten to", () => {
    useStore.setState({ messages: [reply({ searchQuery: "What does SeismoGPT do?" })] });
    render(<ChatPane />);
    expect(screen.getByTestId("search-query").textContent).toBe("Searched for: What does SeismoGPT do?");
  });

  it("stays away when nothing was rewritten", () => {
    useStore.setState({ messages: [reply({})] });
    render(<ChatPane />);
    expect(screen.queryByTestId("search-query")).toBeNull();
  });
});

describe("figures in an answer", () => {
  const figure = { url: "/api/documents/7/figures/figure_792_page90.png", page: 90, caption: "Figure 8.10 Double descent." };
  const source = (id: number, figures?: typeof figure[]) => ({ id, chunk_id: id, doc_id: 7, doc_title: "Book A",
    page_start: 90, page_end: 90, focus_page: 90, heading_path: [], regions: [], figures });

  it("shows the image of a figure in a cited source, and opens its page on a click", () => {
    useStore.setState({ messages: [reply({ sources: [source(1, [figure]), source(2)], citations: [1] })] });
    render(<ChatPane />);
    const image = screen.getByAltText("Figure 8.10 Double descent.");
    expect(image.getAttribute("src")).toBe(figure.url);
    expect(screen.getByTestId("cited-figures").textContent).toContain("S1 · page 90 · Figure 8.10 Double descent.");
    fireEvent.click(image);
    expect(useStore.getState().viewer?.page).toBe(90);
    expect(useStore.getState().viewer?.docId).toBe(7);
  });

  it("shows each cited source's nearest figure, and the rest on request", () => {
    const other = { url: "/api/documents/7/figures/figure_780_page88.png", page: 88, caption: "Figure 8.8 Overfitting." };
    useStore.setState({ messages: [reply({ sources: [source(1, [figure, other])], citations: [1] })] });
    render(<ChatPane />);
    expect(screen.getByAltText("Figure 8.10 Double descent.")).toBeTruthy();
    expect(screen.queryByAltText("Figure 8.8 Overfitting.")).toBeNull();
    fireEvent.click(screen.getByTestId("all-figures"));
    expect(screen.getByAltText("Figure 8.8 Overfitting.")).toBeTruthy();
    expect(screen.queryByTestId("all-figures")).toBeNull();
  });

  it("shows nothing for a figure in a source the answer didn't cite", () => {
    useStore.setState({ messages: [reply({ sources: [source(1, [figure]), source(2)], citations: [2] })] });
    render(<ChatPane />);
    expect(screen.queryByTestId("cited-figures")).toBeNull();
  });

  it("tells the reader that Deep sends figure images too", () => {
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("mode-deep"));
    expect(screen.getByTestId("mode-note").textContent).toContain("the images of any figures among them");
    fireEvent.click(screen.getByTestId("mode-qwen"));
    expect(screen.getByTestId("mode-note").textContent).not.toContain("images");
  });
});

describe("an image with the question", () => {
  const png = () => new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], "shot.png", { type: "image/png" });
  const attach = async (file: File) => {
    fireEvent.change(screen.getByTestId("image-picker"), { target: { files: [file] } });
    await screen.findByAltText("Attached image 1");
  };

  it("with Fast, says Fast can't read it, blocks sending, and switches on a click", async () => {
    render(<ChatPane />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "What is this plot?" } });
    await attach(png());
    expect(screen.getByTestId("image-notice").textContent).toContain("Fast can't read images. Ask with");
    expect(screen.getByTestId("image-notice").textContent).toContain("Balanced (runs here)");
    expect(screen.getByTestId("image-notice").textContent).toContain("Deep (goes to Anthropic)");
    expect((screen.getByText("Send") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTestId("image-use-balanced"));
    expect(useStore.getState().mode).toBe("balanced");
    expect(screen.queryByTestId("image-notice")).toBeNull();
    expect((screen.getByText("Send") as HTMLButtonElement).disabled).toBe(false);
  });

  it("sends it with the question to a vision option, then clears it", async () => {
    const send = vi.fn(async () => undefined);
    useStore.setState({ send, mode: "balanced" });
    render(<ChatPane />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "What is this plot?" } });
    await attach(png());
    fireEvent.click(screen.getByText("Send"));
    expect(send).toHaveBeenCalledWith("What is this plot?", "balanced", [{ media_type: "image/png", data: "iVBORw0KGgo=" }]);
    expect(screen.queryByTestId("attached-images")).toBeNull();
  });

  it("can be removed before sending", async () => {
    render(<ChatPane />);
    await attach(png());
    fireEvent.click(screen.getByLabelText("Remove image 1"));
    expect(screen.queryByTestId("attached-images")).toBeNull();
  });

  it("refuses what isn't a small enough image", async () => {
    await expect(readImage(new File(["<svg/>"], "a.svg", { type: "image/svg+xml" }))).rejects.toThrow("PNG, JPEG, WebP or GIF");
    const big = new File([new Uint8Array(5 * 1024 * 1024 + 1)], "big.png", { type: "image/png" });
    await expect(readImage(big)).rejects.toThrow("the limit is 5 MB");
  });

  it("shows in the question's bubble", () => {
    useStore.setState({ messages: [{ ...reply({}), role: "user", text: "What is this?",
      images: [{ media_type: "image/png", data: "AAAA" }] }] });
    render(<ChatPane />);
    expect(screen.getByAltText("Your image").getAttribute("src")).toBe("data:image/png;base64,AAAA");
  });

  it("is named in Deep's note", () => {
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("mode-deep"));
    expect(screen.getByTestId("mode-note").textContent).toContain("this question and any image you attach");
  });
});

describe("the code and file hint", () => {
  const type = (text: string) => fireEvent.change(screen.getByLabelText("Your question"), { target: { value: text } });

  it("suggests Qwen or Deep when a local option is chosen and the question asks for code", () => {
    render(<ChatPane />);
    expect(screen.queryByTestId("tool-hint")).toBeNull();
    type("So, how can I translate this to code?");
    expect(screen.getByTestId("tool-hint").textContent).toBe(
      "For code, Qwen and Deep are more reliable: they passed all 22 tool tasks, against 16 for Fast and 17 for Balanced.");
    fireEvent.click(screen.getByTestId("mode-balanced"));
    expect(screen.getByTestId("tool-hint")).toBeTruthy();
  });

  it("goes away when Qwen or Deep is chosen, or the question isn't about code", () => {
    render(<ChatPane />);
    type("Write a Python function for Gram-Schmidt.");
    fireEvent.click(screen.getByTestId("mode-qwen"));
    expect(screen.queryByTestId("tool-hint")).toBeNull();
    fireEvent.click(screen.getByTestId("mode-deep"));
    expect(screen.queryByTestId("tool-hint")).toBeNull();
    fireEvent.click(screen.getByTestId("mode-fast"));
    type("What does the likelihood function look like?");
    expect(screen.queryByTestId("tool-hint")).toBeNull();
  });

  it("knows a request for code from a maths question", () => {
    expect(["Implement softmax in NumPy", "Run this:\n```python\nprint(1)\n```", "a script for the fit",
      "how can I translate this to code?"].map(asksForCode)).toEqual([true, true, true, true]);
    expect(["What is the Breit-Wigner function?", "Derive the gradient of log-cosh", "Explain the loss function"]
      .map(asksForCode)).toEqual([false, false, false]);
  });

  it("shows for a file asked of Fast or Balanced, and not of Qwen or Deep", () => {
    render(<ChatPane />);
    type("Can you summarize the paper in 1 pdf page with 1 figure from the paper");
    expect(screen.getByTestId("tool-hint").textContent).toBe(
      "For files, Qwen and Deep are more reliable: they passed all 22 tool tasks, against 16 for Fast and 17 for Balanced.");
    fireEvent.click(screen.getByTestId("mode-balanced"));
    expect(screen.getByTestId("tool-hint")).toBeTruthy();
    fireEvent.click(screen.getByTestId("mode-deep"));
    expect(screen.queryByTestId("tool-hint")).toBeNull();
  });

  it("knows a request for a file, as the server does, and that pdf alone is a density", () => {
    expect(["Make a 5-slide deck on SVD", "Export this answer as a PDF", "write it into a Word document",
      "save the notes as markdown", "summarize the paper in 1 pdf page"].map(asksForFile))
      .toEqual([true, true, true, true, true]);
    expect(["What is the pdf of a Gaussian?", "Derive the marginal pdf of X", "Explain this slide's equation"]
      .map((text) => [asksForFile(text), text])).toEqual([[false, "What is the pdf of a Gaussian?"],
      [false, "Derive the marginal pdf of X"], [true, "Explain this slide's equation"]]);
    expect(toolHint("Write Python code and save it as a PDF")).toMatch(/^For code and files, /);
    expect(toolHint("What is a Gaussian?")).toBeNull();
  });
});

describe("costs", () => {
  it("shows the expected cost next to each hosted option, and none for local ones", () => {
    render(<ChatPane />);
    expect(screen.getByTestId("cost-qwen").textContent).toBe("~$0.015");
    expect(screen.getByTestId("cost-deep").textContent).toBe("~$0.090");
    expect(screen.queryByTestId("cost-fast")).toBeNull();
    expect(screen.queryByTestId("cost-balanced")).toBeNull();
  });

  it("shows this month's hosted spend, with the Anthropic limit", () => {
    render(<ChatPane />);
    expect(screen.getByTestId("month-spend").textContent).toBe(
      "Hosted this month: Anthropic $15.84 of $30 · OpenRouter $0.68");
    fireEvent.click(screen.getByTestId("mode-deep"));
    expect(screen.getByTestId("month-spend").textContent).toContain("Deep prices checked 2026-11-03");
    expect(screen.getByTestId("mode-note").textContent).toContain("and about $0.090 a question.");
  });

  it("shows what a hosted answer cost, from the log", () => {
    useStore.setState({ messages: [reply({ model: "qwen/qwen3.8-27b", hosted: true, option: "qwen",
      spend: { calls: 1, failed_calls: 0, cost_usd: 0.001440675, input_tokens: 7142, output_tokens: 197 } })] });
    render(<ChatPane />);
    expect(screen.getByTestId("answer-cost").textContent).toBe("$0.0014 for this answer · 7,142 in, 197 out tokens");
  });

  it("counts a failed attempt in, and says so", () => {
    expect(costLine({ calls: 2, failed_calls: 1, cost_usd: 0.12, input_tokens: 100, output_tokens: 10 }))
      .toBe("$0.120 for this answer · 100 in, 10 out tokens · 2 calls, 1 of them failed, which is included");
  });

  it("shows no cost for a local answer", () => {
    useStore.setState({ messages: [reply({})] });
    render(<ChatPane />);
    expect(screen.queryByTestId("answer-cost")).toBeNull();
  });

  it("writes small amounts with enough places", () => {
    expect([dollars(0.0013), dollars(0.0903), dollars(15.8412)]).toEqual(["$0.0013", "$0.090", "$15.84"]);
  });
});

describe("while waiting", () => {
  it("says it's searching, then waiting for the model, then thinking, with a clock", () => {
    const started = Date.now() - 83_000;
    useStore.setState({ messages: [reply({ streaming: true, finished: false, searched: false, startedAt: started })] });
    render(<ChatPane />);
    expect(screen.getByTestId("waiting").textContent).toBe("Searching the library… 1:23");
    cleanup();
    useStore.setState({ messages: [reply({ streaming: true, finished: false, model: "gemma4:26b-a4b-it-qat",
      startedAt: started })] });
    render(<ChatPane />);
    expect(screen.getByTestId("waiting").textContent).toBe("Waiting for gemma4:26b-a4b-it-qat… 1:23");
    cleanup();
    useStore.setState({ messages: [reply({ streaming: true, finished: false, thinking: "Hmm", startedAt: started })] });
    render(<ChatPane />);
    expect(screen.getByTestId("waiting").textContent).toBe("Thinking… 1:23");
  });

  it("goes once the answer starts", () => {
    useStore.setState({ messages: [reply({ streaming: true, finished: false, thinking: "Hmm", text: "Proper" })] });
    render(<ChatPane />);
    expect(screen.queryByTestId("waiting")).toBeNull();
  });

  it("formats times", () => {
    expect([clock(5.9), clock(60), clock(188)]).toEqual(["5 s", "1:00", "3:08"]);
    expect([duration(17), duration(103), duration(188)]).toEqual(["17 s", "about 2 min", "about 3 min"]);
  });
});

describe("a Qwen answer that stopped", () => {
  it("says so and offers Deep, which is only sent on a click", () => {
    const resendDeep = vi.fn(async () => undefined);
    useStore.setState({ resendDeep, messages: [reply({
      option: "qwen", model: "qwen/qwen3.8-27b", hosted: true, finished: false, cutOff: true,
      error: "qwen/qwen3.8-27b stopped at its 16,000-token limit, thinking included, before it wrote an answer.",
    })] });
    render(<ChatPane />);
    expect(screen.getByRole("alert").textContent).toContain("16,000-token limit");
    expect(screen.getByText("Qwen ran out of room while thinking.")).toBeTruthy();
    expect(resendDeep).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("resend-deep"));
    expect(resendDeep).toHaveBeenCalledWith(1);
  });

  it("names Balanced when it ran out of room, and resends in Deep on a click", () => {
    const resendDeep = vi.fn(async () => undefined);
    useStore.setState({ resendDeep, messages: [reply({
      option: "balanced", model: "gemma4:26b-a4b-it-qat", finished: false, cutOff: true,
      error: "gemma4:26b-a4b-it-qat stopped at its 8,192-token limit, thinking included, before it wrote an answer.",
    })] });
    render(<ChatPane />);
    expect(screen.getByText("Balanced ran out of room while thinking.")).toBeTruthy();
    expect(screen.getByText("Sends it to Anthropic.")).toBeTruthy();
    expect(resendDeep).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("resend-deep"));
    expect(resendDeep).toHaveBeenCalledWith(1);
  });

  it("isn't offered when Balanced fails for another reason", () => {
    useStore.setState({ messages: [reply({ option: "balanced", error: "Ollama returned 500", cutOff: false })] });
    render(<ChatPane />);
    expect(screen.queryByTestId("resend-deep")).toBeNull();
  });

  it("isn't offered for a good Qwen answer, or for another option's failure", () => {
    useStore.setState({ messages: [reply({ option: "qwen", hosted: true, text: "Fine." }),
      reply({ id: 2, option: "fast", error: "Ollama returned 500" })] });
    render(<ChatPane />);
    expect(screen.queryByTestId("resend-deep")).toBeNull();
  });
});

describe("the answer badge", () => {
  it("names the provider that served a routed model", () => {
    useStore.setState({ messages: [reply({ model: "qwen/qwen3.8-27b", hosted: true, servedBy: "DeepInfra" })] });
    render(<ChatPane />);
    expect(screen.getByTestId("model-badge").textContent).toBe("hosted · qwen/qwen3.8-27b · DeepInfra");
  });

  it("shows a local answer", () => {
    useStore.setState({ messages: [reply({})] });
    render(<ChatPane />);
    expect(screen.getByTestId("model-badge").textContent).toBe("local · gpt-oss:20b");
    expect(screen.queryByTestId("fallback-note")).toBeNull();
  });

  it("shows a hosted answer", () => {
    useStore.setState({ messages: [reply({ model: "claude-opus-5-5", hosted: true })] });
    render(<ChatPane />);
    expect(screen.getByTestId("model-badge").textContent).toBe("hosted · claude-opus-5-5");
  });

  it("is never silent about a fallback", () => {
    useStore.setState({
      messages: [reply({
        model: "claude-sonnet-5-5", hosted: true,
        fallbackFrom: "claude-opus-5-5", fallbackReason: "HTTP 529, overloaded_error",
      })],
    });
    render(<ChatPane />);
    expect(screen.getByTestId("model-badge").textContent).toBe("hosted · claude-sonnet-5-5");
    expect(screen.getByTestId("fallback-note").textContent).toBe(
      "claude-opus-5-5 failed (HTTP 529, overloaded_error), so claude-sonnet-5-5 wrote this answer.",
    );
  });
});

describe("tools in the chat", () => {
  const pdf = { name: "notes.pdf", kind: "pdf", size: 2048, url: "/api/outputs/notes.pdf", preview: null, pages: 3 };
  const deck = { name: "deck.pptx", kind: "pptx", size: 10, url: "/api/outputs/deck.pptx", preview: "## A slide", slides: 5 };
  const plot = { name: "figure.png", kind: "png", size: 10, url: "/api/outputs/figure.png", preview: null };

  it("says what a running tool is doing, then how it went", () => {
    useStore.setState({ messages: [reply({
      streaming: true, finished: false,
      tools: [{ id: "1", name: "run_python", running: true, ok: true, summary: "" }],
    })] });
    render(<ChatPane />);
    expect(screen.getByTestId("tool-activity").textContent).toContain("Running code in the sandbox");
    cleanup();
    useStore.setState({ messages: [reply({
      tools: [{ id: "1", name: "run_python", running: false, ok: false, summary: "Exit code 1" }],
    })] });
    render(<ChatPane />);
    expect(screen.getByTestId("tool-activity").textContent).toBe("Exit code 1");
  });

  it("shows a card with facts and a download link", () => {
    useStore.setState({ messages: [reply({ files: [pdf] })] });
    render(<ChatPane />);
    const card = screen.getByTestId("file-card");
    expect(card.textContent).toContain("notes.pdf");
    expect(card.textContent).toContain("PDF · 2 KB · 3 pages");
    expect(screen.getByText("Download").getAttribute("href")).toBe("/api/outputs/notes.pdf?download=true");
  });

  it("opens a document's preview only when asked", () => {
    useStore.setState({ messages: [reply({ files: [pdf] })] });
    render(<ChatPane />);
    expect(screen.queryByTestId("file-preview")).toBeNull();
    fireEvent.click(screen.getByText("Preview"));
    expect(screen.getByTitle("notes.pdf").getAttribute("src")).toBe("/api/outputs/notes.pdf");
  });

  it("shows a plot straight away", () => {
    useStore.setState({ messages: [reply({ files: [plot] })] });
    render(<ChatPane />);
    expect(screen.getByAltText("figure.png").getAttribute("src")).toBe("/api/outputs/figure.png");
  });

  it("previews a deck as LibreOffice renders it", async () => {
    const fetched = vi.fn(async () => new Response(new Blob(["%PDF-1.7"]), { status: 200 }));
    vi.stubGlobal("fetch", fetched);
    URL.createObjectURL = vi.fn(() => "blob:rendered-deck");
    URL.revokeObjectURL = vi.fn();
    useStore.setState({ messages: [reply({ files: [deck] })] });
    render(<ChatPane />);
    fireEvent.click(screen.getByText("Preview"));
    expect(screen.getByRole("status").textContent).toContain("Rendering with LibreOffice");
    const frame = await screen.findByTestId("rendered-preview");
    expect(frame.getAttribute("src")).toBe("blob:rendered-deck");
    expect(fetched).toHaveBeenCalledWith("/api/outputs/deck.pptx/preview");
    vi.unstubAllGlobals();
  });

  it("makes no object URL for a render that arrives after the preview closed (review R2-C10)", async () => {
    let answer: (response: Response) => void = () => {};
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((resolve) => { answer = resolve; })));
    URL.createObjectURL = vi.fn(() => "blob:late");
    URL.revokeObjectURL = vi.fn();
    useStore.setState({ messages: [reply({ files: [deck] })] });
    const { unmount } = render(<ChatPane />);
    fireEvent.click(screen.getByText("Preview"));
    unmount();
    answer(new Response(new Blob(["%PDF"]), { status: 200 }));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("uses the card's render_url when there is one", async () => {
    const fetched = vi.fn(async () => new Response(new Blob(["%PDF"]), { status: 200 }));
    vi.stubGlobal("fetch", fetched);
    URL.createObjectURL = vi.fn(() => "blob:x");
    URL.revokeObjectURL = vi.fn();
    useStore.setState({ messages: [reply({ files: [{ ...deck, render_url: "/api/outputs/deck.pptx/preview?v=2" }] })] });
    render(<ChatPane />);
    fireEvent.click(screen.getByText("Preview"));
    await screen.findByTestId("rendered-preview");
    expect(fetched).toHaveBeenCalledWith("/api/outputs/deck.pptx/preview?v=2");
    vi.unstubAllGlobals();
  });

  it("falls back to the deck's text, saying why and that it isn't the layout", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "deck.pptx: LibreOffice stopped" }),
                                                          { status: 422, headers: { "content-type": "application/json" } })));
    useStore.setState({ messages: [reply({ files: [deck] })] });
    render(<ChatPane />);
    fireEvent.click(screen.getByText("Preview"));
    await screen.findByText("A slide");
    const preview = screen.getByTestId("file-preview").textContent;
    expect(preview).toContain("LibreOffice couldn't render it (deck.pptx: LibreOffice stopped)");
    expect(preview).toContain("layout in the file itself differs");
    expect(screen.queryByTestId("rendered-preview")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("tells the reader that tool results leave the machine in deep mode", () => {
    useStore.setState({ mode: "deep" });
    render(<ChatPane />);
    const note = screen.getByTestId("mode-note").textContent;
    expect(note).toContain("what tools return");
    expect(note).toContain("code output");
    expect(note).toContain("stay on this machine");
  });
});

describe("web search in the chat (Phase 5d)", () => {
  it("turns web search on for the next question with the Web button", () => {
    useStore.setState({ webNext: false });
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("web-toggle"));
    expect(useStore.getState().webNext).toBe(true);
    expect(screen.getByTestId("web-toggle").getAttribute("aria-pressed")).toBe("true");
  });

  it("hides the Web button for an option without web search", () => {
    useStore.setState({ options: OPTIONS.map((option) => ({ ...option, web: option.id === "fast" ? "off" : "tools" })) });
    render(<ChatPane />);
    expect(screen.queryByTestId("web-toggle")).toBeNull();
    cleanup();
    useStore.setState({ mode: "deep" });
    render(<ChatPane />);
    expect(screen.getByTestId("web-toggle")).toBeTruthy();
  });

  it("lists the pages an answer read, with links, apart from the library's sources", () => {
    const pages = [{ number: 1, title: "Review of Particle Physics", url: "https://pdglive.lbl.gov/x", fetched: "2026-11-04" },
                   { number: 2, title: "A blog", url: "https://blog.example.org/y", fetched: "2026-11-04" }];
    useStore.setState({ messages: [reply({ web: true, webSources: pages, webCitations: [1], text: "m [Web 1]." })] });
    render(<ChatPane />);
    const list = screen.getByTestId("web-sources");
    expect(list.textContent).toContain("Review of Particle Physics · pdglive.lbl.gov");
    expect(list.textContent).toContain("Grey: read but not cited.");
    expect(screen.getByTestId("web-on")).toBeTruthy();
    expect(screen.getByTestId("web-citation").getAttribute("href")).toBe("https://pdglive.lbl.gov/x");
  });

  it("offers a proposed search as a button and searches only on the click", () => {
    const searchWeb = vi.fn(async () => undefined);
    useStore.setState({ searchWeb, messages: [reply({ webProposal: { query: "KATRIN limit 2026", reason: "Not in the library." } })] });
    render(<ChatPane />);
    expect(searchWeb).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Search the web for “KATRIN limit 2026”"));
    expect(searchWeb).toHaveBeenCalledTimes(1);
  });
});

describe("Check in the chat (verification layer)", () => {
  const claim = (line: string, verdict: "passed" | "failed" | "undecided", shown: string = verdict) =>
    ({ line, label: "", verdict, shown, note: verdict === "passed" ? "" : "why" });

  it("turns Check on for the next question, and says what it does for the option", () => {
    useStore.setState({ checkNext: false, mode: "fast",
      options: OPTIONS.map((option) => ({ ...option, check_samples: option.id === "fast" ? 3 : 1,
        check_ceiling_seconds: option.id === "fast" ? 240 : null })) });
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("check-toggle"));
    expect(useStore.getState().checkNext).toBe(true);
    expect(screen.getByTestId("check-hint").textContent).toContain("up to 3 attempts, at most 4 minutes");
    cleanup();
    useStore.setState({ mode: "deep" });
    render(<ChatPane />);
    expect(screen.getByTestId("check-hint").textContent).toContain("checked with SymPy and shown with it");
  });

  it("shows the badge with each claim and how it passed, never 'correct'", () => {
    const checks = { badge: "2 of 3 checked claims passed", source: "answer" as const, failure: null, block: null,
      claims: [claim("x = 1", "passed", "passed (to stated digits)"), claim("t = 3 [s]", "passed", "passed (natural units)"),
               claim("y = 2", "failed")] };
    useStore.setState({ messages: [reply({ text: "An answer.", check: true, checks })] });
    render(<ChatPane />);
    const badge = screen.getByTestId("check-badge");
    expect(screen.getByTestId("check-badge-text").textContent).toBe("2 of 3 checked claims passed");
    expect(badge.textContent).toContain("passed (to stated digits)");
    expect(badge.textContent).toContain("passed (natural units)");
    expect(badge.textContent?.toLowerCase()).not.toContain("correct");
  });

  it("keeps replaced attempts collapsed, with the claims that failed in view", () => {
    const failed = { badge: "0 of 1 checked claim passed", source: "answer" as const, failure: null, block: null,
      claims: [claim("y = 2", "failed")] };
    useStore.setState({ messages: [reply({ text: "Second.", check: true,
      checks: { ...failed, badge: "1 of 1 checked claim passed", claims: [claim("y = 3", "passed")], attempt: 2, attempts: 2 },
      replaced: [{ attempt: 1, answer: "First.", checks: failed, failing: ["y = 2"] }] })] });
    render(<ChatPane />);
    expect(screen.getByTestId("replaced-attempts").textContent).toContain("Attempt 1, replaced: 0 of 1 checked claim passed");
    expect(screen.getByTestId("replaced-failing").textContent).toContain("y = 2");
    expect(screen.getByTestId("replaced-attempts").querySelector("details")?.open).toBe(false);
  });

  it("says when no attempt passed, or the time ran out", () => {
    const checks = { badge: "1 of 2 checked claims passed", source: "answer" as const, failure: null, block: null,
      claims: [], attempt: 1, attempts: 2, samples: 3, none_passed: true, stopped: "time" };
    useStore.setState({ messages: [reply({ text: "Best.", check: true, checks })] });
    render(<ChatPane />);
    expect(screen.getByTestId("check-none-passed").textContent).toContain("None of the 2 attempts passed every check");
    expect(screen.getByTestId("check-stopped").textContent).toContain("Stopped after 2 of 3 attempts: the time limit");
  });

  it("shows the status line while it checks", () => {
    useStore.setState({ messages: [reply({ streaming: true, finished: false, check: true,
      checkStatus: "A claim failed: trying again (2/3)" })] });
    render(<ChatPane />);
    expect(screen.getByTestId("check-status").textContent).toBe("A claim failed: trying again (2/3)");
  });
});

describe("Check badges after round 19", () => {
  const claim = (line: string, verdict: "passed" | "failed") => ({ line, label: "", verdict, shown: verdict, note: "" });
  const checks = (claims: ReturnType<typeof claim>[]) => ({
    badge: `${claims.filter((c) => c.verdict === "passed").length} of ${claims.length} checked claims passed`,
    source: "answer" as const, failure: null, block: null, claims });
  const withBadges = () => OPTIONS.map((option) => ({
    ...option, check_samples: 1,
    check_badge: (option.id === "fast" || option.id === "balanced" ? "limited"
      : option.id === "qwen" ? "unmeasured" : "full") as "limited" | "unmeasured" | "full" }));

  it("is neutral in Fast, with a note that most of the answer isn't checkable", () => {
    useStore.setState({ options: withBadges(),
      messages: [reply({ option: "fast", text: "A.", check: true, checks: checks([claim("a = 1", "passed"), claim("b = 2", "passed")]) })] });
    render(<ChatPane />);
    expect(screen.getByTestId("check-badge-text").textContent)
      .toBe("2 claims checked, both passed. Most of this answer isn't checkable.");
    expect(screen.getByTestId("check-badge").className).toContain("bg-stone-50");
  });

  it("is neutral in Qwen, with a note that its checks haven't been measured", () => {
    useStore.setState({ options: withBadges(),
      messages: [reply({ option: "qwen", text: "A.", check: true, checks: checks([claim("a = 1", "passed")]) })] });
    render(<ChatPane />);
    expect(screen.getByTestId("check-badge-text").textContent)
      .toBe("1 claim checked, it passed. How well Qwen's claims can be checked hasn't been measured yet.");
    expect(screen.getByTestId("check-badge").className).toContain("bg-stone-50");
  });

  it("stays the normal badge in Deep", () => {
    useStore.setState({ options: withBadges(),
      messages: [reply({ option: "deep", text: "A.", check: true, checks: checks([claim("a = 1", "passed")]) })] });
    render(<ChatPane />);
    expect(screen.getByTestId("check-badge-text").textContent).toBe("1 of 1 checked claims passed");
    expect(screen.getByTestId("check-badge").className).toContain("bg-teal-50");
  });

  it("says so in the Check hint for Fast", () => {
    useStore.setState({ options: withBadges(), mode: "fast", checkNext: true });
    render(<ChatPane />);
    expect(screen.getByTestId("check-hint").textContent).toContain("Most of a Fast answer isn't checkable.");
    expect(screen.getByTestId("check-hint").textContent).not.toContain("attempts");
  });
});

describe("Ask about this (viewer selection)", () => {
  it("puts the quoted passage before what's typed, and switches to the chat", async () => {
    useStore.setState({ view: "feedback", chatDraft: null });
    render(<ChatPane />);
    const box = screen.getByLabelText("Your question") as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "Why?" } });
    const { act } = await import("@testing-library/react");
    act(() => useStore.getState().prefillChat("> proper time\n\n(Book, p. 3)\n\n"));
    expect(box.value).toBe("> proper time\n\n(Book, p. 3)\n\nWhy?");
    expect(useStore.getState().view).toBe("chat");
  });
});

describe("the reader's notes as sources (roadmap item 4)", () => {
  const source = (change: object) => ({ id: 1, chunk_id: -4, doc_id: 7, doc_title: "Your note on Book A", page_start: 12,
    page_end: 12, focus_page: 12, heading_path: [], regions: [{ page: 12, bbox: [1, 2, 3, 4] }], kind: "note", note_id: 4, ...change });

  it("offers the Notes scope", () => {
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("scope-notes"));
    expect(useStore.getState().scopeMode).toBe("notes");
  });

  it("opens an anchored note's passage, and the note itself on request", () => {
    useStore.setState({ messages: [reply({ text: "As you noted [S1].", sources: [source({})] as never, citations: [1] })] });
    render(<ChatPane />);
    expect(screen.getByTestId("source").textContent).toContain("Your note on Book A · p. 12");
    fireEvent.click(screen.getByTestId("source"));
    expect(useStore.getState().viewer?.docId).toBe(7);
    fireEvent.click(screen.getByTestId("source-open-note"));
    expect(useStore.getState().view).toBe("notes");
    expect(useStore.getState().shownNote?.id).toBe(4);
  });

  it("opens a free note in the notes", () => {
    useStore.setState({ viewer: null, messages: [reply({ text: "As you noted [S1].",
      sources: [source({ doc_id: 0, doc_title: "Your note", page_start: null, page_end: null, focus_page: null, regions: [] })] as never,
      citations: [1] })] });
    render(<ChatPane />);
    expect(screen.getByTestId("source").textContent).toBe("S1Your note");
    fireEvent.click(screen.getByTestId("source"));
    expect(useStore.getState().view).toBe("notes");
    expect(useStore.getState().viewer).toBeNull();
  });
});

describe("suggested highlights from the chat (roadmap item 4)", () => {
  it("opens them in the viewer", () => {
    useStore.setState({ viewer: null, messages: [reply({ text: "I proposed three.", tools: [{ id: "1", name: "annotate_pdf",
      running: false, ok: true, summary: "Suggested 3 highlights in Book", suggested: { doc_id: 4, page: 12, count: 3 } }] })] });
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("open-suggestions"));
    expect(useStore.getState().viewer).toMatchObject({ docId: 4, page: 12 });
  });
});
