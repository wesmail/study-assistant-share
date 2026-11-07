import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AnswerOption, ChatBody, SseEvent } from "../api";

const events: { current: SseEvent[]; bodies: ChatBody[] } = { current: [], bodies: [] };
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

vi.mock("../api", () => ({
  fetchLibrary: vi.fn(async () => ({ documents: [], settings: { hide_test_collection: false } })),
  saveLibrarySettings: vi.fn(async (settings: object) => ({ hide_test_collection: false, ...settings })),
  fetchOptions: vi.fn(async () => ({ options: OPTIONS, default: "fast", switch_seconds: [20, 55],
    month_spend: [{ provider: "anthropic", usd: 1.5, limit_usd: 30 }] })),
  rateAnswer: vi.fn(async (questionId: number, rating: string, note: string) => ({ id: 3, question_id: questionId, rating, note })),
  streamChat: vi.fn(async (body: ChatBody, onEvent: (event: SseEvent) => void) => {
    events.bodies.push(body);
    events.current.forEach(onEvent);
  }),
}));

const { modelBadge, offersDeepResend, searchedDocuments, useStore } = await import("./store");

const meta = (model: string, hosted: boolean): SseEvent => ({
  event: "meta",
  data: { conversation_id: 1, model, hosted },
});
const done = (model: string, hosted: boolean, done_reason = "end_turn"): SseEvent => ({
  event: "done",
  data: { citations: [], unverified: [], done_reason, model, hosted },
});
const answer = () => useStore.getState().messages.at(-1)!;

beforeEach(() => {
  events.current = [];
  events.bodies = [];
  useStore.setState({
    messages: [], conversationId: null, mode: "fast", localMode: "fast", sending: false, scope: [], scopeMode: "library",
    viewer: null, options: OPTIONS,
  });
});

describe("the mode", () => {
  it("is fast unless chosen", async () => {
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].mode).toBe("fast");
  });

  it("sends deep when chosen, then goes back to fast for the next question", async () => {
    useStore.getState().setMode("deep");
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].mode).toBe("deep");
    expect(useStore.getState().mode).toBe("fast");
    await useStore.getState().send("And again?");
    expect(events.bodies[1].mode).toBe("fast");
  });

  it("is kept when nothing is sent", async () => {
    useStore.getState().setMode("deep");
    await useStore.getState().send("   ");
    expect(useStore.getState().mode).toBe("deep");
  });
});

describe("the search scope", () => {
  const open = { docId: 7, page: 1, regions: [], opened: 1 };

  it("is the whole library by default", async () => {
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].doc_ids).toBeNull();
  });

  it("can be the document open in the viewer", async () => {
    useStore.setState({ viewer: open });
    useStore.getState().setScopeMode("document");
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].doc_ids).toEqual([7]);
  });

  it("follows the viewer when another document is opened", async () => {
    useStore.setState({ viewer: open });
    useStore.getState().setScopeMode("document");
    useStore.getState().openPage(9, 1);
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].doc_ids).toEqual([9]);
  });

  it("switches to the ticked documents when one is ticked", async () => {
    useStore.getState().toggleScope(3);
    expect(useStore.getState().scopeMode).toBe("ticked");
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].doc_ids).toEqual([3]);
  });

  it("goes back to the library when the last tick is removed", () => {
    useStore.getState().toggleScope(3);
    useStore.getState().toggleScope(3);
    expect(useStore.getState().scopeMode).toBe("library");
  });

  it("keeps the ticks while another scope is in use", async () => {
    useStore.getState().toggleScope(3);
    useStore.getState().setScopeMode("library");
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].doc_ids).toBeNull();
    expect(useStore.getState().scope).toEqual([3]);
  });

  it("never narrows the search to nothing", () => {
    expect(searchedDocuments({ scope: [], scopeMode: "document", viewer: null })).toBeNull();
    expect(searchedDocuments({ scope: [], scopeMode: "ticked", viewer: null })).toBeNull();
  });

  it("stays when the mode resets after a question", async () => {
    useStore.getState().toggleScope(3);
    useStore.getState().setMode("deep");
    await useStore.getState().send("Hello?");
    expect(useStore.getState().mode).toBe("fast");
    expect(useStore.getState().scopeMode).toBe("ticked");
  });
});

describe("the badge", () => {
  it("says local or hosted, with the model", () => {
    expect(modelBadge({ model: "gpt-oss:20b", hosted: false })).toBe("local · gpt-oss:20b");
    expect(modelBadge({ model: "claude-opus-5-5", hosted: true })).toBe("hosted · claude-opus-5-5");
    expect(modelBadge({ model: null, hosted: false })).toBeNull();
  });

  it("belongs to each answer, so a conversation can mix them", async () => {
    events.current = [meta("gpt-oss:20b", false), done("gpt-oss:20b", false)];
    await useStore.getState().send("One?");
    events.current = [meta("claude-opus-5-5", true), done("claude-opus-5-5", true)];
    await useStore.getState().send("Two?");
    const answers = useStore.getState().messages.filter((message) => message.role === "assistant");
    expect(answers.map(modelBadge)).toEqual(["local · gpt-oss:20b", "hosted · claude-opus-5-5"]);
  });
});

describe("a fallback", () => {
  const fallback: SseEvent = {
    event: "fallback",
    data: { model: "claude-sonnet-5-5", from: "claude-opus-5-5", reason: "HTTP 529, overloaded_error" },
  };

  it("names the model that really answered, and why", async () => {
    events.current = [meta("claude-opus-5-5", true), fallback, done("claude-sonnet-5-5", true)];
    await useStore.getState().send("Hello?");
    expect(modelBadge(answer())).toBe("hosted · claude-sonnet-5-5");
    expect(answer().fallbackFrom).toBe("claude-opus-5-5");
    expect(answer().fallbackReason).toBe("HTTP 529, overloaded_error");
  });

  it("drops what the first model wrote, even text still waiting for a frame", async () => {
    events.current = [
      meta("claude-opus-5-5", true),
      { event: "thinking", data: { text: "first model's thoughts" } },
      { event: "token", data: { text: "half an ans" } },
      fallback,
      { event: "token", data: { text: "The whole answer." } },
      done("claude-sonnet-5-5", true),
    ];
    await useStore.getState().send("Hello?");
    expect(answer().text).toBe("The whole answer.");
    expect(answer().thinking).toBe("");
  });
});

describe("a cut-off answer", () => {
  it.each(["length", "max_tokens"])("is flagged for %s", async (reason) => {
    events.current = [done("model", false, reason)];
    await useStore.getState().send("Hello?");
    expect(answer().truncated).toBe(true);
  });
});

describe("tool events", () => {
  const file = { name: "a.pdf", kind: "pdf", size: 1, url: "/api/outputs/a.pdf", preview: null };

  it("shows a tool as running, then how it ended", async () => {
    events.current = [
      { event: "tool_start", data: { id: "1", name: "run_python" } },
      { event: "tool_end", data: { id: "1", name: "run_python", ok: false, summary: "Exit code 1" } },
    ];
    await useStore.getState().send("Run it");
    expect(answer().tools).toEqual([{ id: "1", name: "run_python", running: false, ok: false, summary: "Exit code 1" }]);
  });

  it("takes a failed document's text back out of the answer, so a retry shows once", async () => {
    events.current = [
      { event: "token", data: { text: "Here it is.\n\n" } },
      { event: "token", data: { text: "# Summary\nFirst draft" } },
      { event: "tool_end", data: { id: "b1", name: "make_pdf", ok: false, summary: "make_pdf failed: no such image" } },
      { event: "withdraw", data: { keep: 13 } },
      { event: "token", data: { text: "# Summary\nSecond draft" } },
    ];
    await useStore.getState().send("A PDF, please");
    expect(answer().text).toBe("Here it is.\n\n# Summary\nSecond draft");
  });

  it("removes only a rebuilt file's earlier document", async () => {
    events.current = [
      { event: "token", data: { text: "# S\nLong" } },
      { event: "token", data: { text: "\n\nShortening.\n\n# S\nShort" } },
      { event: "withdraw", data: { keep: 0, end: 8 } },
    ];
    await useStore.getState().send("One page");
    expect(answer().text).toBe("\n\nShortening.\n\n# S\nShort");
  });

  it("collects files in order", async () => {
    events.current = [{ event: "file", data: file }, { event: "file", data: { ...file, name: "b.pdf" } }];
    await useStore.getState().send("Make them");
    expect(answer().files.map((item) => item.name)).toEqual(["a.pdf", "b.pdf"]);
  });

  it("replaces a file that a retry rebuilt", async () => {
    events.current = [{ event: "file", data: file }, { event: "file", data: { ...file, size: 2 } }];
    await useStore.getState().send("Make it shorter");
    expect(answer().files.map((item) => item.size)).toEqual([2]);
  });

  it("adds a mid-answer search's passages to the sources", async () => {
    const source = (id: number) => ({ id, chunk_id: id, doc_id: 1, doc_title: "B", page_start: 1, page_end: 1,
      focus_page: 1, heading_path: [], regions: [] });
    events.current = [
      { event: "sources", data: { sources: [source(1)] } },
      { event: "more_sources", data: { sources: [source(2)] } },
    ];
    await useStore.getState().send("Search more");
    expect(answer().sources.map((item) => item.id)).toEqual([1, 2]);
  });

  it("drops the first model's files and tools on a fallback", async () => {
    events.current = [
      { event: "tool_start", data: { id: "1", name: "make_pdf" } },
      { event: "file", data: file },
      { event: "fallback", data: { model: "claude-sonnet-5-5", from: "claude-opus-5-5", reason: "it refused" } },
    ];
    await useStore.getState().send("Make it");
    expect(answer().files).toEqual([]);
    expect(answer().tools).toEqual([]);
  });
});

describe("the selector", () => {
  it("loads the options in config order, with the default", async () => {
    useStore.setState({ options: [], mode: "gone", localMode: "gone" });
    await useStore.getState().loadOptions();
    expect(useStore.getState().options.map((option) => option.id)).toEqual(["fast", "balanced", "qwen", "deep"]);
    expect(useStore.getState().mode).toBe("fast");
  });

  it("keeps a local choice, so the model isn't reloaded for each question", async () => {
    useStore.getState().setMode("balanced");
    await useStore.getState().send("One?");
    await useStore.getState().send("Two?");
    expect(events.bodies.map((body) => body.mode)).toEqual(["balanced", "balanced"]);
  });

  it.each(["qwen", "deep"])("goes back to the last local choice after a %s question", async (hosted) => {
    useStore.getState().setMode("balanced");
    await useStore.getState().send("Local?");
    useStore.getState().setMode(hosted);
    await useStore.getState().send("Hosted?");
    expect(events.bodies[1].mode).toBe(hosted);
    expect(useStore.getState().mode).toBe("balanced");
  });

  it("records which option each answer was asked with", async () => {
    useStore.getState().setMode("qwen");
    await useStore.getState().send("Hello?");
    expect(answer().option).toBe("qwen");
    expect(answer().question).toBe("Hello?");
  });
});

describe("a Qwen answer that stops", () => {
  const cutOff: SseEvent = { event: "error", data: { message: "stopped at its 16,000-token limit", cut_off: true } };

  it("is flagged as cut off while thinking", async () => {
    useStore.getState().setMode("qwen");
    events.current = [meta("qwen/qwen3.8-27b", true), cutOff];
    await useStore.getState().send("Hard?");
    expect(answer().cutOff).toBe(true);
    expect(offersDeepResend(answer())).toBe(true);
  });

  it("is sent again in Deep only on a click, with the same question", async () => {
    useStore.getState().setMode("qwen");
    events.current = [meta("qwen/qwen3.8-27b", true), cutOff];
    await useStore.getState().send("Hard?");
    const stopped = answer();
    expect(events.bodies.length).toBe(1);
    events.current = [meta("claude-opus-5-5", true), done("claude-opus-5-5", true)];
    await useStore.getState().resendDeep(stopped.id);
    expect(events.bodies.map((body) => [body.message, body.mode])).toEqual([["Hard?", "qwen"], ["Hard?", "deep"]]);
    expect(modelBadge(answer())).toBe("hosted · claude-opus-5-5");
    expect(useStore.getState().mode).toBe("fast");
  });

  it("offers Deep for a failure or a cut-off answer, and never for other options", () => {
    const base = { option: "qwen", error: null, truncated: false, streaming: false, cutOff: false, noAnswer: false };
    expect(offersDeepResend({ ...base, error: "HTTP 503" })).toBe(true);
    expect(offersDeepResend({ ...base, truncated: true })).toBe(true);
    expect(offersDeepResend(base)).toBe(false);
    expect(offersDeepResend({ ...base, error: "x", streaming: true })).toBe(false);
    expect(offersDeepResend({ ...base, option: "fast", error: "x" })).toBe(false);
  });
});

describe("the badge for a routed model", () => {
  it("names the provider that served it", async () => {
    events.current = [meta("qwen/qwen3.8-27b", true),
      { event: "done", data: { citations: [], unverified: [], done_reason: "stop", model: "qwen/qwen3.8-27b",
        hosted: true, served_by: "DeepInfra" } }];
    await useStore.getState().send("Hello?");
    expect(modelBadge(answer())).toBe("hosted · qwen/qwen3.8-27b · DeepInfra");
  });
});

describe("hosted spend", () => {
  it("loads this month's spend with the options", async () => {
    await useStore.getState().loadOptions();
    expect(useStore.getState().monthSpend).toEqual([{ provider: "anthropic", usd: 1.5, limit_usd: 30 }]);
  });

  it("keeps the logged cost on the answer, also when it failed", async () => {
    const spend = { calls: 1, failed_calls: 1, cost_usd: 0.03, input_tokens: 9, output_tokens: 16000 };
    useStore.getState().setMode("qwen");
    events.current = [meta("qwen/qwen3.8-27b", true),
      { event: "error", data: { message: "stopped", cut_off: true, spend } }];
    await useStore.getState().send("Hard?");
    expect(answer().spend).toEqual(spend);
  });
});

describe("a Balanced answer that stops at its limit", () => {
  const limit: SseEvent = { event: "error", data: {
    message: "gemma4:26b-a4b-it-qat stopped at its 8,192-token limit, thinking included, before it wrote an answer.",
    cut_off: true } };

  it("offers Deep, and sends the same question there only when asked", async () => {
    useStore.getState().setMode("balanced");
    events.current = [meta("gemma4:26b-a4b-it-qat", false), limit];
    await useStore.getState().send("Derive it?");
    const stopped = answer();
    expect(offersDeepResend(stopped)).toBe(true);
    expect(events.bodies.length).toBe(1);
    expect(useStore.getState().mode).toBe("balanced");
    events.current = [meta("claude-opus-5-5", true), done("claude-opus-5-5", true)];
    await useStore.getState().resendDeep(stopped.id);
    expect(events.bodies.map((body) => [body.message, body.mode])).toEqual([["Derive it?", "balanced"], ["Derive it?", "deep"]]);
    // Deep was one question only: the selector stays on Balanced, not Fast.
    expect(useStore.getState().mode).toBe("balanced");
  });

  it("offers Deep for a cut-off answer, but not for a failure that isn't the limit, or for Fast", () => {
    const base = { option: "balanced", error: null, truncated: false, streaming: false, cutOff: false, noAnswer: false };
    expect(offersDeepResend({ ...base, truncated: true })).toBe(true);
    expect(offersDeepResend({ ...base, error: "The model returned no answer (stop)", noAnswer: true })).toBe(true);
    expect(offersDeepResend({ ...base, error: "Ollama returned 500", cutOff: false })).toBe(false);
    expect(offersDeepResend(base)).toBe(false);
    expect(offersDeepResend({ ...base, option: "fast", error: "limit", cutOff: true })).toBe(false);
  });
});

describe("a Balanced reply with no answer", () => {
  it("offers Deep when only tool lines came back", async () => {
    useStore.getState().setMode("balanced");
    events.current = [
      meta("gemma4:26b-a4b-it-qat", false),
      { event: "error", data: { message: "The model returned no answer (stop)", cut_off: false, no_answer: true } },
    ];
    await useStore.getState().send("Derive it?");
    expect(answer().noAnswer).toBe(true);
    expect(offersDeepResend(answer())).toBe(true);
  });
});

describe("the reader's images", () => {
  it("go with the question and stay with it in the chat", async () => {
    const image = { media_type: "image/png", data: "AAAA" };
    await useStore.getState().send("What is this?", "balanced", [image]);
    expect(events.bodies[0].images).toEqual([image]);
    expect(useStore.getState().messages[0].images).toEqual([image]);
  });

  it("aren't sent when there are none", async () => {
    await useStore.getState().send("Hello?");
    expect("images" in events.bodies[0]).toBe(false);
  });
});

describe("web search, one question at a time (Phase 5d)", () => {
  it("never sends web for an option without it, even with the toggle left on", async () => {
    useStore.setState({ options: OPTIONS.map((option) => ({ ...option, web: option.id === "fast" ? "off" : "tools" })),
                        webNext: true });
    await useStore.getState().send("Latest PyTorch?");
    expect(events.bodies[0].web).toBeUndefined();
    expect(answer().web).toBe(false);
  });

  it("is off unless turned on, and turns itself off once the question is sent", async () => {
    await useStore.getState().send("Q1");
    expect(events.bodies[0].web).toBeUndefined();
    useStore.getState().setWebNext(true);
    await useStore.getState().send("Q2");
    expect(events.bodies[1].web).toBe(true);
    expect(useStore.getState().webNext).toBe(false);
    expect(answer().web).toBe(true);
    await useStore.getState().send("Q3");
    expect(events.bodies[2].web).toBeUndefined();
  });

  it("collects the pages an answer reads and which it cites", async () => {
    const page = { number: 1, title: "PDG", url: "https://pdg.example.org/w", fetched: "2026-11-04" };
    events.current = [
      { event: "web_source", data: page },
      { event: "done", data: { ...done("m", false).data, web_sources: [page], web_citations: [1], web_proposal: null } },
    ];
    useStore.getState().setWebNext(true);
    await useStore.getState().send("W mass?");
    expect(answer().webSources).toEqual([page]);
    expect(answer().webCitations).toEqual([1]);
  });

  it("asks again with the proposed search only when the reader clicks, with the same option", async () => {
    const proposal = { query: "PyTorch latest release", reason: "Releases aren't in the library." };
    events.current = [{ event: "web_proposal", data: proposal }, done("m", false)];
    await useStore.getState().send("Latest PyTorch?", "balanced");
    expect(answer().webProposal).toEqual(proposal);
    expect(events.bodies).toHaveLength(1);                          // nothing searched yet
    events.current = [done("m", false)];
    await useStore.getState().searchWeb(answer().id);
    expect(events.bodies[1]).toMatchObject({ message: "Latest PyTorch?", mode: "balanced", web: true,
                                             web_query: "PyTorch latest release" });
  });
});

describe("feedback", () => {
  beforeEach(() => {
    useStore.setState({ messages: [], conversationId: null, sending: false, options: OPTIONS, view: "chat" });
  });

  it("keeps the stored question's id from the meta event, and saves a rating against it", async () => {
    events.current = [{ event: "meta", data: { conversation_id: 1, question_id: 41, model: "gpt-oss:20b", hosted: false } },
                      done("gpt-oss:20b", false)];
    await useStore.getState().send("What is proper time?");
    expect(answer().questionId).toBe(41);
    await useStore.getState().rate(answer().id, "down", "Wrong sign");
    const api = await import("../api");
    expect(api.rateAnswer).toHaveBeenCalledWith(41, "down", "Wrong sign");
    expect(answer().feedback).toMatchObject({ id: 3, rating: "down", note: "Wrong sign" });
  });

  it("a new chat leaves the review list for the chat", () => {
    useStore.setState({ view: "feedback" });
    useStore.getState().newChat();
    expect(useStore.getState().view).toBe("chat");
  });
});

describe("Check (verification layer)", () => {
  const claim = (line: string, verdict: string, shown = verdict) => ({ line, label: "", verdict, shown, note: "" });
  const report = (badge: string, claims: object[], extra: object = {}) => ({
    badge, source: "answer", failure: null, claims, block: "...", ...extra,
  });

  it("is for the next question only, like Web, and goes with the request", async () => {
    useStore.setState({ checkNext: true });
    await useStore.getState().send("Derive it");
    expect(events.bodies.at(-1)?.check).toBe(true);
    expect(answer().check).toBe(true);
    expect(useStore.getState().checkNext).toBe(false);
    await useStore.getState().send("And again");
    expect(events.bodies.at(-1)?.check).toBeUndefined();
  });

  it("keeps a replaced attempt with its failing claim and streams the next in its place", async () => {
    const source = { id: 1, chunk_id: 1, doc_id: 1, doc_title: "B", page_start: 1, page_end: 1, focus_page: 1,
      heading_path: [], regions: [] };
    const first = report("1 of 2 checked claims passed", [claim("a = 1", "passed"), claim("b = 2", "failed")]);
    events.current = [
      { event: "sources", data: { sources: [source] } },
      { event: "token", data: { text: "First try." } },
      { event: "check_status", data: { stage: "checking", attempt: 1, of: 3 } },
      { event: "attempt_replaced", data: { attempt: 1, answer: "First try.", checks: first, failing: ["b = 2"],
        next: 2, of: 3, sources: [source] } },
      { event: "token", data: { text: "Second try." } },
      { event: "done", data: { citations: [], unverified: [], done_reason: "stop", model: "m", hosted: false,
        answer: "Second try.", checks: report("2 of 2 checked claims passed", [claim("a = 1", "passed")],
          { attempt: 2, attempts: 2, samples: 3 }),
        replaced: [{ attempt: 1, answer: "First try.", checks: first, failing: ["b = 2"] }] } },
    ];
    await useStore.getState().send("Derive it");
    expect(answer().text).toBe("Second try.");
    expect(answer().checks?.badge).toBe("2 of 2 checked claims passed");
    expect(answer().replaced.map((item) => item.failing)).toEqual([["b = 2"]]);
    expect(answer().checkStatus).toBeNull();
  });

  it("shows the attempt the server chose, even an earlier one than the last to stream", async () => {
    events.current = [
      { event: "token", data: { text: "Attempt two, worse." } },
      { event: "done", data: { citations: [], unverified: [], done_reason: "stop", model: "m", hosted: false,
        answer: "Attempt one, fewer failures.", checks: report("1 of 2 checked claims passed", [], { none_passed: true }),
        replaced: [] } },
    ];
    await useStore.getState().send("Derive it");
    expect(answer().text).toBe("Attempt one, fewer failures.");
  });

  it("writes the status lines", async () => {
    const { checkingLine, retryLine } = await import("./store");
    expect(checkingLine(1, 1)).toBe("Checking the claims…");
    expect(checkingLine(2, 3)).toBe("Checking the claims (attempt 2 of 3)…");
    expect(retryLine(2, 3, 1)).toBe("A claim failed: trying again (2/3)");
    expect(retryLine(3, 3, 2)).toBe("2 claims failed: trying again (3/3)");
  });
});

describe("the neutral badge (Fast and Balanced)", () => {
  it("counts the claims checked and never says correct", async () => {
    const { limitedBadge } = await import("./store");
    const v = (...verdicts: string[]) => verdicts.map((verdict) => ({ verdict }));
    expect(limitedBadge(v())).toBe("No claims could be checked.");
    expect(limitedBadge(v("passed"))).toBe("1 claim checked, it passed.");
    expect(limitedBadge(v("passed", "passed"))).toBe("2 claims checked, both passed.");
    expect(limitedBadge(v("passed", "passed", "passed"))).toBe("3 claims checked, all passed.");
    expect(limitedBadge(v("passed", "failed", "undecided"))).toBe("3 claims checked, 1 passed.");
  });
});

describe("the Notes scope (roadmap item 4)", () => {
  it("asks for the notes only, and no documents", async () => {
    useStore.setState({ scopeMode: "notes", scope: [7] });
    await useStore.getState().send("What did I note about Noether?");
    expect(events.bodies[0].notes_only).toBe(true);
    expect(events.bodies[0].doc_ids).toBeNull();
  });

  it("sends nothing about notes otherwise", async () => {
    await useStore.getState().send("Hello?");
    expect(events.bodies[0].notes_only).toBeUndefined();
  });
});

describe("annotate_pdf in the chat (roadmap item 4)", () => {
  it("keeps where the suggestions are", async () => {
    events.current = [
      { event: "tool_start", data: { id: "1", name: "annotate_pdf", arguments: {} } },
      { event: "tool_end", data: { id: "1", name: "annotate_pdf", ok: true, summary: "Suggested 3 highlights in Book",
        detail: { suggested: { doc_id: 4, page: 12, count: 3, batch: "b", not_found: 0 } } } },
    ];
    await useStore.getState().send("Highlight the definitions");
    expect(answer().tools[0].suggested).toEqual({ doc_id: 4, page: 12, count: 3, batch: "b", not_found: 0 });
  });
});

const api = await import("../api");

describe("New chat during an answer (review R2-C2)", () => {
  const token = (text: string): SseEvent => ({ event: "token", data: { text } } as SseEvent);
  // A stream that stays open until it is aborted, then sends what a slow network still had in flight.
  const openStream = (late: SseEvent[] = []) =>
    vi.mocked(api.streamChat).mockImplementationOnce(async (body, onEvent, signal) => {
      events.bodies.push(body);
      onEvent(meta("gpt-oss:20b", false));
      onEvent(token("Half an answer"));
      await new Promise<void>((resolve) => signal?.addEventListener("abort", () => resolve()));
      late.forEach(onEvent);
      throw new DOMException("The operation was aborted.", "AbortError");
    });

  it("leaves an empty chat that can be asked at once, whatever arrives late", async () => {
    openStream([token(" and more"), meta("gpt-oss:20b", false), done("gpt-oss:20b", false)]);
    const first = useStore.getState().send("First?");
    await Promise.resolve();
    expect(useStore.getState().messages).toHaveLength(2);
    useStore.getState().newChat();
    expect(useStore.getState().sending).toBe(false);
    await first;
    expect(useStore.getState().messages).toEqual([]);
    expect(useStore.getState().conversationId).toBeNull();
    expect(useStore.getState().sending).toBe(false);

    events.current = [meta("gpt-oss:20b", false), token("Fresh."), done("gpt-oss:20b", false)];
    await useStore.getState().send("Second?");
    expect(useStore.getState().messages.map((item) => item.text)).toEqual(["Second?", "Fresh."]);
  });

  it("doesn't let the old answer's end touch a question asked after New chat", async () => {
    openStream();
    const first = useStore.getState().send("First?");
    await Promise.resolve();
    useStore.getState().newChat();
    openStream();
    const second = useStore.getState().send("Second?");
    await Promise.resolve();
    await first;                                   // the old stream's finally runs now
    expect(useStore.getState().sending).toBe(true);   // the second question is still being answered
    expect(answer().streaming).toBe(true);
    useStore.getState().stop();
    await second;
    expect(useStore.getState().sending).toBe(false);
    expect(useStore.getState().messages.map((item) => item.text)).toEqual(["Second?", "Half an answer"]);
  });
});
