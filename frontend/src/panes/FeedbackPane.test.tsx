import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FeedbackEntry } from "../api";
import { useStore } from "../store/store";
import type { Message } from "../store/store";
import { ChatPane, canRate } from "./ChatPane";
import { FeedbackPane, localTime, sourcesInOrder } from "./FeedbackPane";

const reply = (change: Partial<Message>): Message => ({
  id: 1, role: "assistant", text: "An answer", thinking: "", sources: [], citations: [], unverified: [],
  error: null, truncated: false, streaming: false, finished: true,
  model: "gpt-oss:20b", hosted: false, fallbackFrom: null, fallbackReason: null, searchQuery: null,
  tools: [], files: [], option: "fast", servedBy: null, question: "Q?", startedAt: Date.now(), searched: true,
  cutOff: false, noAnswer: false, spend: null, images: [], web: false, webSources: [], webCitations: [], webProposal: null,
  questionId: 41, feedback: null, check: false, checkStatus: null, checks: null, replaced: [], ...change,
});

const entry = (change: Partial<FeedbackEntry>): FeedbackEntry => ({
  id: 5, question_id: 41, answer_id: 42, rating: "down", note: "Dropped a term", status: "open", candidate: false,
  question: "Derive the geodesic equation", answer: "From the action [S2] and [Web 1].", error: null,
  option: "deep", model: "claude-opus-5-5", hosted: true, served_by: null,
  sources: [
    { id: 1, doc_id: 7, doc_title: "Spacetime", page_start: 10, page_end: 10, focus_page: 10, heading_path: ["3"], regions: [], cited: false },
    { id: 2, doc_id: 7, doc_title: "Spacetime", page_start: 40, page_end: 41, focus_page: 40, heading_path: ["3.2 Geodesics"], regions: [], cited: true },
  ],
  web_sources: [{ number: 1, title: "Notes", url: "https://example.org/g", fetched: "2026-11-05", cited: true }],
  asked_at: "2026-11-05 13:00:00", answered_at: "2026-11-05 13:00:30", created_at: "2026-11-05 13:01:00",
  updated_at: "2026-11-05 13:01:00", ...change,
});

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  useStore.setState({ messages: [], sending: false, viewer: null, view: "chat", options: [] });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("the thumbs under an answer", () => {
  it("are offered once an answer has finished or failed, and only for a stored question", () => {
    expect(canRate(reply({}))).toBe(true);
    expect(canRate(reply({ finished: false, error: "The model failed" }))).toBe(true);
    expect(canRate(reply({ streaming: true, finished: false }))).toBe(false);
    expect(canRate(reply({ finished: false }))).toBe(false);          // stopped by the reader
    expect(canRate(reply({ questionId: null }))).toBe(false);
  });

  it("save a rating at once, then the note with its own button", async () => {
    const fetch = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      return json(entry({ rating: body.rating, note: body.note }));
    });
    vi.stubGlobal("fetch", fetch);
    useStore.setState({ messages: [reply({})] });
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("rate-down"));
    await screen.findByTestId("feedback-saved");
    expect(fetch.mock.calls[0][0]).toBe("/api/questions/41/feedback");
    expect(fetch.mock.calls[0][1]?.method).toBe("PUT");
    expect(screen.getByTestId("rate-down").getAttribute("aria-pressed")).toBe("true");

    fireEvent.change(screen.getByLabelText("Feedback note"), { target: { value: "Lost the factor 2" } });
    expect(screen.queryByTestId("feedback-saved")).toBeNull();
    fireEvent.click(screen.getByTestId("save-note"));
    await screen.findByTestId("feedback-saved");
    expect(JSON.parse(String(fetch.mock.calls[1][1]?.body))).toEqual({ rating: "down", note: "Lost the factor 2" });
  });

  it("say so when the server refuses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ detail: "No question 41" }, 404)));
    useStore.setState({ messages: [reply({})] });
    render(<ChatPane />);
    fireEvent.click(screen.getByTestId("rate-up"));
    expect((await screen.findByRole("alert")).textContent).toContain("No question 41");
  });
});

describe("the review list", () => {
  const counts = { total: 2, up: 1, down: 1, open: 2, candidates: 0 };

  it("opens from the chat header and goes back, keeping a half-typed question", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ entries: [], counts: { ...counts, total: 0 } })));
    render(<ChatPane />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "half a question" } });
    fireEvent.click(screen.getByTestId("open-feedback"));
    expect(await screen.findByText(/No feedback yet/)).toBeTruthy();
    fireEvent.click(screen.getByTestId("back-to-chat"));
    expect((screen.getByLabelText("Your question") as HTMLTextAreaElement).value).toBe("half a question");
  });

  it("lists open entries first, and filters by rating", async () => {
    const fetch = vi.fn(async (_url: string) => json({ entries: [entry({})], counts }));
    vi.stubGlobal("fetch", fetch);
    render(<FeedbackPane />);
    expect((await screen.findAllByTestId("feedback-entry")).length).toBe(1);
    expect(fetch.mock.calls[0][0]).toBe("/api/feedback?status=open");
    expect(screen.getByTestId("filter-0").textContent).toBe("Open 2");
    fireEvent.click(screen.getByTestId("filter-1"));
    await waitFor(() => expect(fetch.mock.calls.at(-1)?.[0]).toBe("/api/feedback?rating=down"));
  });

  it("shows the answer and its sources, opens a cited page, and makes a gold candidate", async () => {
    const fetch = vi.fn(async (_url: string, init?: RequestInit) =>
      init?.method === "PATCH" ? json(entry({ candidate: true })) : json({ entries: [entry({})], counts }));
    vi.stubGlobal("fetch", fetch);
    render(<FeedbackPane />);
    fireEvent.click((await screen.findAllByRole("button", { expanded: false }))[0]);
    const detail = screen.getByTestId("feedback-detail");
    expect(detail.textContent).toContain("From the action");
    const sources = screen.getByTestId("feedback-sources").querySelectorAll("li");
    expect(sources[0].textContent).toContain("S2");             // cited first
    expect(sources[2].querySelector("a")?.getAttribute("href")).toBe("https://example.org/g");
    fireEvent.click(sources[0].querySelector("button")!);
    expect(useStore.getState().viewer).toMatchObject({ docId: 7, page: 40 });

    fireEvent.click(screen.getByTestId("toggle-candidate"));
    await waitFor(() => expect(fetch.mock.calls.some((call) => call[1]?.method === "PATCH")).toBe(true));
    const patch = fetch.mock.calls.find((call) => call[1]?.method === "PATCH")!;
    expect(patch[0]).toBe("/api/feedback/5");
    expect(JSON.parse(String(patch[1]?.body))).toEqual({ candidate: true });
  });

  it("offers a gold candidate only for a thumbs-down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ entries: [entry({ rating: "up" })], counts })));
    render(<FeedbackPane />);
    fireEvent.click((await screen.findAllByRole("button", { expanded: false }))[0]);
    expect(screen.queryByTestId("toggle-candidate")).toBeNull();
    expect(screen.getByTestId("toggle-done").textContent).toBe("Mark done");
  });
});

describe("helpers", () => {
  it("put cited sources first", () => {
    expect(sourcesInOrder(entry({}).sources).map((source) => source.id)).toEqual([2, 1]);
  });

  it("read SQLite's time as UTC", () => {
    expect(localTime(null)).toBe("");
    expect(localTime("not a time")).toBe("not a time");
    const shown = localTime("2026-11-05 13:00:00");
    expect(shown).toContain("2026");
  });
});
