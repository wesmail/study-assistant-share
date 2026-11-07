import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Annotation } from "../api";
import { useStore } from "../store/store";
import { SuggestPanel } from "./SuggestPanel";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const suggestion = (id: number, batch = "b1"): Annotation => ({
  id, doc_id: 4, page: 12, rects: [[1, 2, 3, 4]], bbox: [1, 2, 3, 4], quote: `quote ${id}`, prefix: "", suffix: "",
  kind: "highlight", color: "blue", body: `note ${id}`, origin: "ai", status: "suggested", created_at: "", batch,
});

describe("the Suggest panel", () => {
  it("sends the instruction and pages, and reports what came back", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify({ batch: "b1", suggestions: [suggestion(1)],
      not_found: [{ page: 12, quote: "x", why: "not found" }], asked: 2, cost_usd: 0.031, option: "deep" })));
    vi.stubGlobal("fetch", fetch);
    useStore.setState({ suggestOptions: ["deep"], options: [] });
    const onChange = vi.fn(async () => {});
    const onGo = vi.fn();
    render(<SuggestPanel docId={4} page={12} pages={30} annotations={[]} onChange={onChange} onGo={onGo} onClose={() => {}} />);
    fireEvent.click(screen.getByText("Mark grammar mistakes and suggest corrections"));
    fireEvent.change(screen.getByTestId("suggest-last"), { target: { value: "13" } });
    fireEvent.click(screen.getByTestId("suggest-run"));
    await waitFor(() => expect(screen.getByTestId("suggest-report").textContent)
      .toBe("1 suggestion; 1 quote the model gave weren't in the text and were dropped · $0.031."));
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/suggestions");
    expect(JSON.parse(init.body as string)).toEqual({ doc_id: 4, first_page: 12, last_page: 13,
      instruction: "Mark grammar mistakes and suggest corrections", option: "deep" });
    expect(onChange).toHaveBeenCalled();
    expect(onGo).toHaveBeenCalledWith(12);
  });

  it("does nothing with a reply that comes back after another document was opened (review R2-C5)", async () => {
    let answer: (response: Response) => void = () => {};
    const fetch = vi.fn(() => new Promise<Response>((resolve) => { answer = resolve; }));
    vi.stubGlobal("fetch", fetch);
    useStore.setState({ suggestOptions: ["deep"], options: [] });
    const onChange = vi.fn(async () => {});
    const onGo = vi.fn();
    const { unmount } = render(<SuggestPanel docId={4} page={12} pages={30} annotations={[]} onChange={onChange} onGo={onGo}
      onClose={() => {}} />);
    fireEvent.click(screen.getByText("Mark grammar mistakes and suggest corrections"));
    fireEvent.click(screen.getByTestId("suggest-run"));
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    unmount();                     // the viewer opened book B: the panel is remade for it
    answer(new Response(JSON.stringify({ batch: "b1", suggestions: [suggestion(1)], not_found: [], asked: 1,
      cost_usd: 0.01, option: "deep" })));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(onChange).not.toHaveBeenCalled();
    expect(onGo).not.toHaveBeenCalled();
  });

  it("lists the waiting suggestions, accepts all of a batch", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify({ changed: 2 })));
    vi.stubGlobal("fetch", fetch);
    render(<SuggestPanel docId={4} page={12} pages={30} annotations={[suggestion(1), suggestion(2), { ...suggestion(3), status: "accepted" }]}
      onChange={async () => {}} onGo={() => {}} onClose={() => {}} />);
    expect(screen.getAllByTestId("suggestion")).toHaveLength(2);
    expect(screen.getAllByTestId("suggestion")[0].textContent).toContain("note 1");
    fireEvent.click(screen.getByTestId("accept-all"));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("/api/annotations/batch/b1", expect.objectContaining({ method: "POST" })));
  });
});
