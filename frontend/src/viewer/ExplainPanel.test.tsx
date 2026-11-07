import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExplainResult } from "../api";
import { Markdown } from "../render/Markdown";
import { useStore } from "../store/store";
import { displayable, ExplainPanel } from "./ExplainPanel";

const result: ExplainResult = {
  kind: "equation", depth: "intermediate", text: "a &= b \\\\ c &= d", option: "deep", cost_usd: 0.012, from_book: true,
  items: [{ span: "a &= b \\\\", explanation: "First $a$." }, { span: "c &= d", explanation: "Then $c$." }],
};

beforeEach(() => {
  useStore.setState({ explanations: [], shownExplanation: null, mode: "deep",
    options: [{ id: "deep", label: "Deep" } as never] });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Explain", () => {
  it("makes a row of an aligned equation stand alone", () => {
    expect(displayable("a &= b \\\\")).toBe("a = b");
    expect(displayable("\\begin{aligned} x &= 1")).toBe("x = 1");
    expect(displayable("\\& stays")).toBe("\\& stays");
  });

  it("shows the spans and explanations side by side, and marks the pair under the pointer", () => {
    useStore.setState({ explanations: [{ id: 1, label: "p. 60", body: { text: "x" }, depth: "intermediate", result, error: null }],
      shownExplanation: 1 });
    render(<ExplainPanel />);
    expect(screen.getAllByTestId("explain-span")).toHaveLength(2);
    expect(screen.getAllByTestId("explain-text")[1].textContent).toContain("Then");
    expect(screen.getByTestId("explain-panel").textContent).toContain("the book's own LaTeX · Deep · $0.012");
    fireEvent.mouseEnter(screen.getAllByTestId("explain-text")[0]);
    expect(screen.getAllByTestId("explain-span")[0].className).toContain("bg-teal-50");
    fireEvent.click(screen.getByTestId("explain-close"));
    expect(useStore.getState().shownExplanation).toBeNull();
  });

  it("asks the server with the selected option, and a depth asks again", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify(result)));
    vi.stubGlobal("fetch", fetch);
    await useStore.getState().explain({ text: "x = 1", kind: "code" }, "code");
    const sent = JSON.parse((fetch.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(sent).toEqual({ text: "x = 1", kind: "code", depth: "intermediate", option: "deep" });
    render(<ExplainPanel />);
    fireEvent.click(screen.getByTestId("depth-intro"));
    await waitFor(() => expect(useStore.getState().explanations).toHaveLength(2));
    expect(useStore.getState().explanations[1].depth).toBe("intro");
  });

  it("shows a failure", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "The model failed" }), { status: 502 })));
    await useStore.getState().explain({ text: "x" }, "x");
    render(<ExplainPanel />);
    expect(screen.getByRole("alert").textContent).toBe("The model failed");
  });

  it("offers Explain on an answer's code blocks and display equations", () => {
    const onExplain = vi.fn();
    render(<Markdown text={"Code:\n\n```python\nx = 1\n```\n\n$$E = mc^2$$"} streaming={false} onCite={() => {}} onExplain={onExplain} />);
    fireEvent.click(screen.getByTestId("explain-code"));
    expect(onExplain).toHaveBeenLastCalledWith("x = 1", "code");
    fireEvent.click(screen.getByTestId("explain-equation"));
    expect(onExplain).toHaveBeenLastCalledWith("E = mc^2", "equation");
  });

  it("offers nothing while an answer streams", () => {
    render(<Markdown text={"```python\nx = 1\n```\n\n$$E = mc^2$$"} streaming onCite={() => {}} onExplain={() => {}} />);
    expect(screen.queryByTestId("explain-code")).toBeNull();
    expect(screen.queryByTestId("explain-equation")).toBeNull();
  });
});
