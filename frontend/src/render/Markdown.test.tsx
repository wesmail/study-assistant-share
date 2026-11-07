import { createRequire } from "node:module";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Markdown } from "./Markdown";

afterEach(cleanup);

const show = (text: string, onCite = vi.fn(), streaming = false) =>
  render(<Markdown text={text} streaming={streaming} onCite={onCite} />).container;

describe("Markdown", () => {
  it("renders the math the model writes with \\( and \\[", () => {
    const page = show("Inline \\(x^2\\) and\n\n\\[\ny = 1\n\\]");
    expect(page.querySelectorAll(".katex")).toHaveLength(2);
    expect(page.querySelectorAll(".katex-display")).toHaveLength(1);
  });

  it("shows math KaTeX can't parse as raw TeX in the muted box", () => {
    const page = show("Bad \\(\\frac{1}{\\)");
    expect(page.querySelector(".katex-error")?.textContent).toContain("\\frac{1}{");
  });

  it("opens the source when a citation is clicked", () => {
    const onCite = vi.fn();
    show("Claim [S2].", onCite);
    fireEvent.click(screen.getByTestId("citation"));
    expect(onCite).toHaveBeenCalledWith(2);
  });

  it("renders a table with math in it", () => {
    const page = show("| a | b |\n|---|---|\n| \\(|x|\\) | c |");
    expect(page.querySelector("td .katex")).not.toBeNull();
  });

  it("puts a code block in the copyable block, plain while it streams", () => {
    show("```python\nprint(1)\n```", vi.fn(), true);
    expect(screen.getByTestId("code-block").textContent).toContain("print(1)");
    expect(screen.getByTestId("code-block").textContent).toContain("Copy");
  });

  it("doesn't turn raw HTML from the model into elements", () => {
    const page = show("<img src=x onerror=alert(1)>");
    expect(page.querySelector("img")).toBeNull();
  });

  it("opens ordinary links in a new tab", () => {
    expect(show("[site](https://example.com)").querySelector("a")?.target).toBe("_blank");
  });
});

describe("KaTeX", () => {
  it("uses one version for the HTML and the CSS, or scripts render at the wrong size", () => {
    const fromRoot = createRequire(`${process.cwd()}/package.json`);
    const fromRehype = createRequire(fromRoot.resolve("rehype-katex"));
    expect(fromRehype.resolve("katex")).toBe(fromRoot.resolve("katex"));
  });
});

describe("Markdown and the web (Phase 5d)", () => {
  it("loads only the app's own images, so an answer can't leak through an image address", () => {
    const page = show("![chart](/api/outputs/plot.png) and ![x](https://track.example-collector.net/p.png?q=secret)");
    expect(page.querySelectorAll("img")).toHaveLength(1);
    expect(page.querySelector("img")?.getAttribute("src")).toBe("/api/outputs/plot.png");
    expect(screen.getByTestId("blocked-image").textContent).toBe("[image from track.example-collector.net not loaded]");
  });

  it("links [Web n] to its page, in a new tab", () => {
    const page = render(<Markdown text="The record [Web 2]." streaming={false} onCite={vi.fn()}
                                  webUrl={(n) => (n === 2 ? "https://www.hutter1.net/prize/index.htm" : undefined)} />).container;
    const link = page.querySelector("[data-testid=web-citation]")!;
    expect(link.textContent).toBe("Web 2");
    expect(link.getAttribute("href")).toBe("https://www.hutter1.net/prize/index.htm");
    expect(link.getAttribute("target")).toBe("_blank");
  });

  it("leaves a [Web n] it can't place as plain text", () => {
    show("Claim [Web 7].");
    expect(screen.queryByTestId("web-citation")).toBeNull();
    expect(screen.getByText("Web 7")).toBeTruthy();
  });
});
