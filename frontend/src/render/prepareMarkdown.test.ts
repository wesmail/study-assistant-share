import { describe, expect, it } from "vitest";
import { prepareMarkdown as prepare } from "./prepareMarkdown";

describe("math delimiters", () => {
  it("turns \\( \\) into $", () => {
    expect(prepare("Let \\(x_n\\) be it")).toBe("Let $x_n$ be it");
  });

  it("turns \\[ \\] into a $$ block", () => {
    expect(prepare("\\[\ns = 1\n\\]")).toBe("$$\ns = 1\n$$");
  });

  it("puts a display formula on its own lines when text comes first", () => {
    expect(prepare("We get \\[ s = 1 \\]\nDone")).toBe("We get \n\n$$\ns = 1\n$$\nDone");
  });

  it("starts a new paragraph when text follows a display formula", () => {
    expect(prepare("\\[ s \\] and more")).toBe("$$\ns\n$$\n\n and more");
  });

  it("moves single-line $$ onto its own lines, or remark-math renders it inline", () => {
    expect(prepare("$$a=b$$")).toBe("$$\na=b\n$$");
  });

  it("keeps a display formula inside its list item", () => {
    expect(prepare("1. Step\n   \\[ s \\]")).toBe("1. Step\n   $$\n   s\n   $$");
  });

  it("indents past a list marker on the same line", () => {
    expect(prepare("- \\[ s \\]")).toBe("- $$\n  s\n  $$");
  });

  it("keeps a display formula inside a block quote", () => {
    expect(prepare("> \\[ s \\]")).toBe("> $$\n> s\n> $$");
  });

  it("leaves a LaTeX line break alone", () => {
    const text = "\\[\\begin{aligned} a \\\\[2pt] b \\end{aligned}\\]";
    expect(prepare(text)).toContain("a \\\\[2pt] b");
    expect(prepare(text).startsWith("$$\n")).toBe(true);
  });

  it("handles the tables gpt-oss writes, with pipes in the math", () => {
    expect(prepare("| \\(|x|\\) | norm |")).toBe("| $\\vert x\\vert$ | norm |");
    expect(prepare("| \\(\\|x\\|\\) | norm |")).toBe("| $\\Vert x\\Vert$ | norm |");
  });

  it("keeps pipes outside tables", () => {
    expect(prepare("So \\(|x|\\) holds")).toBe("So $|x|$ holds");
  });

  it("leaves an unfinished formula alone while it streams", () => {
    expect(prepare("We get \\[ s = ")).toBe("We get \\[ s = ");
  });
});

describe("what must stay untouched", () => {
  it.each([
    "```python\nx = '\\(a\\)'\n```",
    "```\nunfinished \\[ a \\]",
    "Use `\\(x\\)` here",
    "$a$ and $b$",
    "$$\na = b\n$$",
  ])("%s", (text) => {
    expect(prepare(text)).toBe(text);
  });

  it("doesn't take prices for math", () => {
    expect(prepare("It costs $5 and then $6.")).toBe("It costs $5 and then $6.");
  });
});

describe("citations", () => {
  it("links a citation", () => {
    expect(prepare("Claim [S2].")).toBe("Claim [S2](#cite-2).");
  });

  it("links neighbouring citations separately", () => {
    expect(prepare("[S1][S3]")).toBe("[S1](#cite-1)[S3](#cite-3)");
  });

  it("leaves citations in code alone", () => {
    expect(prepare("`[S1]`")).toBe("`[S1]`");
  });

  it("leaves citation-like text in math alone", () => {
    expect(prepare("\\(x[S1]\\)")).toBe("$x[S1]$");
  });

  it("moves a citation out of a formula's tag and links it", () => {
    expect(prepare("\\[x \\tag{S2}\\]")).toBe("$$\nx\n$$\n\n[S2](#cite-2)");
    expect(prepare("$$x \\tag{S1, S5}$$")).toBe("$$\nx\n$$\n\n[S1](#cite-1) [S5](#cite-5)");
  });

  it("keeps a tag's detail as the formula's label", () => {
    expect(prepare("\\[x \\tag{S1: 4.106}\\]")).toBe("$$\nx \\tag{4.106}\n$$\n\n[S1](#cite-1)");
  });

  it("leaves a real equation number alone", () => {
    expect(prepare("\\[x \\tag{4.106}\\]")).toBe("$$\nx \\tag{4.106}\n$$");
  });

  it("keeps a moved citation inside its list item", () => {
    expect(prepare("- \\[x \\tag{S2}\\]")).toContain("\n\n  [S2](#cite-2)");
  });

  it("links a citation the model wrapped in math", () => {
    expect(prepare("Claim. \\([S2]\\)")).toBe("Claim. [S2](#cite-2)");
    expect(prepare("Claim. \\( [S1] [S3] \\)")).toBe("Claim. [S1](#cite-1) [S3](#cite-3)");
    expect(prepare("Claim. $[S2]$")).toBe("Claim. [S2](#cite-2)");
  });

  it("links a citation right after a formula", () => {
    expect(prepare("So \\(a\\) [S1].")).toBe("So $a$ [S1](#cite-1).");
  });
});
